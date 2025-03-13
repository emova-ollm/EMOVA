from abc import ABC
from functools import partial

import torch
import torch.nn as nn

from emova.registry_utils import build_from_cfg

from einops import rearrange, repeat
from emova.registry_utils import Registry

MM_PROJECTOR = Registry('mm_projector')


class BaseMMProjector(ABC, nn.Module):
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def tune(self):
        for p in self.parameters():
            p.requires_grad = True

    @property
    def downsample_rate(self):
        return 1

    @property
    def downsample_rate_per_side(self):
        return 1


class LambdaLayer(nn.Module):
    def __init__(self, fn):
        super(LambdaLayer, self).__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


@MM_PROJECTOR.register_module()
class IdentityMap(BaseMMProjector):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


@MM_PROJECTOR.register_module()
class SimpleResBlock(BaseMMProjector):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


@MM_PROJECTOR.register_module()
class MLPProjector(nn.Sequential, BaseMMProjector):
    def __init__(self, mm_hidden_size, hidden_size, mlp_depth=2):
        modules = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        super(MLPProjector, self).__init__(*modules)


class BaseDownsamplerMMProjector(BaseMMProjector):
    def __init__(self, mm_hidden_size,
                 hidden_size,
                 mlp_depth=2,
                 downsample_rate=4,
                 downsample_size=None,
                 num_input_token=576,
                 add_pos_embed=False,
                 add_pre_norm=False,
                 **kwargs
                 ):
        super(BaseDownsamplerMMProjector, self).__init__()
        self._downsample_rate = downsample_rate
        self.downsample_size = (downsample_size, downsample_size) \
            if isinstance(downsample_size, int) else downsample_size
        assert num_input_token // self.downsample_rate == self.downsample_size[0] * self.downsample_size[1]

        self.downsampler = self.build_downsampler(mm_hidden_size, **kwargs)
        self.mlp = self.build_mlp(mm_hidden_size, hidden_size, mlp_depth)

        self.norm = nn.LayerNorm(mm_hidden_size) if add_pre_norm else None

        self.pos_emb = nn.Parameter(
            torch.randn(num_input_token, mm_hidden_size) * 0.02
        ) if add_pos_embed else None

    def build_mlp(self, mm_hidden_size, hidden_size, mlp_depth):
        modules = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        return nn.Sequential(*modules)

    def build_downsampler(self, mm_hidden_size, **kwargs):
        raise NotImplementedError

    def forward_downsampler(self, x):
        raise NotImplementedError

    def forward(self, x):

        if self.norm is not None:
            x = self.norm(x)

        if self.pos_emb is not None:
            x = x + self.pos_emb.unsqueeze(0)

        x = self.forward_downsampler(x)

        return self.mlp(x)

    @property
    def downsample_rate(self):
        return self._downsample_rate

    @property
    def downsample_rate_per_side(self):
        res = self._downsample_rate ** 0.5
        if res.is_integer():
            return int(res)
        else:
            return res


@MM_PROJECTOR.register_module()
class AvgPoolMMProjector(BaseDownsamplerMMProjector):
    def build_downsampler(self, mm_hidden_size, ):
        return nn.AdaptiveAvgPool2d(self.downsample_size)

    def forward_downsampler(self, x):
        h = x.size(1) ** 0.5
        assert h.is_integer()
        x = rearrange(x, 'b (h w) c -> b c h w', h=int(h))
        x = self.downsampler(x)
        x = rearrange(x, 'b c h w -> b (h w) c ')
        return x


@MM_PROJECTOR.register_module()
class CAbstractorMMProjector(BaseDownsamplerMMProjector):
    def build_downsampler(self, mm_hidden_size,
                          conv_hidden_size=1024, conv_block_depth=3):
        from timm.models.regnet import RegStage
        try:
            from timm.layers import LayerNorm2d
        except:
            from timm.models.layers import LayerNorm2d

        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        s1 = RegBlock(
            conv_block_depth,
            mm_hidden_size,
            conv_hidden_size,
        )
        sampler = nn.AdaptiveAvgPool2d(self.downsample_size)
        s2 = RegBlock(
            conv_block_depth,
            conv_hidden_size,
            conv_hidden_size,
        )
        self.conv_hidden_size = conv_hidden_size
        return nn.Sequential(
            s1,
            sampler,
            s2,
        )

    def build_mlp(self, mm_hidden_size, hidden_size, mlp_depth):
        modules = [nn.Linear(self.conv_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        return nn.Sequential(*modules)

    def forward_downsampler(self, x):
        h = x.size(1) ** 0.5
        assert h.is_integer()
        x = rearrange(x, 'b (h w) c -> b c h w', h=int(h))

        x = self.downsampler(x)

        x = rearrange(x, 'b c h w -> b (h w) c ')
        return x


@MM_PROJECTOR.register_module()
class ConcatChannelMMProjector(BaseDownsamplerMMProjector):
    def build_downsampler(self, mm_hidden_size, **kwargs):
        return LambdaLayer(lambda x: rearrange(x, 'b (h p w l) c -> b (h w) (p l c)',
                                               h=int((x.size(1) // self.downsample_rate) ** 0.5),
                                               p=self.downsample_rate_per_side,
                                               l=self.downsample_rate_per_side))

    def build_mlp(self, mm_hidden_size, hidden_size, mlp_depth):
        modules = [nn.Linear(mm_hidden_size * self.downsample_rate, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        return nn.Sequential(*modules)

    def forward_downsampler(self, x):
        x = self.downsampler(x)
        return x


def build_mm_projector(mm_projector_cfg, **kwargs):
    mm_projector_cfg.update(kwargs)
    trainable = mm_projector_cfg.pop('trainable', True)
    model = build_from_cfg(mm_projector_cfg, MM_PROJECTOR)

    if trainable:
        model.tune()
    else:
        model.freeze()
    return model

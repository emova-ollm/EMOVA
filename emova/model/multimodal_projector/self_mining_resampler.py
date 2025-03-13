import torch
from torch import nn
from torch import einsum

from emova.model.multimodal_projector.builder import MM_PROJECTOR, BaseDownsamplerMMProjector

from einops import rearrange, repeat
from einops_exts import rearrange_many


class CrossAttentionDropOutProj(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8, kv_concat_x=False):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)

        self.kv_concat_x = kv_concat_x

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

    def forward(self, x, context):
        x = self.norm_media(x)
        context = self.norm_context(context)

        h = self.heads

        if self.kv_concat_x:
            kv_input = torch.cat([context, x], dim=1)
        else:
            kv_input = context

        q = self.to_q(x)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)

        q = q * self.scale

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return out


@MM_PROJECTOR.register_module()
class SelfMiningMMProjector(BaseDownsamplerMMProjector):
    def build_downsampler(self, mm_hidden_size, kv_concat_x=True, **kwargs):
        return CrossAttentionDropOutProj(dim=mm_hidden_size,
                                         heads=mm_hidden_size // 64,
                                         kv_concat_x=kv_concat_x)

    def forward_downsampler(self, x):
        h = int((x.size(1) // self.downsample_rate) ** 0.5)
        x = rearrange(x, 'b (h p w k) c -> (b h w) (p k) c',
                      h=h,
                      p=self.downsample_rate_per_side,
                      k=self.downsample_rate_per_side)

        x = self.downsampler(x.mean(1, keepdims=True), x)

        x = rearrange(x, '(b h w) 1 c -> b (h w) c', h=h, w=h)
        return x

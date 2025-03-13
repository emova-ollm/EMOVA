import torch
from transformers import CLIPImageProcessor

from emova.model.multimodal_encoder.builder import VISION_TOWER, BaseModalityEncoder

from .unit import UNITModel


@VISION_TOWER.register_module()
class UNITVisionTower(BaseModalityEncoder):
    VISION_MODEL_OBJ_CLS = UNITModel
    VISION_MODEL_CONFIG_OBJ_CLS = None
    IMAGEPROCESSOR_OBJ_CLS = CLIPImageProcessor

    def __init__(self,
                 pretrained_model_name_or_path,
                 delay_load=False,
                 unfreeze_mm_vision_tower=False,
                 vision_processor_name="",
                 load_norm=False,
                 multi_reso=448,
                 downsample_method="conv",
                 **kwargs,
                 ):
        super().__init__()

        self.is_loaded = False
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.load_norm = load_norm

        self.hparam = kwargs
        self.vision_processor_name = vision_processor_name

        self.multi_reso = multi_reso
        self.downsample_method = downsample_method

        class DotDict(dict):
            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

        self._config = DotDict()
        self._config.hidden_size = 1280
        self._config.image_size = 448
        self._config.patch_size = 14

        if not delay_load:
            self.load_model()
        elif unfreeze_mm_vision_tower:
            self.load_model()
            self.enable_gradient_checkpointing()
        # else:
        #     self.cfg_only = AutoConfig.from_pretrained(self.pretrained_model_name_or_path)

    def load_model(self, device_map=None):
        self.is_loaded = True

        self.vision_tower = self.VISION_MODEL_OBJ_CLS.from_pretrained(
            self.pretrained_model_name_or_path, device_map=device_map)

        high_resolution = self.hparam.get('high_resolution', None)
        self.high_resolution = high_resolution

        self.image_processor = self.IMAGEPROCESSOR_OBJ_CLS.from_pretrained(self.pretrained_model_name_or_path)
        self.norm = torch.nn.LayerNorm(self._config.hidden_size)

    def enable_gradient_checkpointing(self):
        self.vision_tower.unit_model.model.set_grad_checkpointing()

    def forward(self, images, images_multi_reso=[None], multi_reso_flags=[1]):
        with torch.cuda.amp.autocast(dtype=self.dtype):
            self.vision_tower.eval()  # avoid the cpe modified during training.
            summary_feature, image_feature = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
            image_feature = self.norm(image_feature)

        image_feature = image_feature.to(images.dtype)
        return image_feature

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.unit_model.model.fc_norm.weight.dtype

    @property
    def device(self):
        return self.vision_tower.unit_model.model.fc_norm.weight.device

    @property
    def config(self):
        return self._config

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    def tune(self):
        if self.hparam.get('tune_vit_from_layer', None):
            print(f"Tuning vit from layer {self.hparam.get('tune_vit_from_layer')}")
            for n, p in self.vision_tower.named_parameters():
                if 'unit_model.model.blocks.' in n:
                    layer_id = int(
                        n.split('unit_model.model.blocks.')[-1].split('.')[0])
                    if layer_id >= self.hparam.get('tune_vit_from_layer'):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                else:
                    p.requires_grad = False
        else:
            super(UNITVisionTower, self).tune()

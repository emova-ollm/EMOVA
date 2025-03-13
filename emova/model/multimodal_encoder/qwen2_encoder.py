import torch

from emova.model.multimodal_encoder.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from emova.model.multimodal_encoder.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel

from emova.model.multimodal_encoder.builder import VISION_TOWER, BaseModalityEncoder


@VISION_TOWER.register_module()
class Qwen2VisionTower(BaseModalityEncoder):
    VISION_MODEL_OBJ_CLS = Qwen2VisionTransformerPretrainedModel
    VISION_MODEL_CONFIG_OBJ_CLS = None
    IMAGEPROCSSOR_OBJ_CLS = Qwen2VLImageProcessor

    def __init__(self,
                 pretrained_model_name_or_path,
                 min_pixels=4 * 28 * 28,
                 max_pixels=16384 * 28 * 28,  # default range token num per image: 4-16384
                 base_resolution=448,
                 delay_load=False,
                 unfreeze_mm_vision_tower=False,
                 **kwargs, ):
        super().__init__()
        self.is_loaded = False

        self.delay_load = delay_load

        self.model_name_or_path = pretrained_model_name_or_path
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.base_resolution = base_resolution
        self.default_grid_h, self.default_grid_w = base_resolution // 14, base_resolution // 14
        self.unfreeze_mm_vision_tower = unfreeze_mm_vision_tower
        self.hparam = kwargs

        if not delay_load:
            self.load_model()
        if unfreeze_mm_vision_tower:
            self.load_model()
            self.enable_gradient_checkpointing()

    def enable_gradient_checkpointing(self):
        self.vision_tower.gradient_checkpointing = True

    def tune(self):
        if self.hparam.get('tune_vit_from_layer', None):
            print(f"Tuning vit from layer {self.hparam.get('tune_vit_from_layer')}")
            for n, p in self.vision_tower.named_parameters():
                if 'blocks.' in n:
                    layer_id = int(
                        n.split('blocks.')[-1].split('.')[0])
                    if layer_id >= self.hparam.get('tune_vit_from_layer'):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                elif 'merger' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        else:
            super(Qwen2VisionTower, self).tune()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.model_name_or_path))
            return
        self.image_processor = self.IMAGEPROCSSOR_OBJ_CLS.from_pretrained(self.model_name_or_path,
                                                                          min_pixels=self.min_pixels,
                                                                          max_pixels=self.max_pixels)
        self.vision_tower = self.VISION_MODEL_OBJ_CLS.from_pretrained(self.model_name_or_path, device_map=device_map)

        self.is_loaded = True

    def forward(self, pixel_values, image_grid_thw):
        image_features = self.vision_tower(pixel_values, image_grid_thw)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

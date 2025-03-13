import torch

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

from emova.model.multimodal_encoder.builder import VISION_TOWER, BaseModalityEncoder


def exists(x):
    return x is not None


@VISION_TOWER.register_module()
class CLIPVisionTower(BaseModalityEncoder):
    IMAGEPROCSSOR_OBJ_CLS = CLIPImageProcessor
    VISION_MODEL_OBJ_CLS = CLIPVisionModel
    VISION_MODEL_CONFIG_OBJ_CLS = CLIPVisionConfig

    def __init__(self,
                 pretrained_model_name_or_path,
                 mm_vision_select_layer=-1,
                 mm_vision_select_feature='patch',
                 delay_load=False,
                 unfreeze_mm_vision_tower=False,
                 **kwargs, ):
        super().__init__()
        self.is_loaded = False

        self.delay_load = delay_load

        self.model_name_or_path = pretrained_model_name_or_path
        self.select_layer = mm_vision_select_layer
        self.select_feature = mm_vision_select_feature
        self.unfreeze_mm_vision_tower = unfreeze_mm_vision_tower
        self.hparam = kwargs

        if not delay_load:
            self.load_model()
        if unfreeze_mm_vision_tower:
            self.load_model()
            self.enable_gradient_checkpointing()
        else:
            self.cfg_only = self.VISION_MODEL_CONFIG_OBJ_CLS.from_pretrained(self.model_name_or_path)

    def enable_gradient_checkpointing(self):
        self.vision_tower.vision_model.encoder.gradient_checkpointing = True

    def tune(self):
        if self.hparam.get('tune_vit_from_layer', None):
            print(f"Tuning vit from layer {self.hparam.get('tune_vit_from_layer')}")
            for n, p in self.vision_tower.named_parameters():
                if 'vision_model.encoder.layers.' in n:
                    layer_id = int(
                        n.split('vision_model.encoder.layers.')[-1].split('.')[0])
                    if layer_id >= self.hparam.get('tune_vit_from_layer'):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                else:
                    p.requires_grad = False
        else:
            super(CLIPVisionTower, self).tune()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.model_name_or_path))
            return

        self.image_processor = self.IMAGEPROCSSOR_OBJ_CLS.from_pretrained(self.model_name_or_path)
        self.vision_tower = self.VISION_MODEL_OBJ_CLS.from_pretrained(self.model_name_or_path, device_map=device_map)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                                      output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                                   output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

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
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

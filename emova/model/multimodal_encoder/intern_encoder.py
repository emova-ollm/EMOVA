from emova.model.multimodal_encoder.builder import VISION_TOWER, BaseModalityEncoder

from .intern_vit_6b.modeling_intern_vit import InternVisionModel
from .intern_vit_6b.configuration_intern_vit import InternVisionConfig
from .clip_encoder import CLIPVisionTower


@VISION_TOWER.register_module()
class InternVisionTower(CLIPVisionTower):
    VISION_MODEL_OBJ_CLS = InternVisionModel
    VISION_MODEL_CONFIG_OBJ_CLS = InternVisionConfig

    def enable_gradient_checkpointing(self):
        self.vision_tower.encoder.gradient_checkpointing = True

    def tune(self):
        if self.hparam.get('tune_vit_from_layer', None):
            for n, p in self.vision_tower.named_parameters():
                if 'encoder.layers.' in n:
                    layer_id = int(
                        n.split('encoder.layers.')[-1].split('.')[0])
                    if layer_id >= self.hparam.get('tune_vit_from_layer'):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                else:
                    p.requires_grad = False
        else:
            BaseModalityEncoder.tune(self)

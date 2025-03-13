import torch
from torch import nn

from emova.model.multimodal_encoder.builder import VISION_TOWER, BaseModalityEncoder
from transformers import SiglipVisionConfig, SiglipVisionModel

from .clip_encoder import CLIPVisionTower

from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
from typing import Optional, Tuple, Union, Dict
from functools import partial, reduce
from PIL import Image


class SiglipImageProcessor:
    def __init__(self, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5), size={"shortest_edge": 384},
                 crop_size: Dict[str, int] = None, resample=PILImageResampling.BICUBIC, rescale_factor=1 / 255,
                 data_format=ChannelDimension.FIRST):
        crop_size = crop_size if crop_size is not None else {"height": 384, "width": 384}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size if size is not None else {"shortest_edge": 384}
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, images, return_tensors):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            assert isinstance(images, list)

        size = (self.size["shortest_edge"], self.size["shortest_edge"])
        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(resize, size=size, resample=self.resample, data_format=self.data_format),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(normalize, mean=self.image_mean, std=self.image_std, data_format=self.data_format),
            partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)
        data = {"pixel_values": images}

        return BatchFeature(data=data, tensor_type=return_tensors)


@VISION_TOWER.register_module()
class SigLipVisionTower(CLIPVisionTower):
    VISION_MODEL_OBJ_CLS = SiglipVisionModel
    VISION_MODEL_CONFIG_OBJ_CLS = SiglipVisionConfig
    IMAGEPROCSSOR_OBJ_CLS = SiglipImageProcessor

    def load_model(self, device_map=None):
        if self.is_loaded:
            return

        # align with LLaVA-Next
        self.image_processor = self.IMAGEPROCSSOR_OBJ_CLS()
        self.vision_tower = self.VISION_MODEL_OBJ_CLS.from_pretrained(self.model_name_or_path, device_map=device_map)

        del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.vision_model.head = nn.Identity()
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                                      output_hidden_states=True)
                image_feature = image_forward_out.hidden_states[-1].to(image.dtype)
                assert image_features.shape[-2] == 729
                image_features.append(image_feature)
        else:
            images = images.to(device=self.device, dtype=self.dtype)
            image_forward_outs = self.vision_tower(images, output_hidden_states=True)
            image_features = image_forward_outs.hidden_states[-1].to(images.dtype)
            assert image_features.shape[-2] == 729

        return image_features

#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import random
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_mm_projector

from emova.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN

from emova.mm_utils import get_anyres_image_grid_shape

from emova.model.utils import load_state_dict_maybe_zero_3
from emova.utils import rank0_print, local_rank

from ..utils import get_state_maybe_zero_3, dicts_equal


class EmovaMetaModel:
    def __init__(self, config):
        super(EmovaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config.mm_vision_tower, delay_load=True)
            self.mm_projector = build_mm_projector(config.mm_projector,
                                                   mm_hidden_size=self.config.mm_hidden_size,
                                                   hidden_size=self.config.hidden_size)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        mm_patch_merge_type = model_args.mm_patch_merge_type
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        # All trainables states in pretrain stage.
        pretrain_trainables = model_args.get('pretrain_trainables', None)

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args.mm_vision_tower)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        elif hasattr(self.config, 'mm_vision_tower') and \
                not dicts_equal(dict(self.config.mm_vision_tower), dict(model_args.mm_vision_tower),
                                ignore_keys=['delay_load', 'trainable']):
            if dicts_equal(dict(self.config.mm_vision_tower), dict(model_args.mm_vision_tower),
                           ignore_keys=['delay_load', 'trainable', 'tune_vit_from_layer', 'max_pixels', 'min_pixels']):
                rank0_print('Warning: Only tune_vit_from_layer is differnt. It won\'t re-init your vit!!!!!!!!!!!!!!!!')

                self.vision_tower.hparam['tune_vit_from_layer'] = model_args.mm_vision_tower['tune_vit_from_layer'] \
                    if 'tune_vit_from_layer' in model_args.mm_vision_tower else None
                if model_args.mm_vision_tower['trainable']:
                    self.vision_tower.tune()
                else:
                    self.vision_tower.freeze()
                vision_tower = self.vision_tower
            else:
                rank0_print(
                    'Warning: vit config doesn\'t match the pretrained config. Initializing again your vit!!!!!!!!!!!!!!!!')

                # model is not the same.
                if self.config.mm_vision_tower['pretrained_model_name_or_path'] == model_args.mm_vision_tower[
                    'pretrained_model_name_or_path']:
                    state_dict = get_state_maybe_zero_3(self.get_vision_tower().named_parameters())
                    del self.vision_tower
                    vision_tower = build_vision_tower(model_args.mm_vision_tower)
                    msg = load_state_dict_maybe_zero_3(vision_tower,
                                                       state_dict,
                                                       strict=False)
                    rank0_print('[Info] Messege of reloading vit\'s state dict:', msg)
                else:
                    vision_tower = build_vision_tower(model_args.mm_vision_tower)

                if fsdp is not None and len(fsdp) > 0:
                    self.vision_tower = [vision_tower]
                else:
                    self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_vision_tower = model_args.mm_vision_tower
        self.config.mm_projector = model_args.mm_projector
        self.config.mm_patch_merge_type = mm_patch_merge_type
        self.config.mm_hidden_size = vision_tower.hidden_size

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_mm_projector(model_args.mm_projector,
                                                   mm_hidden_size=self.config.mm_hidden_size,
                                                   hidden_size=self.config.hidden_size)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=torch.float32))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            mm_projector_state = get_w(mm_projector_weights, 'mm_projector')
            msg = self.mm_projector.load_state_dict(mm_projector_state)
            msg_path = os.path.abspath('emova_load_mm_projector_state_msg.log')
            rank0_print(f'The message info of loading mm projector weights saves in {msg_path}')
            print(msg, file=open(msg_path, 'w'))
            print(f"State keys: {mm_projector_state.keys()}", file=open(msg_path, 'a'))

        if pretrain_trainables is not None:
            pretrain_trainable_weights = torch.load(pretrain_trainables, map_location='cpu')

            def get_w2(weights, keyword):
                return {k.split(keyword + '.')[1] if k.startswith(keyword) else k: v
                        for k, v in weights.items()}

            msg = load_state_dict_maybe_zero_3(self,
                                               get_w2(get_w2(pretrain_trainable_weights, 'base_model.model.model'),
                                                      'model'),
                                               strict=False)
            if msg is not None:
                assert len(msg.unexpected_keys) == 0, msg
            msg_path = os.path.abspath('emova_load_pretrain_trainable_state_msg.log')
            rank0_print(f'The message info of loading pretrain trainables state in {msg_path}')
            rank0_print(msg, file=open(msg_path, 'w'))
            rank0_print(f"State keys: {pretrain_trainable_weights.keys()}", file=open(msg_path, 'a'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class EmovaMetaForCausalLM(ABC):
    _skip_names = ['mm_projector', 'vision_tower', 'image_newline']
    mask_hires_vision_tokens_p = 0.

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_mm_projector(self):
        return self.get_model().mm_projector

    def tune(self):
        for n, p in self.named_parameters():
            if not any(skip_name in n
                       for skip_name in self._skip_names):
                p.requires_grad_(True)

    def freeze(self):
        for n, p in self.named_parameters():
            if not any(skip_name in n
                       for skip_name in self._skip_names):
                p.requires_grad_(False)

    def encode_images(self, images, image_sizes):
        image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
        mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')

        if image_aspect_ratio == "native_anyres":
            spatial_merge_size = self.get_vision_tower().vision_tower.spatial_merge_size
            image_features = self.get_model().get_vision_tower()(images, image_sizes)
            image_features = image_features.unsqueeze(dim=0)
            image_features = self.get_model().mm_projector(image_features)
            image_features = image_features.squeeze(dim=0)
            new_image_features = []
            start_idx = 0
            for image_size in image_sizes:
                N = image_size[0] * image_size[1] * image_size[2] // spatial_merge_size // spatial_merge_size
                cur_image_features = image_features[start_idx:start_idx + N]
                new_image_features.append(cur_image_features)
                start_idx += N
            image_features = new_image_features
        else:
            image_features = self.get_model().get_vision_tower()(images)
            image_features = self.get_model().mm_projector(image_features)
        return image_features

    def set_mask_hires_vision_tokens_probability(self, p):
        self.mask_hires_vision_tokens = p

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels,
            images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)

            batch_size = 36
            if len(concat_images) > batch_size:
                num_batches = len(concat_images) // batch_size + (1 if len(concat_images) % batch_size > 0 else 0)
                image_features = []
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(concat_images))
                    batch_images = concat_images[start_idx:end_idx]
                    batch_features = self.encode_images(batch_images, image_sizes)
                    image_features.append(batch_features)
                image_features = torch.cat(image_features, dim=0)
            else:
                image_features = self.encode_images(concat_images, image_sizes)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        downsample_per_side = self.get_mm_projector().downsample_rate_per_side
                        height = width = int(height / downsample_per_side)
                        assert height * width == base_image_feature.shape[0], (height, width, base_image_feature.shape)
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx],
                                                                                            self.config.image_grid_pinpoints,
                                                                                            self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(
                                    image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)

                        if self.mask_hires_vision_tokens_p > 0.:
                            if random.random() < self.mask_hires_vision_tokens_p:
                                image_feature = base_image_feature
                            else:
                                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)

                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images, image_sizes)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in
                     zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
                cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device,
                                   dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype,
                                       device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                              device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                             device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

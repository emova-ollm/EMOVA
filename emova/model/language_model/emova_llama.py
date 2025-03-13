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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
    LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers.cache_utils import Cache, DynamicCache, StaticCache

from .builder import LANGUAGE_MODEL
from ..emova_arch import EmovaMetaModel, EmovaMetaForCausalLM


class EmovaConfig(LlamaConfig):
    model_type = "emova_llama"


class EmovaLlamaModel(EmovaMetaModel, LlamaModel):
    config_class = EmovaConfig

    def __init__(self, config: LlamaConfig):
        super(EmovaLlamaModel, self).__init__(config)

    # def _update_causal_mask(
    #         self,
    #         attention_mask: torch.Tensor,
    #         input_tensor: torch.Tensor,
    #         cache_position: torch.Tensor,
    #         past_key_values,
    #         output_attentions: bool,
    # ):
    #     # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    #     # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    #     # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    #     # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114
    #
    #     if self.config._attn_implementation == "flash_attention_2":
    #         if attention_mask is not None and 0.0 in attention_mask:
    #             return attention_mask
    #         return None
    #
    #     # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
    #     # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
    #     # to infer the attention mask.
    #     past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    #     using_static_cache = isinstance(past_key_values, StaticCache)
    #
    #     # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
    #     if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
    #         if AttentionMaskConverter._ignore_causal_mask_sdpa(
    #                 attention_mask,
    #                 inputs_embeds=input_tensor,
    #                 past_key_values_length=past_seen_tokens,
    #                 is_training=self.training,
    #         ):
    #             return None
    #
    #     dtype, device = input_tensor.dtype, input_tensor.device
    #     min_dtype = torch.finfo(dtype).min
    #     sequence_length = input_tensor.shape[1]
    #     if using_static_cache:
    #         target_length = past_key_values.get_max_length()
    #     else:
    #         target_length = (
    #             attention_mask.shape[-1]
    #             if isinstance(attention_mask, torch.Tensor)
    #             else past_seen_tokens + sequence_length + 1
    #         )
    #
    #     if attention_mask is not None and attention_mask.dim() == 4:
    #         # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
    #         if attention_mask.max() != 0:
    #             raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
    #         causal_mask = attention_mask
    #     else:
    #         causal_mask = torch.full(
    #             (sequence_length, target_length), fill_value=min_dtype, dtype=torch.float32, device=device
    #         )
    #         if sequence_length != 1:
    #             causal_mask = torch.triu(causal_mask, diagonal=1)
    #         causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    #         causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
    #         if attention_mask is not None:
    #             causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
    #             mask_length = attention_mask.shape[-1]
    #             padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
    #                 causal_mask.device)
    #             padding_mask = padding_mask == 0
    #             causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
    #                 padding_mask, min_dtype
    #             )
    #     if (
    #             self.config._attn_implementation == "sdpa"
    #             and attention_mask is not None
    #             and attention_mask.device.type == "cuda"
    #             and not output_attentions
    #     ):
    #         # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
    #         # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
    #         # Details: https://github.com/pytorch/pytorch/issues/110213
    #         causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
    #
    #     return causal_mask


@LANGUAGE_MODEL.register_module()
class EmovaLlamaForCausalLM(LlamaForCausalLM, EmovaMetaForCausalLM):
    config_class = EmovaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = EmovaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            image_sizes: Optional[List[List[int]]] = None,
            return_dict: Optional[bool] = None,
            cache_position=None,  # Required for inference
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        llm_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        return llm_output

    @torch.no_grad()
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            images: Optional[torch.Tensor] = None,
            image_sizes: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs


AutoConfig.register("emova_llama", EmovaConfig)
AutoModelForCausalLM.register(EmovaConfig, EmovaLlamaForCausalLM)

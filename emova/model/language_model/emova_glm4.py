from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .builder import LANGUAGE_MODEL
from .glm_4.configuration_chatglm import ChatGLMConfig
from .glm_4.modeling_chatglm import ChatGLMModel, ChatGLMForConditionalGeneration
from ..emova_arch import EmovaMetaModel, EmovaMetaForCausalLM

import torch
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union, List, Callable, Dict, Any

from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)


class EmoveGLM4Config(ChatGLMConfig):
    model_type = "emova_glm4"


class EmovaGLM4Model(EmovaMetaModel, ChatGLMModel):
    config_class = EmoveGLM4Config

    def __init__(self, config: EmoveGLM4Config):
        super(EmovaGLM4Model, self).__init__(config)

    @property
    def embed_tokens(self):
        return self.embedding


@LANGUAGE_MODEL.register_module()
class EmovaGLM4ForCausalLM(ChatGLMForConditionalGeneration, EmovaMetaForCausalLM):
    config_class = EmoveGLM4Config

    def __init__(self, config):
        super(ChatGLMForConditionalGeneration, self).__init__(config)
        self.max_sequence_length = config.max_length
        self.transformer = EmovaGLM4Model(config)
        self.config = config

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def lm_head(self):
        return self.transformer.output_layer

    def get_input_embeddings(self):
        return self.transformer.embedding.word_embeddings

    def get_model(self):
        return self.transformer

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


AutoConfig.register("emova_glm4", EmoveGLM4Config)
AutoModelForCausalLM.register(EmoveGLM4Config, EmovaGLM4ForCausalLM)

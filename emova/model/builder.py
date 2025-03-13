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
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from transformers.utils import is_torch_npu_available
import torch
from emova.model import *
from emova.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from emova.utils import smart_tokenizer_and_embedding_resize
from emova.model.language_model.builder import build_language_model


def load_pretrained_model(
        model_path, model_base, model_name,
        load_8bit=False, load_4bit=False, device_map="auto",
        device="cuda", use_flash_attn=False, config=None,
        torch_dtype=torch.float16,
        **kwargs):
    assert config is not None

    model_args = config.model_args

    kwargs = {"device_map": device_map, **kwargs}

    if isinstance(device, str) and device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch_dtype

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2' if not is_torch_npu_available() else 'sdpa'

    if 'emova' in model_name.lower():
        # Load EMOVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn(
                'There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
        if 'lora' in model_name.lower() and model_base is not None:
            from emova.model.language_model.emova_llama import EmovaConfig
            lora_cfg_pretrained = EmovaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading EMOVA from base model...')

            if 'llama3' in model_name.lower():
                lora_cfg_pretrained.pad_token_id = None
                lora_cfg_pretrained.vocab_size = lora_cfg_pretrained.vocab_size - 1

            model_args.language_model.update(pretrained_model_name_or_path=model_base,
                                             from_pretrained=True,
                                             from_scratch=False)
            model = build_language_model(model_args.language_model,
                                         config=lora_cfg_pretrained,
                                         low_cpu_mem_usage=True,
                                         **kwargs)

            if 'llama3' in model_name.lower():
                print(f"Adding pad token as '<pad>'")
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(pad_token="<pad>"),
                    tokenizer=tokenizer,
                    model=model,
                )

            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(
                    torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional EMOVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'),
                                                 map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')

                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
                                   non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in
                                       non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading EMOVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'),
                                    os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer_args = dict(use_fast=True)
                config_args = dict(trust_remote_code=True)
            else:
                tokenizer_args = dict(use_fast=False)
                config_args = dict(model_path)

            tokenizer = AutoTokenizer.from_pretrained(model_base, **tokenizer_args)
            cfg_pretrained = AutoConfig.from_pretrained(model_path, **config_args)
            kwargs = dict(low_cpu_mem_usage=True, config=cfg_pretrained, )
            model_args.language_model.update(pretrained_model_name_or_path=model_base,
                                             from_pretrained=True,
                                             from_scratch=False)
            model = build_language_model(model_args.language_model, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to() for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)

            if os.path.exists(os.path.join(model_path, 'trainables.bin')):
                trainables_weights = torch.load(os.path.join(model_path, 'trainables.bin'), map_location='cpu')
                trainables_weights = {k: v.to(torch_dtype) for k, v in trainables_weights.items()}
                msg = model.load_state_dict(trainables_weights, strict=False)
                print(msg)
        else:
            if 'mpt' in model_name.lower():
                tokenizer_args = dict(use_fast=False)
            elif 'mistral' in model_name.lower():
                tokenizer_args = dict()
            elif 'qwen2' in model_name.lower() or 'qwen-2' in model_name.lower():
                tokenizer_args = dict(padding_side="right",
                                      use_fast=True,)
            else:
                tokenizer_args = dict(use_fast=False)
            tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_args)
            model_args.language_model.update(pretrained_model_name_or_path=model_path,
                                             from_pretrained=True,
                                             from_scratch=False)

            model = build_language_model(model_args.language_model,
                                         low_cpu_mem_usage=True,
                                         **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch_dtype)
        else:
            tokenizer_args = dict(use_fast=False)
            model_from_pretrained_args = kwargs
            model_from_pretrained_args.update(low_cpu_mem_usage=True)

            if 'mpt' in model_name.lower():
                tokenizer_args.update(use_fast=True)
                model_from_pretrained_args.update(trust_remote_code=True)
            elif 'mistral' in model_name.lower():
                tokenizer_args = dict()
            elif 'qwen2' in model_name.lower() or 'qwen-2' in model_name.lower():
                tokenizer_args = dict(padding_side="right",
                                      use_fast=True, )
            else:
                tokenizer_args = dict(use_fast=False)

            tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_args)
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_from_pretrained_args)

    image_processor = None

    if 'emova' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch_dtype)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "tokenizer_model_max_length"):
        context_len = model.config.tokenizer_model_max_length
    else:
        print("'max_sequence_length' is not set. Default using 2048.")
        context_len = 2048

    return tokenizer, model, image_processor, context_len


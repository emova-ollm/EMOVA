# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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
import argparse
import os
from dataclasses import dataclass, field, fields
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers
import tokenizers

from emova.data.dataset import make_supervised_data_module
from emova.model.language_model.builder import build_language_model
from emova.model.language_model.deepseek_vl2 import DeepseekVLV2Processor
from emova.train.emova_trainer import EMOVATrainer

from emova import conversation as conversation_lib
from emova.model import *

from emova.utils import read_config, set_local_rank, rank0_print, smart_tokenizer_and_embedding_resize, \
    get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer, \
    find_all_linear_names
from packaging import version

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

try:
    import torch_npu
    from torch_npu.npu import amp
    from torch_npu.contrib import transfer_to_npu

    print('Successful import torch_npu')
except Exception as e:
    print(e)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    def __post_init__(self):
        super(TrainingArguments, self).__post_init__()


def parse_args():
    parser = argparse.ArgumentParser(description="EMOVA Args.")
    parser.add_argument("config", type=str, help='config')
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    args, unknown = parser.parse_known_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args, unknown


def read_args():
    args, unknown = parse_args()
    cfg = read_config(args.config)

    print(unknown)
    overrides = []
    for item in unknown:
        if item.startswith('--'):
            item = item.strip('--')
        k, v = item.split('=')
        overrides.append((k, v))

    # Apply overrides
    for key, value in overrides:
        try:
            cfg.set_nested(key, value)
        except AttributeError as e:
            print(f"Warning: {e}")

    fields_name = set([f.name for f in fields(TrainingArguments)])

    extract_args = dict()
    for k, v in cfg.training_args.items():
        if k in fields_name:
            extract_args[k] = v

    training_args = TrainingArguments(**extract_args)
    for k, v in cfg.training_args.items():
        if not hasattr(training_args, k):
            setattr(training_args, k, v)

    cfg.training_args = training_args

    return cfg


def train():
    cfg = read_args()

    model_args, data_args, training_args = cfg.model_args, cfg.data_args, cfg.training_args

    local_rank = training_args.local_rank
    set_local_rank(local_rank)
    rank0_print(model_args, data_args, training_args)

    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))

    training_args.tune_language_model = model_args.language_model.get('trainable', False)
    training_args.tune_vision_tower = model_args.mm_vision_tower.get('trainable', False)
    training_args.tune_mm_mlp_adapter = model_args.mm_projector.get('trainable', True)
    is_pretrain_stage = not training_args.tune_language_model

    model = build_language_model(model_args.language_model,
                                 cache_dir=training_args.cache_dir,
                                 **bnb_model_from_pretrained_args)

    model.config.use_cache = False

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    pretrained_model_name_or_path = model_args.language_model.pretrained_model_name_or_path
    if 'mpt' in pretrained_model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    elif 'glm' in pretrained_model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="left",
            use_fast=False,
            trust_remote_code=True,
        )
    elif 'qwen' in pretrained_model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="left",
            use_fast=True,
        )
    elif 'deepseek' in pretrained_model_name_or_path.lower():
        vl_chat_processor = DeepseekVLV2Processor.from_pretrained(pretrained_model_name_or_path)
        tokenizer = vl_chat_processor.tokenizer
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    elif model_args.version == "v1":
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    else:
        if tokenizer.pad_token is None:
            rank0_print(f"Adding pad token as '<pad>'")
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="<pad>"),
                tokenizer=tokenizer,
                model=model,
            )

        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["llama3"]
        rank0_print(f"Using conversation format: {conversation_lib.default_conversation.version}")

    if model_args.mm_vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_aspect_ratio == 'anyres':
            base_size = vision_tower.config.image_size

            if data_args.get('max_num_slices', None):
                from emova.utils import find_possible_grids
                max_num_slices = data_args.get('max_num_slices')
                grids = find_possible_grids(max_num_slices)
            else:
                if hasattr(data_args, 'grids'):
                    grids = data_args.grids
                else:
                    grids = [[1, 2], [2, 1], [2, 2], [3, 1], [1, 3], [4, 1], [1, 4]]
            rank0_print(f"Enabling any-resolution training. Grid: {grids}")
            model.config.image_grid_pinpoints = data_args.image_grid_pinpoints = [
                [g[0] * base_size, g[1] * base_size] for g in grids]

        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        if training_args.tune_vision_tower:
            model.get_vision_tower().tune()
        else:
            model.get_vision_tower().freeze()

        if training_args.tune_mm_mlp_adapter:
            model.get_mm_projector().tune()
        else:
            model.get_mm_projector().freeze()

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        rank0_print(
            f"==============================training_args.mm_projector_lr{training_args.mm_projector_lr}=============================")
        if hasattr(training_args, 'mm_vision_tower_lr'):
            rank0_print(
                f"==============================training_args.mm_vision_tower_lr{training_args.mm_vision_tower_lr}=============================")

        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    rank0_print(model)
    for n, p in model.named_parameters():
        if p.requires_grad:
            rank0_print("Trainable Param", n, p.dtype)
        else:
            rank0_print("Non-trainable Param", n, p.dtype)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)

    trainer = EMOVATrainer(model=model,
                           tokenizer=tokenizer,
                           args=training_args,
                           **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        rank0_print(f"Resuming from {training_args.output_dir}")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir,
                                       is_pretrain_stage=is_pretrain_stage)


if __name__ == "__main__":
    train()

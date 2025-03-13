_base_ = [
    '../../../_base_/models/llama3_1_internvit_anyres.py',
    '../../../_base_/datasets/emova_alignment_7M.py',
    '../../../_base_/training/default.py'
]

data_args = dict(
    image_aspect_ratio='anyres',
    max_num_slices=9,
)

model_args = dict(
    version="llama3",
    pretrain_mm_mlp_adapter="./logdir/emova-llama3_1-70b-internvit-anyres-9slices-pretrain/mm_projector.bin",

    language_model=dict(trainable=True,
                        attn_implementation="flash_attention_2",
                        pretrained_model_name_or_path='Emova-ollm/Meta-Llama-3.1-70B-Instruct_add_speech_token_4096_nostrip/'
                        ),
    mm_vision_tower=dict(trainable=True,
                         unfreeze_mm_vision_tower=True,
                         tune_vit_from_layer=24, ),
    mm_projector=dict(trainable=True)
)

training_args = dict(
    output_dir="./logdir/emova-llama3_1-70b-internvit-anyres-9slices-alignment/",
    deepspeed="./scripts/zero3.json",
    save_on_each_node=True,
    bf16=True,
    tf32=True,
    fp16=False,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,
    evaluation_strategy="no",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    learning_rate=1e-5,
    weight_decay=0.,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=1,
    model_max_length=4096,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    group_by_modality_length=True,
    report_to="tensorboard",

    seed=423,
)

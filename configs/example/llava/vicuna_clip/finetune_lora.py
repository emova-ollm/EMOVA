_base_ = [
    '../../../_base_/models/vicuna_clip.py',
    '../../../_base_/datasets/llava_665k_finetune.py',
    '../../../_base_/training/default.py'
]

model_args = dict(
    version="v1",

    pretrain_mm_mlp_adapter="./logdir/emova_llava-v1.5-7b-pretrain/mm_projector.bin",

    language_model=dict(trainable=True),
    mm_vision_tower=dict(trainable=False),
    mm_projector=dict(trainable=True)
)

training_args = dict(
    output_dir="./logdir/emova_llava-v1.5-7b-finetune-lora",
    deepspeed="./scripts/zero3.json",
    bf16=False,
    fp16=True,
    tf32=False,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    evaluation_strategy="no",
    save_strategy="steps",
    save_steps=50000,
    save_total_limit=1,
    learning_rate=2e-4,
    mm_projector_lr=2e-5,
    weight_decay=0.,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    model_max_length=2048,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    group_by_modality_length=True,
    report_to="tensorboard",
    lora_enable=True,
    lora_r=128,
    lora_alpha=256,
)

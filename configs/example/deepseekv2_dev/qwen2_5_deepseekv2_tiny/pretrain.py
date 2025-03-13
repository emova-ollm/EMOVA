_base_ = [
    '../../../_base_/models/qwen2_5_deepseekv2_tiny.py',
    '../../../_base_/datasets/llava_556k_pretrain.py',
    '../../../_base_/training/default.py'
]

data_args = dict(
    image_aspect_ratio='native_anyres',
)

model_args = dict(
    version="plain",
    # version="deepseek",

    language_model=dict(
        # attn_implementation="sdpa",
        trainable=False,
        from_scratch=True,
    ),
    mm_vision_tower=dict(
        max_pixels=4096 * 28 * 28,
        trainable=False,
    ),
    mm_projector=dict(trainable=True),
)

training_args = dict(
    output_dir="./logdir/emova-deepseekv2-tiny-qwen2vit600m-pretrain/",
    deepspeed="./scripts/zero2.json",
    bf16=True,
    tf32=True,
    fp16=False,

    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    evaluation_strategy="no",
    save_strategy="steps",
    save_steps=24000,
    save_total_limit=1,
    learning_rate=1e-3,
    weight_decay=0.,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=1,
    model_max_length=4096 * 2,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    report_to="tensorboard",

)

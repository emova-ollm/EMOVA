_base_ = [
    '../../../_base_/models/vicuna_clip.py',
    '../../../_base_/datasets/llava_558k_pretrain.py',
    '../../../_base_/training/default.py'
]

data_args = dict(
    image_aspect_ratio='anyres',
)

model_args = dict(
    version="plain",
    language_model=dict(trainable=False),
    mm_vision_tower=dict(trainable=False),
    mm_projector=dict(type='SelfMiningMMProjector',
                      downsample_rate=4,
                      downsample_size=(12, 12),
                      trainable=True)
)

training_args = dict(
    output_dir="./logdir/emova_llava_next-vicuna-7b-selfMining-pretrain/",
    deepspeed="./scripts/zero2.json",
    bf16=False,
    tf32=False,
    fp16=True,
    num_train_epochs=1,
    per_device_train_batch_size=16,
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
    model_max_length=4096,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    report_to="tensorboard",

)

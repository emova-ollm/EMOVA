_base_ = [
    '../../../_base_/models/qwen2_5_qwen2vit.py',
    '../../../_base_/datasets/emova_sft_4M.py',
    '../../../_base_/training/default.py'
]

data_args = dict(
    image_aspect_ratio='native_anyres',
)

model_args = dict(
    version="qwen2",

    language_model=dict(trainable=True,
                        attn_implementation="sdpa",
                        pretrained_model_name_or_path="./logdir/emova-qwen2_5-7b-qwen2vit600m_4096vokens-alignment-npu/",
                        ),
    mm_vision_tower=dict(
        max_pixels=4096 * 28 * 28,
        trainable=True,
        unfreeze_mm_vision_tower=True),
    mm_projector=dict(trainable=True)
)

training_args = dict(
    output_dir="./logdir/emova-qwen2_5-7b-qwen2vit600m_4096vokens-finetune-npu/",
    deepspeed="./scripts/zero3.json",
    bf16=True,
    tf32=False,
    fp16=False,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    evaluation_strategy="no",
    save_strategy="steps",
    save_steps=1000,
    save_on_each_node=True,
    save_total_limit=1,
    learning_rate=2e-5,
    mm_projector_lr=2e-5,
    mm_vision_tower_lr=2e-6,
    weight_decay=0.,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=1,
    model_max_length=4096 * 2,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    group_by_modality_length=True,
    report_to="tensorboard",
)

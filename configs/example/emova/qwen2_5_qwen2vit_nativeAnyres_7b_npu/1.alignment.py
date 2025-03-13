_base_ = [
    '../../../_base_/models/qwen2_5_qwen2vit.py',
    '../../../_base_/datasets/emova_alignment_7M.py',
    '../../../_base_/training/default.py'
]

data_args = dict(
    image_aspect_ratio='native_anyres',
)

model_args = dict(
    version="qwen2",
    pretrain_mm_mlp_adapter="./logdir/emova-qwen2_5-7b-qwen2vit600m-pretrain-npu/mm_projector.bin",

    language_model=dict(trainable=True, attn_implementation="sdpa",
                        pretrained_model_name_or_path='Emova-ollm/Qwen2.5-7B-Instruct_add_speech_token_4096_nostrip',
                        ),
    mm_vision_tower=dict(
        max_pixels=4096 * 28 * 28,
        unfreeze_mm_vision_tower=True,
        tune_vit_from_layer=16,
        trainable=True,
    ),
    mm_projector=dict(trainable=True)
)

training_args = dict(
    output_dir="./logdir/emova-qwen2_5-7b-qwen2vit600m-alignment-npu/",
    deepspeed="./scripts/zero3.json",
    save_on_each_node=True,
    bf16=True,
    tf32=False,
    fp16=False,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    evaluation_strategy="no",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    learning_rate=2e-5,
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

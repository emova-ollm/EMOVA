model_args = dict(
    version="qwen2",  # Note that, in pretrain stage, version='plain'.
    freeze_backbone=False,

    pretrain_mm_mlp_adapter=None,
    mm_use_im_start_end=False,
    mm_use_im_patch_token=False,

    mm_patch_merge_type='flat',

    language_model=dict(
        type='EmovaQwen2ForCausalLM',
        pretrained_model_name_or_path='Qwen/Qwen2.5-7B-Instruct',
        from_pretrained=True,
    ),
    mm_vision_tower=dict(
        type='Qwen2VisionTower',
        pretrained_model_name_or_path="Emova-ollm/qwen2vit600m/",
        trainable=False
    ),
    mm_projector=dict(
        type='MLPProjector',
        mlp_depth=2,
    ),
)

model_args = dict(
    version="v1",  # Note that, in pretrain stage, version='plain'.
    freeze_backbone=False,

    pretrain_mm_mlp_adapter=None,
    mm_use_im_start_end=False,
    mm_use_im_patch_token=False,

    mm_patch_merge_type='flat',

    language_model=dict(
        type='EmovaLlamaForCausalLM',
        pretrained_model_name_or_path='lmsys/vicuna-7b-v1.5',
        # _attn_implementation="sdpa",
        from_pretrained=True,
    ),
    mm_vision_tower=dict(
        type='CLIPVisionTower',
        pretrained_model_name_or_path='openai/clip-vit-large-patch14-336',
        mm_vision_select_layer=-2,
        mm_vision_select_feature='patch',
    ),
    mm_projector=dict(
        type='MLPProjector',
        mlp_depth=2,
    ),
)

model_args = dict(
    version="llama3",  # Note that, in pretrain stage, version='plain'.
    freeze_backbone=False,

    pretrain_mm_mlp_adapter=None,
    mm_use_im_start_end=False,
    mm_use_im_patch_token=False,

    mm_patch_merge_type='spatial_unpad',

    language_model=dict(
        type='EmovaLlamaForCausalLM',
        pretrained_model_name_or_path='meta-llama/Llama-3.1-8B-Instruct',
        # _attn_implementation="sdpa",
        from_pretrained=True,
    ),
    mm_vision_tower=dict(
        type='SigLipVisionTower',
        pretrained_model_name_or_path="google/siglip-so400m-patch14-384",
        unfreeze_mm_vision_tower=False,
    ),
    mm_projector=dict(
        type='CAbstractorMMProjector',
        conv_block_depth=2,
        downsample_rate=2.84,
        downsample_size=(16, 16),
        num_input_token=729,
        add_pos_embed=False,
        add_image_newline_embed=False,
    ),
)

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
        trainable=True,
    ),
    mm_vision_tower=dict(
        type='RADIODistillMultiVisionTower_1B',
        pretrained_model_name_or_path="yeeaa/UNIT_1B",
        vision_processor_name='openai/clip-vit-large-patch14-336',
        high_resolution=448,
        multi_reso=896,
        downsample_method="conv",
        load_norm=True,
        trainable=True,
    ),
    mm_projector=dict(
        type='CAbstractorMMProjector',
        add_pre_norm=True,
        conv_block_depth=2,
        downsample_rate=4,
        downsample_size=(16, 16),
        num_input_token=1024,
        add_pos_embed=False,
        trainable=True,
    ),
)

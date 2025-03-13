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
        type='InternVisionTower',
        pretrained_model_name_or_path='OpenGVLab/InternViT-6B-448px-V1-5',
        mm_vision_select_layer=-1,
        mm_vision_select_feature='patch',
    ),
    mm_projector=dict(type='CAbstractorMMProjector',
                      conv_block_depth=2,
                      downsample_rate=4,
                      downsample_size=(16, 16),
                      num_input_token=1024,
                      add_pos_embed=False)
)

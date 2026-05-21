_base_ = './ovtr_5_frame_train_val.py'

mcip_enable = True
mcip_detach_memory = True
mcip_memory_momentum = 0.8
mcip_use_semantic_memory = True
mcip_use_motion_ref = False
mcip_motion_momentum = 0.7
mcip_motion_scale_init = 0.0
mcip_max_memory_update = 0.05
mcip_max_residual_ratio = 0.05
mcip_motion_offset_cap = 0.02
mcip_semantic_topk = 5
mcip_gate_use_txt = False
debug_mcip = False

attention_protection_mode = "kl"
attention_protection_topk = 3
attention_protection_conf_thresh = 0.25

train_tracking_only = [
    'track_embed',
    'track_embed.memory_residual_adapter',
    'track_embed.mcip_img_obs_norm',
    'track_embed.mcip_img_memory_norm',
    'track_embed.mcip_sem_memory_norm',
    'track_embed.memory_inject_logit',
    'track_embed.motion_scale',
    'update_attn',
    'norm4',
    'decoder.layers.0',
]

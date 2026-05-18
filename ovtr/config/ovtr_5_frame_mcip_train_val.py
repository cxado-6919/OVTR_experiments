_base_ = './ovtr_5_frame_train_val.py'

mcip_enable = True
mcip_detach_memory = True
mcip_memory_momentum = 0.8
mcip_use_semantic_memory = True
mcip_use_motion_ref = True
mcip_motion_momentum = 0.7
mcip_motion_scale_init = 0.0
mcip_gate_use_txt = False
debug_mcip = False

attention_protection_mode = "kl"
attention_protection_topk = 3
attention_protection_conf_thresh = 0.25

train_tracking_only = [
    'track_embed',
    'track_embed.gate_mlp',
    'track_embed.memory_img_proj',
    'track_embed.memory_sem_proj',
    'track_embed.motion_scale',
    'update_attn',
    'norm4',
    'decoder.layers.0',
]

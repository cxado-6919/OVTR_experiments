_base_ = "./ovtr_5_frame_train_val.py"

# A4: A3 + gate-aware memory update.
use_ov_dptd = True
ov_dptd_use_historical_offsets = True
ov_dptd_fusion = "semantic_gate"
ov_dptd_id_path_text = "none"
ov_dptd_fuse_cti = False
ov_dptd_store_debug = True

use_dptd_semantic_memory = True
dptd_memory_min_gate = 0.2
dptd_memory_use_alignment_feature = True
dptd_memory_allow_untrained_visual_projection = False
dptd_memory_debug = True

use_dptd_semantic_gate = True
dptd_gate_mode = "heuristic"
dptd_gate_min_appearance = 0.1
dptd_gate_debug = True

ov_dptd_semantic_gate_id_proj_init = "small_random"
ov_dptd_semantic_gate_id_proj_init_std = 1e-3
ov_dptd_reinit_dead_semantic_gate_id_proj = False

# Inference-only; keep disabled during training.
use_dptd_update_suppression = False
use_dptd_semantic_update_suppression = False

use_checkpoint_track = False
use_transformer_ckpt = False

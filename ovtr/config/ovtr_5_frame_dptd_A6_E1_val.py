_base_ = "./ovtr_5_frame_train_val.py"

# A6 + E1: v6-loss checkpoint with inference update suppression.
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

use_dptd_losses = True
dptd_loss_ofa_consistency_weight = 0.02
dptd_loss_semantic_memory_weight = 0.01
dptd_loss_visual_memory_weight = 0.01
dptd_loss_offset_consistency_weight = 1e-4
dptd_loss_same_category_contrast_weight = 0.0
dptd_loss_store_debug = True

use_dptd_update_suppression = True
dptd_update_suppression_thresh = 0.4
use_dptd_semantic_update_suppression = False

use_checkpoint_track = False
use_transformer_ckpt = False

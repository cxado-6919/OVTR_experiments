_base_ = "./ovtr_5_frame_train_val.py"

# A2: DPTD core + historical offsets.
use_ov_dptd = True
ov_dptd_use_historical_offsets = True
ov_dptd_fusion = "linear_sum"
ov_dptd_id_path_text = "none"
ov_dptd_fuse_cti = False
ov_dptd_store_debug = True

use_dptd_semantic_memory = False
use_dptd_semantic_gate = False

# Inference-only; keep disabled during training.
use_dptd_update_suppression = False
use_dptd_semantic_update_suppression = False

use_checkpoint_track = False
use_transformer_ckpt = False

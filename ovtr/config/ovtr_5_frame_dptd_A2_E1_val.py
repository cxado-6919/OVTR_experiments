_base_ = "./ovtr_5_frame_train_val.py"

# A2 + E1: DPTD core with inference update suppression.
use_ov_dptd = True
ov_dptd_use_historical_offsets = True
ov_dptd_fusion = "linear_sum"
ov_dptd_id_path_text = "none"
ov_dptd_fuse_cti = False
ov_dptd_store_debug = True

use_dptd_semantic_memory = False
use_dptd_semantic_gate = False

use_dptd_update_suppression = True
dptd_update_suppression_thresh = 0.4
use_dptd_semantic_update_suppression = False

use_checkpoint_track = False
use_transformer_ckpt = False

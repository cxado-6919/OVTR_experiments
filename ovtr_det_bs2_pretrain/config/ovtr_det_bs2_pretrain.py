modelname = "OVTR_det"
backbone = 'resnet50'

dilation = False
position_embedding = "sine"
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
unic_layers = 0

pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0

num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
two_stage_type = "standard"
two_stage_bbox_embed_share = False
dec_pred_bbox_embed_share = True
transformer_activation = "relu"
max_text_len = 256
fusion_dropout = 0.0
fusion_droppath = 0.1
use_fusion_layer = False
use_checkpoint = False 
use_transformer_ckpt = False 
use_text_cross_attention = True
embed_init_tgt = True 

# scaling
prior_prob = 0.005
log_scale = 0.0
text_dim = 256

# prompt
Clip_text_embeddings = '../model_zoo/iou_neg5_ens.pth' # from DetPro
Clip_image_embeddings = '../model_zoo/clip_image_embedding_all.pt'

distribution_based_sampling = True
isolation_mask = True
run_inference = False

# freezing
initial_grad = True
initial_grad_allowed = ['patch2query',
                        'feature_align',
                        'input_align',
                        'fusion_layers',
                        'log_scale',
                        'bias_lang',
                        'bias0',
                        'ca_text',
                        'catext_norm',
                        ]
global_grad_allowed_epoch = 1

lvis_path = '../data/lvis_v1'
lvis_anno = '../data/lvis_clear_75_60.json'
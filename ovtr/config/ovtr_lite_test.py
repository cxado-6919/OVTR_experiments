modelname = "OVTR"
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
extra_track_attn = True
use_checkpoint_track = False
use_checkpoint_track = False

computed_aux = [0, 1, 2, 3, 4, 5]

# OV-DPTD v1
use_ov_dptd = False
ov_dptd_use_historical_offsets = True
ov_dptd_fusion = "linear_sum"
ov_dptd_id_path_text = "none"
ov_dptd_fuse_cti = False
ov_dptd_store_debug = False
use_dptd_update_suppression = False
dptd_update_suppression_thresh = 0.4
dptd_update_suppression_restore_fields = [
    "query_tgt",
    "query_pos",
    "ref_pts",
    "dptd_sampling_offsets",
    "output_embedding_img",
    "output_embedding_txt",
]
dptd_update_suppression_track_id_based = True
use_dptd_semantic_memory = False
dptd_memory_ema = 0.8
dptd_memory_min_score = 0.4
dptd_memory_max_entropy = 0.75
dptd_memory_use_alignment_feature = True
dptd_memory_allow_untrained_visual_projection = False
dptd_memory_store_topk = 5
dptd_memory_debug = False
dptd_id_text_topk = 5
dptd_id_text_num_heads = 4
dptd_id_text_dropout = 0.0
dptd_id_text_out_zero_init = True
dptd_id_text_score_eps = 1e-8
dptd_id_text_debug = False
dptd_store_topk_text_embeddings = True
use_dptd_semantic_gate = False
dptd_gate_mode = "heuristic"
dptd_gate_min_score = 0.3
dptd_gate_max_entropy = 0.8
dptd_gate_semantic_cos_tau = 0.25
dptd_gate_visual_cos_tau = 0.25
dptd_gate_offset_tau = 0.2
dptd_gate_box_iou_tau = 0.3
dptd_gate_temperature = 10.0
dptd_gate_min_appearance = 0.1
dptd_gate_debug = False
use_dptd_semantic_update_suppression = False
dptd_semantic_update_suppression_thresh = 0.3
ov_dptd_semantic_gate_id_proj_init = "small_random"
ov_dptd_semantic_gate_id_proj_init_std = 1e-3
ov_dptd_reinit_dead_semantic_gate_id_proj = False
use_dptd_losses = False
dptd_loss_ofa_consistency_weight = 0.0
dptd_loss_semantic_memory_weight = 0.0
dptd_loss_visual_memory_weight = 0.0
dptd_loss_offset_consistency_weight = 0.0
dptd_loss_same_category_contrast_weight = 0.0
dptd_loss_min_reliability = 0.5
dptd_loss_max_entropy = 0.75
dptd_loss_min_score = 0.3
dptd_loss_query_scope = "track_only"
dptd_loss_apply_aux = False
dptd_loss_ofa_target = "ada_stopgrad"
dptd_loss_memory_target = "previous_stopgrad"
dptd_loss_offset_target = "historical_stopgrad"
dptd_contrast_temperature = 0.07
dptd_contrast_min_negatives = 1
dptd_loss_store_debug = False
use_dptd_semantic_offset_residual = False
dptd_offset_residual_scale = 0.05
dptd_offset_residual_clamp = 0.1
dptd_offset_residual_hidden_dim = 256
dptd_offset_residual_memory_dim = 512
dptd_offset_residual_use_semantic_proto = True
dptd_offset_residual_use_visual_memory = True
dptd_offset_residual_use_box_delta = True
dptd_offset_residual_use_memory_age = True
dptd_offset_residual_detach_memory = True
dptd_offset_residual_zero_init = True
dptd_offset_residual_debug = False

# scaling
prior_prob = 0.005
log_scale = 0.0
text_dim = 256

#prompt
Clip_text_embeddings = '../model_zoo/iou_neg5_ens.pth' # from DetPro
Clip_image_embeddings = '../model_zoo/clip_image_embedding_all.pt'

distribution_based_sampling = True
isolation_mask = True
run_inference = False

# freezing
initial_grad = True
initial_grad_allowed = ['patch2query',
                        'feature_align',
                        'encoder_align',
                        'input_align',
                        'fusion_layers',
                        'log_scale',
                        'bias_lang',
                        'bias0',
                        'ca_text',
                        'catext_norm',
                        'norm5',
                        'linear3',
                        'linear4',
                        'decoder_norm_inter',
                        'ov_dptd_',
                        'dptd_visual_memory_proj',
                        ]
global_grad_allowed_epoch = 1
global_grad_allowed_epoch_track = 1 
train_tracking_only = ['track_embed',
                       'update_attn',
                       'norm4',
                       'decoder.layers.0',
                       'ov_dptd_',
                       'dptd_visual_memory_proj']
train_tracking_keep = ['backbone',
                       'encoder',
                       'tgt_embed',
                       'enc_',
                       'patch2query',
                       'input_align',
                       'input_proj',
                       'level_embed']
 
train_with_artificial_img_seqs = True 

# ovtr datasets pipeline settings
img_scale = (800, 1333)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'), 
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]

# ovtr datasets settings
dataset_type = 'TaoDataset'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=3,
    persistent_workers=True,
    train=dict(
                type=dataset_type,
                classes='../data/lvis_classes_v1.txt',
                load_as_video=True,
                ann_file='../data/lvis_clear_75_60.json',
                key_img_sampler=dict(interval=1),
                ref_img_sampler=dict(num_ref_imgs=1, scope=1, method='uniform', pesudo=True),
                pipeline=[
                    dict(
                        type='LoadMultiImagesFromFile',
                        file_client_args=dict(
                            img_db_path='../data/lvis_filtered_train_images.h5',
                            backend='hdf5',
                            type='lvis'),
                        ),
                    dict(type='SeqLoadAnnotations', with_bbox=True, with_ins_id=True),],
                    
                train_pipeline = [
                    dict(type='DynamicMosaic', 
                         img_scale = (400, 666), # Increasing Mosaic img_scale will increase CPU overhead
                         pad_val=114.0, 
                         shuffle_ratio=0.1,
                         dislocation_ratio=0.25,
                         single_ratio_range=(0.7, 1.2)),
                    dict(type='RandomOcclusion',
                         n_holes=20,
                         cutout_ratio=(0.6, 0.2),
                         fill_in=[(90, 100, 110), (110, 90, 75) ,(100, 85, 110),
                                (90, 120, 100), (90, 75, 110) ,(100, 120, 95)],
                         holes_area=25000,
                         occlusion_ratio=(0.1, 0.13),
                         ),
                    dict(type='SeqRandomFlip',
                         share_params=False, 
                         flip_ratio=0.5,
                         deny_ratio=0.4), # Increasing can reduce the flipping probability.
                    dict(
                        type='SeqRandomAffine',
                        scaling_ratio_range=(0.5, 1.4),
                        scaling_ratio_range_Mosaic=(0.6, 1.6),
                        max_translate=(0.1, 0.25, 0.4, 0.55),
                        max_rotate_degree=12.0,
                        max_shear_degree=5.0
                        ),
                    dict(type='SeqYOLOXHSVRandomAug'),
                    dict(
                        type='SeqResize',
                        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736), (1333, 768), (1333, 800)],
                        share_params=True,
                        multiscale_mode='value',
                        keep_ratio=True),
                    dict(type='SeqNormalize', **img_norm_cfg),
                    dict(
                        type='SeqPad',
                        size_divisor=32,
                        # If the image is three-channel, the pad value needs
                        # to be set separately for each channel
                        ),
                    dict(type='SeqFilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
                    dict(type='SeqDefaultFormatBundle'),
                    dict(
                        type='SeqCollect',
                        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_match_indices'],
                        ref_prefix='ref'),
                    ],
                ),
    calib=dict(
        type=dataset_type,
        classes='../data/lvis_classes_v1.txt',
        ann_file='../data/lvis_clear_75_60_calib_512.json',
        img_prefix='../data/lvis_v1/',
        ref_img_sampler=None,
        pipeline=test_pipeline),
    val=dict(
        type=dataset_type,
        classes='../data/lvis_classes_v1.txt',
        ann_file='../data/tao_test_burst_v1.json',
        img_prefix='../data/TAO/',
        ref_img_sampler=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes='../data/lvis_classes_v1.txt',
        ann_file='../data/tao_test_burst_v1.json',
        img_prefix='../data/TAO/',
        ref_img_sampler=None,
        pipeline=test_pipeline)
)

evaluation = dict(metric=['track'], resfile_path='/scratch/tmp/')


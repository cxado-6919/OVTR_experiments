CUDA_DEVICES="0, 1, 2, 3"
MASTER_PORT=9987
NPROC_GPU=4

PRETRAIN_MODEL="../model_zoo/dino_ep33_4scale_double_feedforward.pth"
PRETRAIN_OUTPUT="./det_pretrain_weights"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python -m torch.distributed.launch --master_port=${MASTER_PORT} --nproc_per_node=${NPROC_GPU} \
    --use_env \
    ./main.py \
    --config_file ./config/ovtr_det_bs2_pretrain.py \
    --dataset_file lvis \
    --epochs 1 \
    --with_box_refine \
    --two_stage \
    --lr 4e-4 \
    --lr_drop 20 \
    --pretrained ${PRETRAIN_MODEL} \
    --num_workers 32 \
    --batch_size 2 \
    --calculate_negative_samples \
    --max_len 13 \
    --output_dir ${PRETRAIN_OUTPUT} \


CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python -m torch.distributed.launch --master_port=${MASTER_PORT} --nproc_per_node=${NPROC_GPU} \
    --use_env \
    ./main.py \
    --config_file ./config/ovtr_det_bs2_pretrain.py \
    --dataset_file lvis \
    --epochs 33 \
    --with_box_refine \
    --two_stage \
    --lr 4e-5 \
    --lr_drop 20 \
    --resume ${PRETRAIN_OUTPUT}/checkpoint0000.pth \
    --num_workers 32 \
    --batch_size 2 \
    --calculate_negative_samples \
    --max_len 13 \
    --output_dir ${PRETRAIN_OUTPUT} \

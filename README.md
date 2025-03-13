# OVTR: End-to-End Open-Vocabulary Multiple Object Tracking with Transformer


<p align="center"><img src="assets/Method.png" width="600"/></p>

> **[ICLR 2025] OVTR: End-to-End Open-Vocabulary Multiple Object Tracking with Transformer**
> 
> Jinyang Li, En Yu, Sijia Chen, Wenbing Tao
> 
> *[openreview](https://openreview.net/forum?id=GDS5eN65QY)*

- We propose the first end-to-end open-vocabulary multi-object tracking algorithm, introducing a novel perspective to the OVMOT field, achieving faster inference speeds, and possessing strong scalability with potential for further improvement.
- We propose the category information propagation (CIP) strategy to enhance the stability of tracking and classification, along with the attention isolation strategies that ensure open-vocabulary perception and tracking operate in harmony.
- We propose a dual-branch decoder guided by an alignment mechanism, empowering the model with strong open-vocabulary perception and multimodal interaction capabilities while eliminating the need for time-consuming preprocessing.

<p align="center"><img src="assets/Overview_ovtr.png" width="700"/></p>

## 💡 News
* We release the code, scripts and checkpoints on TAO
* Our paper is accepted by ICLR 2025!

## 🔧 Installation

```shell
# create a virtual env
conda create -n OVTR python=3.9
# activate the env
conda activate OVTR

# install OVTR
git clone https://github.com/jinyanglii/OVTR.git
cd OVTR
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 pytorch-cuda=11.1 -c pytorch -c nvidia
pip install -r requirements.txt
# Different installation versions may lead to changes in evaluation scores.
```

compile the Deformable Attention CUDA ops:

```shell
# from https://github.com/fundamentalvision/Deformable-DETR
cd ovtr/models/ops/
sh make.sh
# You can test this ops if you need:
python test.py

# for the detection pretraining
cd ../../..
cd ovtr_det_bs2_pretrain/models/ops/
sh make.sh
python test.py
```


## 💽 Data

You should put the unzipped TAO and Lvis datasets into the `data/`. And then generate the ground truth files by running the corresponding script: [./process/lvis_filter.ipynb](./process/lvis_filter.ipynb). 

Finally, you should get the following structure of Dataset and Annotations:
```
data/
  ├── Lvis_v1/
  │ ├── train2017/
  │ ├── (val2017/)
  │ └──  annotations/
  │   └── lvis_v1_train.json
  ├── lvis_filtered_train_images.h5 # Filter out images that only contain rare category targets
  ├── TAO/
  │ ├── val/
  │ └──  test/
  ├── lvis_image_v1.json
  ├── lvis_clear_75_60.json
  ├── lvis_classes_v1.txt
  ├── validation_ours_v1.json # From OVTrack
  ├── tao_test_burst_v1.json

 ```

## Training
 - **Train a complete OVTR**
```shell
cd ovtr/
sh tools/ovtr_multi_frame_train.sh
```
 - **Train Lite version (recommended)**
```shell
cd ovtr/
sh tools/ovtr_multi_frame_lite_train.sh
```
### Detection Pre-training
```shell
cd ovtr_det_bs2_pretrain/
sh tools/ovtr_detection_pretrain.sh
```

## Evaluation
 - **Evaluate OVTR on the TAO validation set using the OVMOT metric (TETA).**
```shell
cd ovtr/
sh tools/ovtr_ovmot_eval_e15_val.sh
```
 - **Evaluate OVTR on the TAO test set using the OVMOT metric (TETA).**
```shell
cd ovtr/
sh tools/ovtr_ovmot_eval_e15_test.sh
```
 - **Evaluate OVTR-Lite on the TAO validation set using the OVMOT metric (TETA).**
```shell
cd ovtr/
sh tools/ovtr_ovmot_eval_lite_val.sh
```
 - **Evaluate OVTR-Lite on the TAO test set using the OVMOT metric (TETA).**
```shell
cd ovtr/
sh tools/ovtr_ovmot_eval_lite_test.sh
```

## 🎬 Demo
<img src="ovtr/results/track_demo.gif" width="800"/>


 - **Run a demo of OVTR.**
```shell
cd ovtr/
sh tools/ovtr_demo.sh
```

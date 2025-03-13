# OVTR: End-to-End Open-Vocabulary Multiple Object Tracking with Transformer


<p align="center"><img src="assets/sota.png" width="500"/></p>

> **[ICLR 2025] OVTR: End-to-End Open-Vocabulary Multiple Object Tracking with Transformer**
> 
> Jinyang Li, En Yu, Sijia Chen, Wenbing Tao
> 
> *[openreview](https://openreview.net/forum?id=GDS5eN65QY)*

- We propose the first end-to-end open-vocabulary multi-object tracking algorithm, introducing a novel perspective to the OVMOT field, achieving faster inference speeds, and possessing strong scalability with potential for further improvement.
- We propose the category information propagation (CIP) strategy to enhance the stability of tracking and classification, along with the attention isolation strategies that ensure open-vocabulary perception and tracking operate in harmony.
- We propose a dual-branch decoder guided by an alignment mechanism, empowering the model with strong open-vocabulary perception and multimodal interaction capabilities while eliminating the need for time-consuming preprocessing.

## 💡 News
* We release the code, scripts and checkpoints on TAO
* Our paper is accepted by ICLR 2025!

## Installation

```shell
# create a virtual env
conda create -n OVTR python=3.9
# activate the env
conda activate OVTR

conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 pytorch-cuda=11.1 -c pytorch -c nvidia

pip install -r requirements.txt
```

You also need to compile the Deformable Attention CUDA ops:

```shell
# From https://github.com/fundamentalvision/Deformable-DETR
cd ./models/ops/
sh make.sh
# You can test this ops if you need:
python test.py
```

## 🎬 Demo
<img src="ovtr/results/track_demo.gif" width="800"/>

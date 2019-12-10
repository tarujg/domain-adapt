
Data Augmentation using Domain Adaptation of Synthetic Data for Semantic Segmentation
===================================

_Autonomous driving is an open ongoing research topic, which has seen a tremendous rise over the last few years. We explore semantic segmentation of the cityscape dataset with additional data augmented using domain adaptation with the help of CycleGANs._


_**Contributors:** Atman Patel, Mudit Jain, Taruj Goyal, and Harshita Krishna (Team: sCVngers)_

Project Organization
------------

    ├── README.md   		<- The top-level README for developers using this project.
    ├── cycleGAN    		<- cycleGANs for image-to-image translation
    ├── OCNet       		<- Semantic segmentation module
	    ├── run_asp_oc.sh	<- Training, validating, testing script
    ├── run.ipynb   		<- Jupyter Notebook for quick demo

Requirements
-----------
 - Python 3.6+
 - Git
 - PyTorch 0.4.1
- Linux (tested on Ubuntu 18.04)
- NVIDIA GPU is strongly recommended
- [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn)
 - Conda
 - Docker 19.03

To clone the repository
``` bash
$ git clone https://github.com/tarujg/domain-adapt.git
```
 ``` bash
$ pip install -r requirements.txt
```

 - NVIDIA Docker Setup (Ubuntu 16.04/18.04, Debian Jessie/Stretch/Buster)(https://hub.docker.com/r/rainbowsecret/pytorch04/tags/)
#### Add the package repositories
``` bash
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
$ sudo systemctl restart docker
```

---
## Datasets
We are currently using [Cityscapes](https://www.cityscapes-dataset.com/), GTA2Cityscapes and [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) dataset for our project.

| Cityscapes | GTA5 | GTA2Cityscapes
|:------:|:------:|:------:|
|[link]([https://www.cityscapes-dataset.com/](https://www.cityscapes-dataset.com/))|[link]([https://download.visinf.tu-darmstadt.de/data/from_games/](https://download.visinf.tu-darmstadt.de/data/from_games/))|[link]([http://efrosgans.eecs.berkeley.edu/cyclegta/cityscapes2gta.zip](http://efrosgans.eecs.berkeley.edu/cyclegta/cityscapes2gta.zip))


#### References and used code sources
- [InplaceABN](https://github.com/mapillary/inplace_abn)
- [Non-local_pytorch](https://github.com/AlexHex7/Non-local_pytorch).
- [Pytorch-Deeplab](https://github.com/speedinghzl/Pytorch-Deeplab)
- [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
- [semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch)
```
TODO: Add citations here
1. OCNet
2. CycleGAN
```

#### Please contact us if you have any questions.
Atman Patel at <a2patel@ucsd.edu>
Taruj Goyal at <tgoyal@ucsd.edu>
Mudit Jain at <mujain@ucsd.edu>
Harshita Krishna at <h1krishn@ucsd.edu>

# TTGA-U-Net
TTGA U-Net: Two-stage Two-stream Graph Attention U-Net for Hepatic Vessel Connectivity Enhancement
## 1. Environment
Please prepare an environment with python=3.8, and then use the command "pip install -r requirments.txt" for the dependencies.
## 2. First-stage Network
* We use nine baseline medical image segmentation methods for first stage segmentation, and UNETR++ achieved the best performance. The implementation of UNETR++ details is [here](https://github.com/Amshaker/unetr_plus_plus).
## 3. Second-stage Network
* Graph construction for generating graph for the second-stage network.
* ["train.py"](https://github.com/muxin0527/TTGA-U-Net/blob/main/second-stage/TGA%20U-Net/train.py) and ["test.py"](https://github.com/muxin0527/TTGA-U-Net/blob/main/second-stage/TGA%20U-Net/test.py) for training and testing, [2d_visual.py](https://github.com/muxin0527/TTGA-U-Net/blob/main/second-stage/TGA%20U-Net/2d_visual.py) for visualization.
## 4. Evaluation
* ["/utils/metrics.py"](https://github.com/muxin0527/TTGA-U-Net/blob/main/utils/metric.py) for quantitative evaluation.
## Acknowledgement
This repository is built based on [Av-casNet](https://github.com/xjtu-mia/octa?tab=readme-ov-file#citation) repository.

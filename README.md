# Pytorch implementation of DYNAS
This is the implementation of the paper "Scheduling Weight Transitions for Quantization-Aware Training".

For detailed information, please checkout the project site [[website](https://cvlab.yonsei.ac.kr/projects/TRS/)] or the paper [[arXiv](https://arxiv.org/abs/2404.19248)].






## Requirements
This repository has been tested with the following libraries:
* Python 3.8.8
* Numpy 1.19.2
* PyTorch 1.8.1
* Torchvision 0.9.1
* TensorBoard 2.5.0

## How to run
We provide examples of commands in the ``run.sh`` file. \
Training data (i.e., CIFAR-10/100) will be automatically downloaded. \
You can modify the scripts for different settings (e.g., bit-widths).

## Citation

```
@inproceedings{lee2024scheduling,
  title={Scheduling Weight Transitions for Quantization-Aware Training},
  author={Lee, Junghyup and Jeon, Jeimin and Kim, Dohyung and Ham, Bumsub},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```




# ICCV-2023-FedETF
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


>**This is the official implementation of ICCV 2023 paper "[No Fear of Classifier Biases: Neural Collapse Inspired Federated Learning with Synthetic and Fixed Classifier](https://openaccess.thecvf.com/content/ICCV2023/html/Li_No_Fear_of_Classifier_Biases_Neural_Collapse_Inspired_Federated_Learning_ICCV_2023_paper.html) ([arixv](https://arxiv.org/abs/2303.10058))".**

## Paper Overview
**TLDR**: We devise FedETF which is inspired by the neural collapse phenomenon, showing both strong generalization and personalization performances.

**Abstract**: Data heterogeneity is an inherent challenge that hinders the performance of federated learning (FL). Recent studies have identified _the biased classifiers_ of local models as the key bottleneck. Previous attempts have used classifier calibration after FL training, but this approach falls short in improving the poor feature representations caused by training-time classifier biases. Resolving the classifier bias dilemma in FL requires a full understanding of the mechanisms behind the classifier. Recent advances in neural collapse have shown that the classifiers and feature prototypes under perfect training scenarios collapse into an optimal structure called simplex equiangular tight frame (ETF). Building on this neural collapse insight, we propose a solution to the FL's classifier bias problem by _utilizing a synthetic and fixed ETF classifier during training_. The optimal classifier structure enables all clients to learn unified and optimal feature representations even under extremely heterogeneous data. We devise several effective modules to better adapt the ETF structure in FL, achieving both high generalization and personalization. Extensive experiments demonstrate that our method achieves state-of-the-art performances on CIFAR-10, CIFAR-100, and Tiny-ImageNet.

## Citing This Repository

Please cite our paper if you find this repo useful in your work:

```
@InProceedings{Li_2023_ICCV,
    author    = {Li, Zexi and Shang, Xinyi and He, Rui and Lin, Tao and Wu, Chao},
    title     = {No Fear of Classifier Biases: Neural Collapse Inspired Federated Learning with Synthetic and Fixed Classifier},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {5319-5329}
}
```

# SwapGT
This is the code for our NeurIPS 2025 paper 
[**Rethinking Tokenized Graph Transformers for Node Classification**](https://arxiv.org/abs/2502.08101).

![SwapGT](./SwapGT.png)

![Token swapping](./Token_swapping.png)

## Requirements

Python == 3.8

Pytorch == 1.11

torch-geometric == 2.3

torch-scatter == 2.0

torch-sparse == 0.6

dgl == 0.9

CUDA == 10.2

## Split

We provide two strategies to split datasets, called **dense splitting** and **sparse splitting**.

For dense splitting, we adopt 50%/25%/25% splitting for training set, validation set and test set.

For sparse splitting, we adopt 2.5%/2.5%/95% splitting for training set, validation set and test set.

## Usage

You can run each command in "cmd.sh".

You could change the hyper-parameters of SwapGT if necessary.

Due to the space limitation, please refer to this [link](https://drive.google.com/file/d/1Selz5yL1_-dedgfWluWAJtWJcpp7YBwT/view?usp=sharing) to download the datasets as well as pre-computing data. 
Once you have done this, please put them into the corresponding folders (./dataset/de/, ./dataset/sp/, ./pre_cal/de/ and ./pre_cal/sp/)



## Cite
If you find this code useful, please consider citing the original work by authors:
```
@inproceedings{SwapGT,
author={Jinsong Chen and Chenyang Li and Gaichao Li and John E. Hopcroft and Kun He},
 title = {Rethinking Tokenized Graph Transformers for Node Classification}, 
 booktitle = {Proceedings of the 39th Annual Conference on Neural Information Processing Systems},
 volume = {38},
 pages = {134725--134750},
 year = {2025}
}
```
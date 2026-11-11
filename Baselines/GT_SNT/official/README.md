# GT-SNT

## Environments
+ torch == 2.2.1+cu121
+ torch-cluster == 1.6.3
+ spikingjelly == 0.0.0.0.14
+ torch_geometric == 2.5.1
+ torch-scatter == 2.1.2
+ torch-sparse == 0.6.18
+ CUDA 12.1
+ cuDNN 8.9.0

## Reproduction
Run experiments using [`node_cls.py`](./node_cls.py)/[`node_cls_subgraph.py`](./node_cls_subgraph.py). Configurations are specified in [`config`](./config).
+ Cora
```
python node_cls.py --config_dir "./config/cora.yaml"
```
+ Citeseer
```
python node_cls.py --config_dir "./config/citeseer.yaml"
```
+ Pubmed
```
python node_cls.py --config_dir "./config/pubmed.yaml"
```
+ CS
```
python node_cls.py --config_dir "./config/cs.yaml"
```
+ Physics 
```
python node_cls.py --config_dir "./config/physics.yaml"
```
+ ogbn-arxiv
```
python node_cls.py --config_dir "./config/ogbn-arxiv.yaml"
```
+ ogbn-products
```
python node_cls_subgraph.py --config_dir "./config/ogbn-products.yaml"
```

## Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/Zhhuizhe/GT-SNT/issues).

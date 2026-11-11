# sparse

python train_test_sh.py --dataset uai  --dropout 0.1 --attention_dropout 0.1 --hidden 512 --depth 2 --n_heads 8 --n_layers 1 --peak_lr 0.001  --pp_k 2 --sample_num 6 --num_aug 4 --lambda1 1 --lambda2 0.5 --dstyle sp

python train_test_sh.py --dataset BlogCatalog  --dropout 0.5 --attention_dropout 0.5 --hidden 512 --depth 2 --n_heads 8 --n_layers 1 --peak_lr 0.001  --pp_k 2 --sample_num 6 --num_aug 3 --lambda1 10 --lambda2 0.1 --dstyle sp

python train_test_sh.py --dataset flickr  --dropout 0.1 --attention_dropout 0.1 --hidden 512 --depth 2 --n_heads 8 --n_layers 1 --peak_lr 0.01  --pp_k 2 --sample_num 8 --num_aug 6 --lambda1 1 --lambda2 0.6 --dstyle sp

python train_test_sh.py --dataset wikics  --dropout 0.1 --attention_dropout 0.5 --hidden 256 --depth 2 --n_heads 8 --n_layers 1 --peak_lr 0.0005  --pp_k 3 --sample_num 10 --num_aug 3 --lambda1 10 --lambda2 0.2 --dstyle sp

python train_test_sh.py --dataset computer  --dropout 0.1 --attention_dropout 0.5 --hidden 512 --depth 2 --n_heads 8 --n_layers 1 --peak_lr 0.001  --pp_k 4 --sample_num 8 --num_aug 3 --lambda1 15 --lambda2 0.1 --dstyle sp

python train_test_sh.py --dataset citeseer  --dropout 0.1 --attention_dropout 0.1 --hidden 512 --depth 2 --n_heads 8 --n_layers 1 --peak_lr 0.001  --pp_k 3 --sample_num 10 --num_aug 3 --lambda1 1 --lambda2 0.9 --dstyle sp

python train_test_sh.py --dataset acm  --dropout 0.1 --attention_dropout 0.1 --hidden 512 --depth 2 --n_heads 8 --n_layers 1 --peak_lr 0.001  --pp_k 2 --sample_num 4 --num_aug 4 --lambda1 10 --lambda2 0.1 --dstyle sp

python train_test_sh.py --dataset photo  --dropout 0.1 --attention_dropout 0.1 --hidden 512 --depth 2 --n_heads 8 --n_layers 1 --peak_lr 0.001  --pp_k 3 --sample_num 6 --num_aug 4 --lambda1 15 --lambda2 0.1 --dstyle sp


# dense

python train_test_sh.py --dataset citeseer  --dropout 0.1 --attention_dropout 0.1 --hidden 512 --depth 2 --n_heads 8 --n_layers 1 --peak_lr 0.001  --pp_k 3 --sample_num 10 --num_aug 2 --lambda1 1 --lambda2 0.6 --dstyle de

python train_test_sh.py --dataset flickr  --dropout 0.5 --attention_dropout 0.5 --hidden 512 --depth 2 --n_heads 8 --n_layers 1 --peak_lr 0.01  --pp_k 2 --sample_num 6 --num_aug 2 --lambda1 0.5 --lambda2 0.5 --dstyle de

python train_test_sh.py --dataset wikics  --dropout 0.1 --attention_dropout 0.1 --hidden 512 --depth 2 --n_heads 8 --n_layers 1 --peak_lr 0.001  --pp_k 2 --sample_num 10 --num_aug 2 --lambda1 0.3 --lambda2 0.3 --dstyle de

python train_test_sh.py --dataset uai  --dropout 0.1 --attention_dropout 0.5 --hidden 512 --depth 2 --n_heads 8 --n_layers 1 --peak_lr 0.001  --pp_k 2 --sample_num 8 --num_aug 2 --lambda1 0.6 --lambda2 0.2 --dstyle de

python train_test_sh.py --dataset photo  --dropout 0.1 --attention_dropout 0.1 --hidden 512 --depth 2 --n_heads 8 --n_layers 1 --peak_lr 0.001  --pp_k 3 --sample_num 4 --num_aug 2 --lambda1 10 --lambda2 0.1 --dstyle de

python train_test_sh.py --dataset acm  --dropout 0.1 --attention_dropout 0.1 --hidden 512 --depth 2 --n_heads 8 --n_layers 1 --peak_lr 0.001  --pp_k 3 --sample_num 6 --num_aug 4 --lambda1 15 --lambda2 0.5 --dstyle de

python train_test_sh.py  --dataset computer  --dropout 0.1 --attention_dropout 0.5 --hidden 512 --depth 2 --n_heads 8 --n_layers 1 --peak_lr 0.0005  --pp_k 4 --sample_num 6 --num_aug 3 --lambda1 20 --lambda2 0.1 --dstyle de

python train_test_sh.py  --dataset BlogCatalog  --dropout 0.1 --attention_dropout 0.1 --hidden 512 --depth 2 --n_heads 8 --n_layers 1 --peak_lr 0.0005  --pp_k 2 --sample_num 6 --num_aug 3 --lambda1 15 --lambda2 0.4 --dstyle de 


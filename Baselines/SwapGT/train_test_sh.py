from data import get_dataset
import time
import utils
import random
import numpy as np
import torch
import torch.nn.functional as F
from early_stop import EarlyStopping, Stop_args
from model import SwapGT
from lr import PolynomialDecayLR
import os.path
import torch.utils.data as Data
import argparse
#import wandb

# Training settings
def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()

    # main parameters
    parser.add_argument('--name', type=str, default="basic")
    parser.add_argument('--dataset', type=str, default='cora',
                        help='Choose from {pubmed}')
    parser.add_argument('--device', type=int, default=0,
                        help='Device cuda id')
    parser.add_argument('--seed', type=int, default=3407,
                        help='Random seed.')
    parser.add_argument('--dstyle', type=str, default="sp",
                        help='de, sp')

    parser.add_argument('--split_seed', type=int, default=66,
                        help='the split seed')



    # model parameters

    parser.add_argument('--hidden', type=int, default=512,
                        help='Hidden layer size')
    parser.add_argument('--ffn_dim', type=int, default=64,
                        help='FFN layer size')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of Transformer heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout')
    parser.add_argument('--attention_dropout', type=float, default=0.1,
                        help='Dropout in the attention layer')

    parser.add_argument('--alpha', type=float, default=0.1,
                        help='memory bank update')

    parser.add_argument('--beta', type=float, default=0.8,
                        help='memory bank update')

    parser.add_argument('--pp_k', type=int, default=3,
                        help='propagation steps')

    parser.add_argument('--ratio', type=float, default=0.6,
                        help='the ratio of training set')

    parser.add_argument('--num_aug', type=int, default=2,
                        help='the number of augmentation token sequences')

    parser.add_argument('--depth', type=int, default=1,
                        help='the number of augmentation depth')


    parser.add_argument('--sample_num', type=int, default=4,
                        help='Number of nodes to be sampled')

    parser.add_argument('--lambda1', type=float, default=0.5,
                        help='the weight for contrastive loss')
    parser.add_argument('--lambda2', type=float, default=0.5,
                        help='the weight for attribute and topology fusion')


    # training parameters
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size')
    parser.add_argument('--sample_size', type=int, default=50000,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to train.')
    parser.add_argument('--tot_updates', type=int, default=1000,
                        help='used for optimizer learning rate scheduling')
    parser.add_argument('--warmup_updates', type=int, default=400,
                        help='warmup steps')
    parser.add_argument('--peak_lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--end_lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience for early stopping')

    parser.add_argument('--runs', type=int, default=5,
                        help='Number of independent training runs')

    return parser.parse_args()

args = parse_args()

print(args)

acc_list = []

device = args.device

for run in range(args.runs):
    random.seed(args.seed + run)
    np.random.seed(args.seed+ run)
    torch.manual_seed(args.seed+ run)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed+ run)

    # Load and pre-process data
    adj, features, labels, idx_train, idx_val, idx_test = get_dataset(args.dataset, args.dstyle)

    data_file = "./pre_cal/"+args.dstyle+'/'+args.dataset+"_"+str(args.sample_num)+"_"+str(args.pp_k)+"_"+str(args.num_aug)+"_"+str(args.depth)+".pt"
    
    if os.path.isfile(data_file):
        final_pre_features = torch.load(data_file)
    else:
        #top_k sampling
        attr_processed_features, attr_tokens, label_tokens = utils.node_seq_feature(features, args.sample_num, args.sample_size, labels)  # return (N, sample_num+1, d)

        ppr_features = utils.node_neighborhood_feature(adj, features, args.pp_k, args.alpha)  # return (N, d)
        
        ppr_processed_features, topo_tokens, label_tokens = utils.node_seq_feature(ppr_features, args.sample_num, args.sample_size, labels) # return (N, sample_num+1, d)

        attr_aug_features = utils.mix_token(attr_processed_features, attr_tokens, args.num_aug, args.depth, features)
        
        ppr_aug_features = utils.mix_token(ppr_processed_features, topo_tokens, args.num_aug, args.depth, features)


        final_pre_features = torch.cat((attr_aug_features, ppr_aug_features), dim=1) 

        # store the data 
        # torch.save(final_pre_features, data_file)


    labels = labels.to(device)

    batch_data_train = Data.TensorDataset(final_pre_features[idx_train], labels[idx_train])
    batch_data_val = Data.TensorDataset(final_pre_features[idx_val], labels[idx_val])
    batch_data_test = Data.TensorDataset(final_pre_features[idx_test], labels[idx_test])

    train_data_loader = Data.DataLoader(batch_data_train, batch_size=args.batch_size, shuffle=True)
    val_data_loader = Data.DataLoader(batch_data_val, batch_size=args.batch_size, shuffle=True)
    test_data_loader = Data.DataLoader(batch_data_test, batch_size=args.batch_size, shuffle=True)


    n_class=labels.max().item() + 1


    # model configuration
    model = SwapGT(n_layers=args.n_layers,
                input_dim=final_pre_features.shape[-1],
                hidden_dim=args.hidden,
                n_class=n_class,
                num_heads=args.n_heads,
                ffn_dim=args.ffn_dim,
                dropout_rate=args.dropout,
                attention_dropout_rate=args.attention_dropout,
                args=args).to(device)

    class_bank = torch.eye(n_class, requires_grad=False).to(device)
    print(model)
    print('total params:', sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup_updates=args.warmup_updates,
        tot_updates=args.tot_updates,
        lr=args.peak_lr,
        end_lr=args.end_lr,
        power=1.0,
    )


    def update_bank(output, labels, beta):
        for i in range(n_class):
            find_label = (labels==i).nonzero()
            out_label = output[find_label]
            out_label = out_label.mean(dim=0)
            class_bank[i] = beta * class_bank[i] + (1 - beta) * out_label
        return 0


    def train_valid_epoch(epoch):
        model.train()
        loss_train_b = 0
        acc_train_b = 0
        for _, item in enumerate(train_data_loader):
            nodes_features = item[0].to(device)
            labels = item[1].to(device)

            optimizer.zero_grad()

            output, loss, ce_loss, loss_sim = model(nodes_features, labels, True, class_bank)

            sing = update_bank(output, labels, args.beta)
            

            loss_train = loss

            loss_train.backward()
            optimizer.step()
            lr_scheduler.step()

            loss_train_b += loss_train.item()

            acc_train_b += utils.accuracy_batch(output, labels).item()


        model.eval()
        loss_val = 0
        acc_val = 0
        for _, item in enumerate(val_data_loader):
            with torch.no_grad():
                nodes_features = item[0].to(device)
                labels = item[1].to(device)

                output, loss, ce_loss, loss_sim = model(nodes_features, labels, False, class_bank)
                
                loss_val += ce_loss #+ args.alpha * l2_loss

                # loss_val += F.nll_loss(output, labels).item()

                acc_val += utils.accuracy_batch(output, labels).item()


        print('Epoch: {:04d}'.format(epoch + 1),
            'loss_train: {:.4f}'.format(loss_train_b),
            'acc_train: {:.4f}'.format(acc_train_b / len(idx_train)),
            'loss_val: {:.4f}'.format(loss_val),
            'acc_val: {:.4f}'.format(acc_val / len(idx_val)))
        

        return loss_val.item(), acc_val


    def test():
        loss_test = 0
        acc_test = 0
        
        for _, item in enumerate(test_data_loader):
            with torch.no_grad():
                nodes_features = item[0].to(device)
                labels = item[1].to(device)

                model.eval()

                output, loss, ce_loss, loss_sim = model(nodes_features, labels, False, class_bank)
                
                loss_test += ce_loss #+ args.alpha * l2_loss
                # loss_test += F.nll_loss(output, labels).item()
                acc_test += utils.accuracy_batch(output, labels).item()
                torch.cuda.empty_cache()
                
        print("Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test / len(idx_test)))

        
        return acc_test / len(idx_test)


    t_total = time.time()
    stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
    early_stopping = EarlyStopping(model, **stopping_args)
    for epoch in range(args.epochs):
        loss_val, acc_val = train_valid_epoch(epoch)
        if early_stopping.check([acc_val, loss_val], epoch):
            break

    print("Optimization Finished!")
    print("Train cost: {:.4f}s".format(time.time() - t_total))
    # Restore best model
    print('Loading {}th epoch'.format(early_stopping.best_epoch + 1))
    model.load_state_dict(early_stopping.best_state)

    acc = test()

    acc_list += [acc]


acc_mean = np.mean(acc_list)

acc_std = np.std(acc_list, ddof = 1)


print("Final Test results:",
        "acc_mean= {:.4f}".format(acc_mean),
        "acc_std= {:.4f}".format(acc_std))


import torch
import math
import torch.nn as nn
from torch.nn import functional as F
# from einops import rearrange

def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)

        # print(x)
        x = self.att_dropout(x)


        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        # 初始的残差x, 求和特征
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)

        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)

        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)


    def forward(self, x, attn_bias=None):


        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y
        ## 实现的是transformer 和 FFN的LayerNorm 以及相关操作
        
        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y

        return x


class SwapGT(nn.Module):
    def __init__(
        self,
        n_layers,
        input_dim,
        hidden_dim,
        n_class,
        num_heads,
        ffn_dim,
        dropout_rate,
        attention_dropout_rate,
        args):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.n_layers = n_layers
        self.n_class = n_class
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        self.node_emb = nn.Linear(self.input_dim,  self.hidden_dim)
        self.hop_emb = nn.Linear(self.input_dim,  self.hidden_dim)


        
        encoders = [EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.attention_dropout_rate, self.num_heads)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)

        self.final_ln = nn.LayerNorm(hidden_dim)

        #self.out_proj = nn.Linear(self.hidden_dim, int(self.hidden_dim / 2))

        #self.Linear1 = nn.Linear(int(self.hidden_dim / 2), self.n_class)
        
        self.out_proj = nn.Linear(2*self.hidden_dim, self.hidden_dim)

        self.Linear1 = nn.Linear(self.hidden_dim, self.n_class)


        self.args = args

        self.CEloss  = nn.CrossEntropyLoss()
        #self.loss = nn.NLLLoss()

        self.MSE = torch.nn.MSELoss(reduction='mean')
 
        self.KLloss = torch.nn.KLDivLoss(reduction='batchmean')

        self.apply(lambda module: init_params(module, n_layers=n_layers))


    def forward(self, batched_data, labels, isTrain, class_bank):
        
        # print(batched_data.shape)
        # size: (n, (k+1)(aug+1)+(k+1)(aug+1), d)

        node_embedding = self.node_emb(batched_data)

        #Transformer block
        x0 = node_embedding

        for i, l in enumerate(self.layers):

            node_embedding = self.layers[i](node_embedding)
            node_embedding = x0 + node_embedding
            

        node_embedding = self.final_ln(node_embedding)


        
        node_embedding = torch.reshape(node_embedding, (node_embedding.shape[0], -1, (self.args.sample_num+1), self.hidden_dim))

        # print(node_embedding.shape)

        # node_embedding: [n, (1+num_aug)*2, (k+1), d]
        # readout function: 1. 取每个token的第一行， 2.求均值

        # 取seq的首token
        # [ACE-baseline patch] official: node_embedding[:, :, 0, :].squeeze(); the tensor is already
        # [n, (1+num_aug)*2, d], so squeeze() only changes it when a batch holds a single node, where it
        # would drop the batch axis and break the split below.
        output = node_embedding[:, :, 0, :] # [n, (1+num_aug)*2, d]

        # 拆分属性视角和拓扑视角的token sequence.
        split_list = [self.args.num_aug+1, self.args.num_aug+1]
        split_tensor = torch.split(output, split_list, dim=1)

        attr_token_embeddings = split_tensor[0]
        topo_token_embeddings = split_tensor[1]


        attr_mean = attr_token_embeddings.mean(dim=1).unsqueeze(dim=1) # [n,1,d]
        attr_loss_sim = 1 - torch.nn.functional.cosine_similarity(attr_token_embeddings, attr_mean, dim=2)

        topo_mean = topo_token_embeddings.mean(dim=1).unsqueeze(dim=1) # [n,1,d]
        topo_loss_sim = 1 - torch.nn.functional.cosine_similarity(topo_token_embeddings, topo_mean, dim=2)

        loss_sim = attr_loss_sim.mean() + topo_loss_sim.mean()

        output1 = attr_token_embeddings[:,0,:]
        output2 = attr_token_embeddings[:,1:,:].mean(dim=1)
        out_attr = torch.cat((output1, output2), dim=1)

        output3 = topo_token_embeddings[:,0,:]
        output4 = topo_token_embeddings[:,1:,:].mean(dim=1)
        out_topo = torch.cat((output3, output4), dim=1)

        output = self.args.lambda2 * out_attr + (1-self.args.lambda2 ) * out_topo  # [n, 2d]

        output = self.Linear1(torch.relu(self.out_proj(output)))  # [n, c]

        ce_loss = self.CEloss(output, labels)

        output = torch.softmax(output, dim=1)

        loss = ce_loss + self.args.lambda1 * loss_sim

        return  output, loss, ce_loss, loss_sim





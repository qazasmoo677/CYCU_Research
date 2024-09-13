import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
from Attention import Attention


class POI_Aggregator(nn.Module):
    """
    POI Aggregator: for aggregating embeddings of POI neighbors.
    """

    def __init__(self, features, v2e, embed_dim, cuda="cpu"):
        super(POI_Aggregator, self).__init__()

        self.features = features
        self.device = cuda
        self.v2e = v2e
        self.embed_dim = embed_dim
        self.att = Attention(self.embed_dim)

    def forward(self, nodes, to_neighs):
        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        for i in range(len(nodes)):
            tmp_adj = to_neighs[i]
            num_neighs = len(tmp_adj)
            # 
            e_v = self.v2e.weight[list(tmp_adj)] # fast: user embedding 
            #slow: item-space user latent factor (item aggregation)
            #feature_neigbhors = self.features(torch.LongTensor(list(tmp_adj)).to(self.device))
            #e_v = torch.t(feature_neigbhors)

            v_rep = self.v2e.weight[nodes[i]]

            att_w = self.att(e_v, v_rep, num_neighs)
            att_history = torch.mm(e_v.t(), att_w).t()
            embed_matrix[i] = att_history
        to_feats = embed_matrix

        return to_feats

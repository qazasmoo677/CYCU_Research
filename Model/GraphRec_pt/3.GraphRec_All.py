import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pickle
import numpy as np
import time
import random
from collections import defaultdict
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Social_Encoders import Social_Encoder
from Social_Aggregators import Social_Aggregator
from POI_Encoders import POI_Encoder
from POI_Aggregators import POI_Aggregator
from SentimentAndTimeGNNCombiner import SentimentAndTimeGNNCombiner
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os

class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v, r2e):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v = enc_v
        self.embed_dim = enc_u.embed_dim

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_v):
        embeds_u = self.enc_u(nodes_u)
        embeds_v = self.enc_v(nodes_v)

        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)

def train(graphrec, sentimentAndTime_model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    graphrec.train()
    sentimentAndTime_model.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list, sentiment_vectors, timeProb = data
        optimizer.zero_grad()
        
        # GNN的預測評分
        gnn_score = graphrec(batch_nodes_u.to(device), batch_nodes_v.to(device))

        # 確保gnn_score的形狀為 [batch_size, 1]
        if gnn_score.dim() == 1:
            gnn_score = gnn_score.unsqueeze(1)
        
        # 結合GNN預測評分和情感向量的模型
        combined_score = sentimentAndTime_model(gnn_score, timeProb.to(device), sentiment_vectors.to(device))

        # 確保labels_list的形狀為 [batch_size, 1]
        labels_list = labels_list.unsqueeze(1).to(device)
        
        # 計算損失
        loss = nn.MSELoss()(combined_score, labels_list.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best mae/rmse: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_mae, best_rmse))
            running_loss = 0.0
    return 0

def test(graphrec, sentimentAndTime_model, device, test_loader):
    graphrec.eval()
    sentimentAndTime_model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target, sentiment_vectors, timeProb in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            
            # GNN的預測評分
            gnn_score = graphrec(test_u, test_v)

            # 確保gnn_score的形狀為 [batch_size, 1]
            if gnn_score.dim() == 1:
                gnn_score = gnn_score.unsqueeze(1)
            
            # 結合GNN預測評分和情感向量的模型
            combined_score = sentimentAndTime_model(gnn_score, timeProb.to(device), sentiment_vectors.to(device))
            
            tmp_pred.append(list(combined_score.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    expected_mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, expected_mae


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim
    dir_data = './data/final'

    path_data = dir_data + ".pickle"
    data_file = open(path_data, 'rb')
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, train_s, train_t, test_u, test_v, test_r, test_s, test_t, social_adj_lists, POI_adj_lists, ratings_list = pickle.load(
        data_file)

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r), torch.FloatTensor(train_s), torch.FloatTensor(train_t))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r), torch.FloatTensor(test_s), torch.FloatTensor(test_t))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)
    num_users = history_u_lists.__len__()
    num_items = history_v_lists.__len__()
    num_ratings = ratings_list.__len__()

    u2e = nn.Embedding(num_users, embed_dim).to(device)
    v2e = nn.Embedding(num_items, embed_dim).to(device)
    r2e = nn.Embedding(num_ratings, embed_dim).to(device)

    # user feature
    # features: item * rating
    agg_u_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
    enc_u_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device, uv=True)
    # neighobrs
    agg_u_social = Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), u2e, embed_dim, cuda=device)
    enc_u = Social_Encoder(lambda nodes: enc_u_history(nodes).t(), embed_dim, social_adj_lists, agg_u_social,
                           base_model=enc_u_history, cuda=device)

    # item feature
    # features: user * rating
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device, uv=False)
    # neighbors
    agg_v_neighbor = POI_Aggregator(lambda nodes: enc_v_history(nodes).t(), v2e, embed_dim, cuda=device)
    enc_v = POI_Encoder(lambda nodes: enc_v_history(nodes).t(), embed_dim, POI_adj_lists, agg_v_neighbor,
                           base_model=enc_v_history, cuda=device)

    # model
    # graphrec = GraphRec(enc_u, enc_v_history, r2e).to(device)
    graphrec = GraphRec(enc_u, enc_v, r2e).to(device)
    sentimentAndTime_model = SentimentAndTimeGNNCombiner().to(device)

    optimizer = torch.optim.RMSprop(list(graphrec.parameters()) + list(sentimentAndTime_model.parameters()), lr=args.lr, alpha=0.9)

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0

    for epoch in range(1, args.epochs + 1):

        train(graphrec, sentimentAndTime_model, device, train_loader, optimizer, epoch, best_rmse, best_mae)
        expected_rmse, expected_mae = test(graphrec, sentimentAndTime_model, device, test_loader)
    
        # please add the validation set to tune the hyper-parameters based on your datasets.

        # early stopping (no validation set in toy dataset)
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            endure_count = 0
        else:
            endure_count += 1
        if best_mae > expected_mae:
            best_mae = expected_mae
        print("mae: %.4f, rmse:%.4f " % (expected_mae, expected_rmse))

        if endure_count > 5:
            print('this embedding size as ', embed_dim)
            print('this learning rate as ', args.lr)
            break


if __name__ == "__main__":
    main()

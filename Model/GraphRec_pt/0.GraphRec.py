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
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from Evaluation import Evaluation
import datetime
import argparse
import os

class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v_history, r2e):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
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
        embeds_v = self.enc_v_history(nodes_v)

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

def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae, best_precision, best_recall, best_map, best_ndcg, ks):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch, i, running_loss / 100))
            print('The best mae/rmse: %.6f / %.6f' % (best_mae, best_rmse))
            for idx in range(len(ks)):
                print('The best Evaluation@%d: precision: %.6f, recall: %.6f, map: %.6f, ndcg: %.6f' % (ks[idx], best_precision[idx], best_recall[idx], best_map[idx], best_ndcg[idx]))
            running_loss = 0.0
    return 0


def test(model, device, test_loader, userList, ks):
    model.eval()
    tmp_pred = []
    target = []
    userScores = {}

    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))

            # 建立預測使用者評分字典
            for user, poi_id, score in zip(test_u.cpu().numpy(), test_v.cpu().numpy(), val_output.data.cpu().numpy()):
                if user not in userScores:
                    userScores[user] = []
                userScores[user].append({'POI_id': poi_id, 'score': score})
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))

    # 算MAE跟RMSE要去掉沒去過的地方
    nTmpPred = []
    nTarget = []
    for i in range(len(tmp_pred)):
        if target[i] != -1:
            nTmpPred.append(tmp_pred[i])
            nTarget.append(target[i])
    finalPred = np.array(nTmpPred)
    finalTarget = np.array(nTarget)

    expected_rmse = sqrt(mean_squared_error(finalPred, finalTarget))
    expected_mae = mean_absolute_error(finalPred, finalTarget)

    expected_precision = list()
    expected_recall = list()
    expected_map = list()
    expected_ndcg = list()
    Eva = Evaluation()
    for k in ks:
        needCalcUser = 0
        # 從userScores中取出每個user的前k個POI
        top_poi_ids = {}
        for user, scores in userScores.items():
            sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)[:k]
            top_poi_ids[user] = [poi['POI_id'] for poi in sorted_scores]

        # 有多少user需要計算
        for userIdx in range(len(userScores)):
            if Eva.checkNeedCalcUser(top_poi_ids[userIdx], userList[userIdx]):
                needCalcUser += 1

        # 算Precision, Recall
        all_precision = 0.0
        all_recall = 0.0
        for userIdx in range(len(userScores)):
            precision, recall = Eva.calcPrecisionAndRecall(top_poi_ids[userIdx], userList[userIdx])
            all_precision += precision
            all_recall += recall

        expected_precision.append(all_precision / needCalcUser)
        expected_recall.append(all_recall / needCalcUser)

        # 算MAP, NDCG
        all_map = 0.0
        all_ndcg = 0.0
        for userIdx in range(len(userScores)):
            t_map = Eva.calcMAP(top_poi_ids[userIdx], userList[userIdx])
            t_ndcg = Eva.calcNDCG(top_poi_ids[userIdx], userList[userIdx])
            all_map += t_map
            all_ndcg += t_ndcg

        expected_map.append(all_map / needCalcUser)
        expected_ndcg.append(all_ndcg / needCalcUser)
    return expected_rmse, expected_mae, expected_precision, expected_recall, expected_map, expected_ndcg

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=32, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train')
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
                                              torch.FloatTensor(train_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))
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

    # item feature: user * rating
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device, uv=False)

    # model
    graphrec = GraphRec(enc_u, enc_v_history, r2e).to(device)
    optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=args.lr, alpha=0.9)

    ks = [5,10,15,20]
    best_rmse = 9999.0
    best_mae = 9999.0
    best_precision = np.zeros(len(ks)).tolist()
    best_recall = np.zeros(len(ks)).tolist()
    best_map = np.zeros(len(ks)).tolist()
    best_ndcg = np.zeros(len(ks)).tolist()
    endure_count = 0

    for epoch in range(1, args.epochs + 1):
        train(graphrec, device, train_loader, optimizer, epoch, best_rmse, best_mae, best_precision, best_recall, best_map, best_ndcg, ks)
        expected_rmse, expected_mae, expected_precision, expected_recall, expected_map, expected_ndcg = test(graphrec, device, test_loader, history_u_lists, ks)
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

        for idx in range(len(ks)):
            if best_precision[idx] < expected_precision[idx]:
                best_precision[idx] = expected_precision[idx]
            if best_recall[idx] < expected_recall[idx]:
                best_recall[idx] = expected_recall[idx]
            if best_map[idx] < expected_map[idx]:
                best_map[idx] = expected_map[idx]
            if best_ndcg[idx] < expected_ndcg[idx]:
                best_ndcg[idx] = expected_ndcg[idx]
            print("Evaluation@%d: precision: %.4f, recall:%.4f, map:%.4f, ndcg:%.4f " % (ks[idx], expected_precision[idx], expected_recall[idx], expected_map[idx], expected_ndcg[idx]))

        if endure_count > 5:
            break

        print('this embedding size as ', embed_dim)
        print('this learning rate as ', args.lr)


if __name__ == "__main__":
    start = time.time()
    main()
    print('time cost:', time.time() - start)

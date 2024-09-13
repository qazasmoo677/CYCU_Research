import torch
import torch.nn as nn
import torch.optim as optim

class SentimentGNNCombiner(nn.Module):
    def __init__(self):
        super(SentimentGNNCombiner, self).__init__()
        self.fc1 = nn.Linear(3073, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, gnn_score, sentiment_vector):
        if gnn_score.dim() == 1:
            gnn_score = gnn_score.unsqueeze(1)
            
        x = torch.cat((gnn_score, sentiment_vector), dim=1)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
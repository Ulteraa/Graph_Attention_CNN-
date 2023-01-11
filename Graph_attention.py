import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from rdkit import Chem
import pandas as pd
import torch.optim as optim
import torch_geometric.nn as GNN
from datase_setup import  MyOwnDataset
from torch_geometric.data import DataLoader
import csv
import seaborn as sns
import sklearn.metrics as sk_metric
from tqdm import tqdm
def balanced_data():
    file = "HIV.csv"
    data_ = "data_file.csv"
    negative_ = []; posetive_ = []
    with open(file, 'r') as origFile:
            csvreader = csv.DictReader(origFile)
            for line in csvreader:
                if line['HIV_active'] == '0':
                    negative_.append(line)
                else:
                    posetive_.append(line)
    header=['smiles', 'activity', 'HIV_active']
    with open(data_, 'w', newline='') as prosses_:
        csvwriter = csv.writer(prosses_)
        csvwriter.writerow(header)
        for index in range(len(posetive_)):
            csvwriter.writerow([posetive_[index]['smiles'], posetive_[index]['activity'], int(posetive_[index]['HIV_active'])])
            csvwriter.writerow([negative_[index]['smiles'], negative_[index]['activity'], int(negative_[index]['HIV_active'])])
    prosses_.close()
    origFile.close()

    # data = pd.read_csv('HIV.csv')
    # negative_ = []; posetive = []
    # for i in range(len(data)):
    #     negative_.append(i) if data.values[i][2]==0 else posetive.append(i)

class GCNN_attention(nn.Module):
    def __init__(self, feature_dim):
        super(GCNN_attention, self).__init__()
        embed_dim = 1024
        num_class = 2
        self.conv1 = GNN.GATConv(feature_dim, embed_dim, heads=3, dropout=0.2)
        self.lnr_1 = nn.Linear(3*embed_dim, embed_dim)
        self.pool_1 = GNN.TopKPooling(embed_dim, ratio=0.7)

#-----------------------------------------------------------------------

        self.conv2 = GNN.GATConv(embed_dim, embed_dim, heads=3, dropout=0.2)
        self.lnr_2 = nn.Linear(3*embed_dim, embed_dim)
        self.pool_2 = GNN.TopKPooling(embed_dim, ratio=0.7)

#--------------------------------------------------------------------------

        self.conv3 = GNN.GATConv(embed_dim, embed_dim, heads=3, dropout=0.2)
        self.lnr_3 = nn.Linear(3*embed_dim, embed_dim)
        self.pool_3 = GNN.TopKPooling(embed_dim, ratio=0.7)

#--------------------------------------------------------------------------

        self.linear_1 = nn.Linear(2*embed_dim, embed_dim)
        self.relu = nn. ReLU()
        self.drop_out = nn.Dropout(0.3)
        self.linear_2 = nn.Linear(embed_dim, num_class)

    def forward(self, x, edge_att, edge_index, batch_index):
        x = self.conv1(x, edge_index)
        x = self.lnr_1(x)
        x, edge_index, edge_att, batch_index, _, _ = self.pool_1(x, edge_index, None, batch_index)
        x1 = torch.cat((GNN.global_mean_pool(x, batch_index),GNN.global_max_pool(x, batch_index)), dim=1)
        #------------------------------------------------------
        x = self.conv2(x, edge_index)
        x = self.lnr_2(x)
        x, edge_index, edge_att, batch_index, _, _ = self.pool_2(x,edge_index, None, batch_index)
        x2 = torch.cat((GNN.global_mean_pool(x, batch_index), GNN.global_max_pool(x, batch_index)), dim=1)
        #--------------------------------------------------------------------------
        x = self.conv3(x, edge_index)
        x = self.lnr_3(x)
        x, edge_index, edge_att, batch_index, _, _ = self.pool_3(x, edge_index, None, batch_index)
        x3 = torch.cat((GNN.global_mean_pool(x, batch_index), GNN.global_max_pool(x, batch_index)), dim=1)
        #----------------------------------------------------------------------

        x = x1+x2+x3
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.drop_out(x)
        x = self.linear_2(x)
        return x

def train(GGN_model, optimizer, train_data, criterion, device, epoch):
    loss = 0
    for index, batch in enumerate(tqdm(train_data)):
        batch = batch.to(device)
        predict_ = GGN_model(batch.x, batch.edge_attr, batch.edge_index, batch.batch)
        loss_ = criterion(predict_, batch.y)
        loss += loss_
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
    return loss
    # if epoch % 100 == 0:
    #     print(f'loss at epoch {epoch} is equal to {loss}')

def train_fn():
    train_dataset = MyOwnDataset(root='prosses_data/train', file_name_='train.csv')
    train_data = DataLoader(train_dataset, batch_size=64, shuffle=True)
    GNN_model = GCNN_attention(feature_dim=train_dataset[0].x.shape[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GNN_model = GNN_model.to(device)
    optimizer = optim.SGD(GNN_model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    # scheduler=optim.lr_scheduler.ExponentialLR()
    loss_arr = []
    for epoch in range(2001):
        loss = train(GNN_model, optimizer, train_data, criterion, device, epoch)
        loss_arr.append(loss_arr)
        if epoch % 100 == 0:
            print(f'testing the method with various metrics at epoch {epoch}')
            with torch.no_grad():
                 test(GNN_model, device)
    visulize(loss_arr)

def test(GNN_model, device):
    test_dataset = MyOwnDataset(root='prosses_data/test', file_name_='test.csv')
    print(len(test_dataset))
    batch = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for _, batch in enumerate(batch):
            batch = batch.to(device)
            predict_ = GNN_model(batch.x, batch.edge_attr, batch.edge_index, batch.batch)
            predict_ = torch.argmax(predict_, dim=1)
            compute_acc(predict_.cpu().detach().numpy(), batch.y.cpu().detach().numpy())

def visulize(loss_arr):
    losses_float = [float(loss_arr.cpu().detach().numpy()) for loss in loss_arr]
    plt_ = sns.lineplot(losses_float)
    plt_.set(xlabel='epoch', ylabel='error')
    plt.savefig('train.png')

def compute_acc(predic, ground_trouth):
    print(f"\n Confusion matrix: \n {sk_metric.confusion_matrix(predic, ground_trouth)}")
    print(f"F1 Score: {sk_metric.f1_score(ground_trouth, predic)}")
    print(f"Accuracy: {sk_metric.accuracy_score(ground_trouth, predic)}")
    prec = sk_metric.precision_score(ground_trouth, predic)
    rec = sk_metric.recall_score(ground_trouth, predic)
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    roc = sk_metric.roc_auc_score(ground_trouth, predic)
    print(f"ROC AUC: {roc}")

if __name__=='__main__':
    #balanced_data()
    train_fn()














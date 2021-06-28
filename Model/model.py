import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
from torch.autograd import Variable

# Seed
SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)
model_path = "./Model/"

# GraphPPIS parameters
MAP_CUTOFF = 14
HIDDEN_DIM = 256
LAYER = 8
DROPOUT = 0.1
ALPHA = 0.7
LAMBDA = 1.5
VARIANT = True # From GCNII

LEARNING_RATE = 1E-3
WEIGHT_DECAY = 0
BATCH_SIZE = 1
NUM_CLASSES = 2 # [not bind, bind]

device = torch.device('cpu')


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


def load_features(ID, data_path, mode):
    if mode == "fast":   
        blosum_feature = np.load(data_path + "blosum/" + ID + '.npy')
        dssp_feature = np.load(data_path + "dssp/" + ID + '.npy')
        node_features = np.concatenate([blosum_feature, dssp_feature], axis = 1).astype(np.float32)
    else:
        pssm_feature = np.load(data_path + "pssm/" + ID + '.npy')
        hhm_feature = np.load(data_path + "hhm/" + ID + '.npy')
        dssp_feature = np.load(data_path + "dssp/" + ID + '.npy')
        node_features = np.concatenate([pssm_feature, hhm_feature, dssp_feature], axis = 1).astype(np.float32)
    return node_features


def load_graph(ID, data_path):
    matrix = np.load(data_path + 'dismap/' + ID + '.npy').astype(np.float32)
    matrix = normalize(matrix)
    return matrix


class ProDataset(Dataset):
    def __init__(self, dataframe, data_path, mode):
        self.IDs = dataframe['ID'].values
        self.data_path = data_path
        self.mode = mode

    def __getitem__(self, index):
        ID = self.IDs[index]

        # L * 34/54
        node_features = load_features(ID, self.data_path, self.mode)
        # L * L
        adjacency_matrix = load_graph(ID, self.data_path)

        return node_features, adjacency_matrix

    def __len__(self):
        return len(self.IDs)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = min(1, math.log(lamda/l+1))
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual: # speed up convergence of the training process
            output = output+input
        return output


class deepGCN(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha, variant):
        super(deepGCN, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant,residual=True))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return layer_inner


class GraphPPIS(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha, variant):
        super(GraphPPIS, self).__init__()

        self.deep_gcn = deepGCN(nlayers = nlayers, nfeat = nfeat, nhidden = nhidden, nclass = nclass,
                                dropout = dropout, lamda = lamda, alpha = alpha, variant = variant)
        self.criterion = nn.CrossEntropyLoss() # automatically do softmax to the predicted value and one-hot to the label
        self.optimizer = torch.optim.Adam(self.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

    def forward(self, x, adj):          # x.shape = (seq_len, FEATURE_DIM); adj.shape = (seq_len, seq_len)
        x = x.float()
        output = self.deep_gcn(x, adj)  # output.shape = (seq_len, NUM_CLASSES)
        return output


def evaluate(model, data_loader):
    model.eval()
    pred = []

    for data in data_loader:
        with torch.no_grad():
            node_features, adjacency_matrix = data

            node_features = Variable(node_features)
            graphs = Variable(adjacency_matrix)

            node_features = torch.squeeze(node_features)
            graphs = torch.squeeze(graphs)

            y_pred = model(node_features, graphs)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            y_pred = y_pred.cpu().detach().numpy()
            pred += [pred[1] for pred in y_pred]

    return pred


def test(test_dataframe, data_path, mode):
    test_loader = DataLoader(dataset=ProDataset(test_dataframe, data_path, mode), batch_size=BATCH_SIZE, shuffle=False, num_workers=3)

    INPUT_DIM = (34 if mode == "fast" else 54)
    GraphPPIS_model = GraphPPIS(LAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT, LAMBDA, ALPHA, VARIANT)
    GraphPPIS_model.load_state_dict(torch.load(model_path + "GraphPPIS_{}.pkl".format(mode), map_location = device))

    test_pred = evaluate(GraphPPIS_model, test_loader)

    return test_pred

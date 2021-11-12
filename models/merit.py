import numpy as np
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from layers.readout import AvgReadout
from layers.discriminator import Discriminator


class MLP(nn.Module):

    def __init__(self, inp_size, outp_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, outp_size)
        )

    def forward(self, x):
        return self.net(x)


class GraphEncoder(nn.Module):

    def __init__(self, 
                  gnn,
                  projection_hidden_size,
                  projection_size):
        
        super().__init__()
        
        self.gnn = gnn
        self.projector = MLP(512, projection_size, projection_hidden_size)           
        
    def forward(self, adj, in_feats, sparse):
        representations = self.gnn(in_feats, adj, sparse)
        representations = representations.view(-1, representations.size(-1))
        projections = self.projector(representations)  # (batch, proj_dim)
        return projections

    
class EMA():
    
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def sim(h1, h2):
    z1 = F.normalize(h1, dim=-1, p=2)
    z2 = F.normalize(h2, dim=-1, p=2)
    return torch.mm(z1, z2.t())


def consistency_loss(h1, h2):
    x = F.normalize(h1, dim=-1, p=2)
    y = F.normalize(h2, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def neg_pos_label(nodes_num):
    x = torch.ones(1, nodes_num)
    y = torch.zeros(1, nodes_num)
    lbl = torch.cat((x, y), 1)
    return lbl


class MERIT(nn.Module):
    
    def __init__(self, 
                 gnn,
                 feat_size,
                 projection_size, 
                 projection_hidden_size,
                 prediction_size,
                 prediction_hidden_size,
                 moving_average_decay,
                 beta):
        
        super().__init__()

        self.online_encoder = GraphEncoder(gnn, projection_hidden_size, projection_size)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.negative_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(self.target_encoder, False)
        set_requires_grad(self.negative_encoder, False)
        self.target_ema_updater = EMA(moving_average_decay)
        self.negative_ema_updater = EMA(moving_average_decay)
        self.online_predictor = MLP(projection_size, prediction_size, prediction_hidden_size)

        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(projection_size)

        #self.bce = nn.BCEWithLogitsLoss()

        self.beta = beta
                   
    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_ma(self):
        assert self.target_encoder and self.negative_encoder is not None, 'target or negative encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)
        update_moving_average(self.negative_ema_updater, self.negative_encoder, self.online_encoder)

    def forward(self, aug_adj_1, aug_adj_2, aug_feat_1, aug_feat_2, adj, shuf_fts, sparse, msk = None, samp_bias1 = None, samp_bias2 = None):

        online_proj = self.online_encoder(aug_adj_1, aug_feat_1, sparse)
        online_pred = self.online_predictor(online_proj)

        with torch.no_grad():
            target_proj = self.target_encoder(aug_adj_2, aug_feat_2, sparse)
            negative_proj = self.negative_encoder(adj, shuf_fts, sparse)


        l1 = consistency_loss(online_pred, target_proj)

        online_pred_axis = online_pred[np.newaxis]
        global_view = self.read(online_pred_axis, msk)
        global_view = self.sigm(global_view)

        negative_proj_axis = negative_proj[np.newaxis]
        logits = self.disc(global_view, online_pred_axis, negative_proj_axis, samp_bias1, samp_bias2)
        lbl = neg_pos_label(adj.shape[0])
        #l2 = self.bce(logits, lbl)
        loss = l1.mean()

        return loss, logits, lbl





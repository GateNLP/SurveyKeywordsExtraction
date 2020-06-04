import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo

class FeedForward(nn.Module):
    def __init__(self, d_model, d_out, dropout = 0.1):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        return x


class ElmoEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        elmo_path = config['elmo']
        elmo_option_file = os.path.join(elmo_path, "elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json")
        elmo_weight_file = os.path.join(elmo_path, "elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5")
        self.elmo = Elmo(elmo_option_file, elmo_weight_file, 2)
        for p in self.elmo.parameters():
            p.requires_grad = False

    def forward(self, x):
        elmo_embed = self.elmo(x)['elmo_representations']
        cat_embd = torch.cat(elmo_embed, 2)
        return cat_embd


class QuestionFocusModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        elmo_path = config['elmo']
        elmo_option_file = os.path.join(elmo_path, "elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json")
        elmo_weight_file = os.path.join(elmo_path, "elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5")
        self.elmo = Elmo(elmo_option_file, elmo_weight_file, 2)
        for p in self.elmo.parameters():
            p.requires_grad = False

        self.elmo_reduction = FeedForward(1024*2, 300)

        self.v_cls = nn.Parameter(torch.randn([300,1]).float())
        self.k = FeedForward(300, 300)
        self.v = FeedForward(300, 300)
        self.softmax_simi = nn.Softmax(dim=1)


    def forward(self, tensor_input):
        elmo_embed = self.elmo(tensor_input)['elmo_representations']
        cat_embd = torch.cat(elmo_embed, 2)
        redu_embd = self.elmo_reduction(cat_embd)
        linear_key = self.k(redu_embd)
        linear_value = self.v(redu_embd)
        dotProducSimi = linear_key.matmul(self.v_cls)
        normedSimi = self.softmax_simi(dotProducSimi)
        attVector = linear_value.mul(normedSimi)
        weightedSum = torch.sum(attVector, dim=1)
        return weightedSum, dotProducSimi





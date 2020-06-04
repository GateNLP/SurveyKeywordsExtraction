import torch
import torch.nn as nn
from allennlp.modules.elmo import Elmo
import torch.nn.functional as F
import os
from .QuestionFocusModel import FeedForward

class QuestionMatchingModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        elmo_dim = 1024*2
        self.v_cls = nn.Parameter(torch.randn([300,1]).float())
        self.k = FeedForward(elmo_dim, 300)
        self.v = FeedForward(elmo_dim, elmo_dim)
        self.softmax_simi = nn.Softmax(dim=1)


    def forward(self, tensor_input):
        #print(tensor_input.shape)
        linear_key = self.k(tensor_input)
        linear_value = self.v(tensor_input)
        dotProducSimi = linear_key.matmul(self.v_cls)
        normedSimi = self.softmax_simi(dotProducSimi)
        attVector = linear_value.mul(normedSimi)
        weightedSum = torch.sum(attVector, dim=1)
        #print(weightedSum.shape)
        return weightedSum, dotProducSimi





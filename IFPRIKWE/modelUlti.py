import sys
import numpy as np
import os
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import torch.optim as optim
import math

class modelUlti:
    def __init__(self, net, gpu=False):
        print(gpu)
        self.gpu = gpu
        self.net = net
        self.optimizer = optim.Adam(self.net.parameters())
        self.criterion = nn.CrossEntropyLoss()
        if self.gpu:
            self.net.cuda()
            self.criterion.cuda()

    def train(self, batchGen, num_epoch=10):
        for epoch in range(num_epoch):
            all_loss = []
            i=0
            print("processing epoch", epoch)
            trainIter = self.pred(batchGen, train=True)
            for y, attw, loss_value, _ in trainIter:
                all_loss.append(loss_value)
            print(sum(all_loss)/len(all_loss))
        
    def pred(self, batchGen, train=False):
        i=0
        loss_value = 0
        if train:
            self.net.train()
            self.optimizer.zero_grad()
        else:
            self.net.eval()

        if hasattr(batchGen, 'tensorinputType'):
            tensorinputType = batchGen.tensorinputType
        else:
            if self.gpu:
                tensorinputType = torch.cuda.LongTensor
            else:
                tensorinputType = torch.LongTensor

        if hasattr(batchGen, 'tensorlabelType'):
            tensorlabelType = batchGen.tensorlabelType
        else:
            if self.gpu:
                tensorlabelType = torch.cuda.LongTensor
            else:
                tensorlabelType = torch.LongTensor

        for input_tensor, label in batchGen:
            print("processing batch", i, end='\r')
            if self.gpu:
                label = label.type(tensorlabelType)
                input_tensor = input_tensor.type(tensorinputType)
                label.cuda()
                input_tensor.cuda()

            y, attw=self.net(input_tensor)
            if train:
                #print(y.shape, label.shape)
                loss = self.criterion(y, label)
                loss.backward()
                self.optimizer.step()
                loss_value = float(loss.data.item())
                self.optimizer.zero_grad()
            i+=1
            yield y, attw, loss_value, label

    def eval(self, batchGen):
        all_prediction = []
        all_true_label = []
        for y, attw, loss_value, label in self.pred(batchGen):
            current_batch_out = F.softmax(y, dim=-1)
            label_prediction = torch.max(current_batch_out, -1)[1]
            current_batch_out_list = current_batch_out.to('cpu').detach().numpy()
            label_prediction_list = label_prediction.to('cpu').detach().numpy()
            label_list = label.to('cpu').detach().numpy()
            all_prediction.append(label_prediction_list)
            all_true_label.append(label_list)

        all_prediction = np.concatenate(all_prediction)
        all_true_label = np.concatenate(all_true_label)
        num_correct = (all_prediction == all_true_label).sum()
        accuracy = num_correct / len(all_prediction)
        print(accuracy)

    def saveWeights(self, save_path='.'):
        model_path = os.path.join(save_path, 'net.weights')
        torch.save(self.net.state_dict(), model_path)

    def saveLabels(self, labels, save_path='.'):
        label_save_path = os.path.join(save_path, 'net.labels')
        with open(label_save_path, 'w') as fo:
            fo.write('\t'.join(labels))

    def loadWeights(self, load_path='.', cpu=True):
        model_path = os.path.join(load_path, 'net.weights')
        if cpu:
            self.net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
        else:
            self.net.load_state_dict(torch.load(model_path), strict=False)
        self.net.eval()

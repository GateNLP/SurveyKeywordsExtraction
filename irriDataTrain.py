import os
import math
import sys
import pandas as pd
import nltk
import copy
import torch
import torch.nn as nn
from IFPRIKWE import IFPRI_ExcelReader_TypeB, modelUlti
from IFPRIKWE.models import QuestionMatchingModel
from allennlp.modules.elmo import batch_to_ids
import random
from pathlib import Path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("irriTrainxls", help="irri labelled training file")
    parser.add_argument("--gpu", default=False, action='store_true', help="use gpu in training")
    parser.add_argument("--num_epoches", type=int, default=10, help="num epoches")

    args = parser.parse_args()
    script_path = os.path.abspath(__file__)
    parent = os.path.dirname(script_path)
    elmo_folder = os.path.join(parent, 'elmo')
    cust_config = {}
    cust_config['elmo'] = elmo_folder
    trainGen = IFPRI_ExcelReader_TypeB(args.irriTrainxls, elmoConfig=cust_config, gpu=args.gpu, with_label=True)
    net = QuestionMatchingModel(cust_config)
    model = modelUlti(net)
    model.criterion = nn.KLDivLoss()
    model.train(trainGen, num_epoch=args.num_epoches)
    model.saveWeights()

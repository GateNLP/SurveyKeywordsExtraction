import os
import math
import sys
import pandas as pd
import nltk
import copy
import torch
import torch.nn as nn
from IFPRIKWE import modelUlti
#from IFPRIKWE.models import QuestionMatchingModel
from allennlp.modules.elmo import batch_to_ids
import random
from pathlib import Path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trainingFile", help="labelled training file")
    parser.add_argument("--gpu", default=False, action='store_true', help="use gpu in training")
    parser.add_argument("--num_epoches", type=int, default=10, help="num epoches")
    parser.add_argument("--model", default="irri", help="model, default irri, models: irri, qsclass")
    parser.add_argument("--output", default="", help="model output path")

    args = parser.parse_args()
    training_file = args.trainingFile
    script_path = os.path.abspath(__file__)
    parent = os.path.dirname(script_path)
    elmo_folder = os.path.join(parent, 'elmo')
    cust_config = {}
    cust_config['elmo'] = elmo_folder

    if args.model == 'irri':
        from IFPRIKWE.models import QuestionMatchingModel
        from IFPRIKWE import IFPRI_ExcelReader_TypeB
        trainGen = IFPRI_ExcelReader_TypeB(training_file, elmoConfig=cust_config, gpu=args.gpu, with_label=True)
        net = QuestionMatchingModel(cust_config)
        model = modelUlti(net, gpu=args.gpu)
        model.criterion = nn.KLDivLoss()
    elif args.model == 'qsclass':
        from IFPRIKWE.models import QuestionClassificationModel
        from IFPRIKWE import QuestionClassifierReader
        trainGen = QuestionClassifierReader(training_file)
        cust_config['nClasses'] = len(trainGen.target_labels)
        net = QuestionClassificationModel(cust_config)
        model = modelUlti(net, gpu=args.gpu)

    #model = modelUlti(net)
    #model.criterion = nn.KLDivLoss()
    model.train(trainGen, num_epoch=args.num_epoches)
    model.saveWeights()

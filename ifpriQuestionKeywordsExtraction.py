import sys
import os
import pandas as pd
import nltk
import copy
from IFPRIKWE import modelUlti
from IFPRIKWE.models import QuestionMatchingModel, ElmoEmbedding
from allennlp.modules.elmo import batch_to_ids
from nltk.corpus import stopwords
import torch
from pathlib import Path
import argparse

from IFPRIKWE import IFPRI_ExcelReader_TypeB
from IFPRIKWE import IFPRI_ExcelReader_TypeA

class QuestionAttModelUlti(modelUlti):
    def __init__(self, net, elmoEmbedder, gpu=True):
        super().__init__(net, gpu=gpu)
        self.elmoEmbedder = elmoEmbedder


    def setStopwords(self, userlistfile=None):
        self.stopwords = set(stopwords.words('english'))
        print(self.stopwords)

        if userlistfile:
            with open(userlistfile, 'r') as fi:
                for line in fi:
                    self.stopwords.add(line.strip())


    def getAttWeights(self, questionReader):
        sorted_list = []
        original_questions = []
        for input_tensor,_ in questionReader:
            if self.gpu:
                input_tensor = input_tensor.type(torch.cuda.LongTensor)
                input_tensor.cuda()
            else:
                input_tensor = input_tensor.type(torch.LongTensor)
            embedded = self.elmoEmbedder(input_tensor)
            _, attw=self.net(embedded)
            raw_text = questionReader.selected_texts
            #check_id = 0
            #print(raw_text[check_id])
            #print(attw[check_id])
            maxatt = torch.argmax(attw, dim=1)
            #print(maxatt.shape)

            att_shape = attw.shape
            num_queston = att_shape[0]
            num_words = att_shape[1]
            
            for eachDoc_id in range(num_queston):
                sorted_list.append([])
                current_question = raw_text[eachDoc_id]
                original_questions.append(' '.join(current_question))
                for current_word_id, current_word in enumerate(current_question):
                    current_att_score = attw[eachDoc_id][current_word_id]
                    sorted_list[-1].append([current_word, current_att_score.data.item()])
                sorted_list[-1].sort(key=lambda tup: tup[1], reverse=True)

        return sorted_list, original_questions
        #print(sorted_list)

    def filter_by_stopwords(self, sorted_list):
        for question_id in range(len(sorted_list)):
            new_list = []
            selected_words = []
            for word_id in range(len(sorted_list[question_id])):
                current_scored_word = sorted_list[question_id][word_id]
                if (len(current_scored_word[0]) > 1) and (current_scored_word[0] not in self.stopwords) and (current_scored_word[0] not in selected_words):
                    selected_words.append(current_scored_word[0])
                    new_list.append(current_scored_word)
            sorted_list[question_id] = new_list
        return sorted_list


    def selectKeyWords(self, questionReader):
        output_line = 'question\ttop3words\n'
        sorted_list, original_questions = self.getAttWeights(questionReader)
        sorted_list = self.filter_by_stopwords(sorted_list)
        for question_id, scored_question in enumerate(sorted_list):
            all_words = [item[0] for item in scored_question]
            current_line = original_questions[question_id] + '\t'+' | '.join(all_words[:3])+'\n'
            output_line+=current_line
            #print(original_questions[question_id])
            #print(scored_question[:3])
        return output_line
        

            
def loadLabels(load_path='.'):
    label_path = os.path.join(load_path, 'net.labels')
    with open(label_path, 'r') as fi:
        line = next(fi)
        labels = line.split('\t')
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputxls", help="excel file input for keywords extraction")
    parser.add_argument("outputTsv", help="tsv output for extracted keywords")
    parser.add_argument("--modelPath", help="file to trained model")
    parser.add_argument("--stopwords", help="file to stop words")
    parser.add_argument("--excelType", default='typeB', help="excelType")
    parser.add_argument("--gpu", default=False, action='store_true', help="use gpu in training")

    parser.add_argument("--target_field", default='PO_0009010', help="target field")
    parser.add_argument("--question_field", default='Survey term', help="Survey term")
    parser.add_argument("--term_field", default='Term', help="Term")
    parser.add_argument("--options_field", default='Options', help="Options")
    parser.add_argument("--sheetId", default=0, type=int, help="sheetName")
    args = parser.parse_args()
    script_path = os.path.abspath(__file__)
    parent = os.path.dirname(script_path)
    elmo_folder = os.path.join(parent, 'elmo')
    cust_config = {}
    cust_config['elmo'] = elmo_folder
    elmoEmbedder = ElmoEmbedding(cust_config)
    elmoEmbedder.to('cpu')
    elmoEmbedder.eval()
    if args.gpu:
        elmoEmbedder.cuda()

    net = QuestionMatchingModel(cust_config)
    net.to('cpu')
    model = QuestionAttModelUlti(net, elmoEmbedder, gpu=args.gpu)

    if args.modelPath:
        model.loadWeights(args.modelPath)
    else:
        model.loadWeights(parent)

    if args.stopwords:
        model.setStopwords(args.stopwords)
    else:
        stopwordspath = os.path.join(parent, 'stopwords.lst')
        model.setStopwords(stopwordspath)

    if args.excelType == 'typeB':
        from IFPRIKWE import IFPRI_ExcelReader_TypeB
        questionReader = IFPRI_ExcelReader_TypeB(args.inputxls, gpu=args.gpu, sheetName=args.sheetId, target_field=args.target_field, question_field=args.question_field, term_field=args.term_field, option_field=args.options_field)
    elif args.excelType == 'typeA':
        from IFPRIKWE import IFPRI_ExcelReader_TypeA
        questionReader = IFPRI_ExcelReader_TypeA(args.inputxls)



    output_tsv_text = model.selectKeyWords(questionReader)
    with open(args.outputTsv, 'w') as fo:
        fo.write(output_tsv_text)


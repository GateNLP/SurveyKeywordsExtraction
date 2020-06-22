from .IFPRI_Reader_base import IFPRIReader
import torch
import nltk
import pandas as pd
import re
from allennlp.modules.elmo import batch_to_ids

class IFPRI_ExcelReader_TypeB(IFPRIReader):
    def __init__(self, excel_file, 
            sheetName=1, 
            batch_size=32, 
            elmoConfig=None, 
            gpu=False, 
            with_label=False, 
            mergeOptions=False, 
            target_field='PO_0009010', 
            question_field='Survey term', 
            term_field='Term', 
            option_field='Options'):
        super().__init__(batch_size=batch_size)
        self.target_field = target_field
        self.question_field = question_field
        self.term_field = term_field
        self.option_field = option_field

        self.mergeOptions = mergeOptions
        self.with_label = with_label
        self.readQuestions(excel_file, sheetName)
        self.elmoEmbd = None
        self.gpu = gpu

        if elmoConfig:
            from .models import ElmoEmbedding
            self.elmoEmbd = ElmoEmbedding(elmoConfig)
            self.elmoEmbd.eval()
            if gpu:
                self.tensorinputType = torch.cuda.FloatTensor
                self.tensorlabelType = torch.cuda.FloatTensor
                self.elmoEmbd.cuda()
            else:
                self.tensorinputType = torch.FloatTensor
                self.tensorlabelType = torch.FloatTensor


    def readQuestions(self, excel_file, sheetName):
        onto_targets = []
        survy_pd_frame = pd.read_excel(excel_file, sheet_name=sheetName, header=0)
        column_list = list(survy_pd_frame.columns)
        print(column_list)
        if self.with_label:
            list_of_targets = list(survy_pd_frame[self.target_field].unique())
            for item in list_of_targets:
                #print(pd.isna(item))
                if pd.isna(item) == False:
                    current_target = item.strip().lower()
                    if current_target not in onto_targets:
                        onto_targets.append(current_target)
        else:
            list_of_targets = [None]
            onto_targets = [None]
        print(list_of_targets)
        print(len(list_of_targets))
        print(len(onto_targets))


        for eachrow in survy_pd_frame.iterrows():
            #print(eachrow)
            try:
                onto_class = None
                onto_terms = None
                survey_choice = None
                survey_term = eachrow[1][self.question_field]
                print(survey_term)
                survey_term = re.sub('\n', ' ', survey_term)
                survey_term = re.sub('^Q\.?\d+\.?\s', '', survey_term)
                if self.with_label: 
                    onto_class = eachrow[1][self.target_field]
                    onto_terms = eachrow[1][self.term_field]
                    #survey_choice = eachrow[1]['Options']

                if self.mergeOptions:
                    survey_choice = eachrow[1][self.option_field]
                    survey_term = survey_term.strip()+' '+survey_choice
                
                if pd.isna(onto_class) == False or (not self.with_label):
                    question_tok = nltk.word_tokenize(survey_term.lower())
                    survey_choices = survey_choice
                    target = nltk.word_tokenize(str(onto_terms).lower())

                    self.all_questions_list.append([question_tok, survey_choices, target])
            except:
                print('bad line')

    def _postProcess(self):
        if self.with_label:
            lab_idx = batch_to_ids(self.selected_labels)
            if self.gpu:
                lab_idx = lab_idx.type(torch.cuda.LongTensor)
            lab_idx = self.elmoEmbd(lab_idx)
            lab_idx = torch.sum(lab_idx, dim=1)
        else:
            lab_idx = None

        str_idx = batch_to_ids(self.selected_texts)
        if self.elmoEmbd:
            if self.gpu:
                str_idx = str_idx.type(torch.cuda.LongTensor)
                #lab_idx = lab_idx.type(torch.cuda.LongTensor)
            str_idx = self.elmoEmbd(str_idx)
            #lab_idx = self.elmoEmbd(lab_idx)
            #lab_idx = torch.sum(lab_idx, dim=1)

        return str_idx, lab_idx

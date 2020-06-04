from allennlp.modules.elmo import batch_to_ids
import torch

class IFPRIReader:
    def __init__(self, batch_size=32):
        self.all_questions_list = [] #self.all_questions_list = [[question_tok, list_choices, target],...]
        self.batch_size = batch_size
        self.selected_texts = []
        self.selected_labels = []
        self.label2idx = False

    def __iter__(self):
        i=0
        for each_question in self.all_questions_list:
            if i == 0:
                self.selected_texts = []
                self.selected_labels = []
            question_tok = each_question[0]
            question_choice = each_question[1]
            target_label = each_question[2]
            self.selected_texts.append(question_tok)
            if target_label:
                self.selected_labels.append(target_label)
            i+=1
            if i == self.batch_size:
                i=0
                yield self._postProcess()
        if i>0:
            i=0
            yield self._postProcess()

    def label2ids(self):
        lab_idx = []
        for raw_labem in self.selected_labels:
            lab_idx.append(self.target_labels.index(raw_labem))
        return torch.tensor(lab_idx)

    def _postProcess(self):
        str_idx = batch_to_ids(self.selected_texts)
        if self.label2idx:
            lab_idx = self.label2ids()
        else:
            lab_idx = self.selected_labels
        return str_idx, lab_idx


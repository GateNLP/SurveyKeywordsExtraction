import torch
import torch.nn as nn
from .QuestionFocusModel import QuestionFocusModel

class QuestionClassificationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_classes = config["nClasses"]
        self.questionFocusModel = QuestionFocusModel(config)
        self.layer_output = torch.nn.Linear(300, self.n_classes)

    def forward(self, tensor_input):
        weightedSum, dotProducSimi = self.questionFocusModel(tensor_input)
        output = self.layer_output(weightedSum)
        return output, dotProducSimi


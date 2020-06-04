from .IFPRI_Reader_base import IFPRIReader
import nltk


class QuestionClassifierReader(IFPRIReader):
    def __init__(self, inputFile, batch_size=64, get_labels=True, label2idx=True):
        super().__init__(batch_size=batch_size)
        self.all_questions_list = [] 
        self.readFile(inputFile)
        self.target_labels = []
        if get_labels:
            self._getLabels()
        self.label2idx = label2idx

    def readFile(self, inputFile):
        with open(inputFile, 'r') as fi:
            for line in fi:
                lineTok = line.split()
                rawLabel = lineTok[0]
                rawText = ' '.join(lineTok[1:])
                label = rawLabel
                textTok = nltk.word_tokenize(rawText.lower())
                self.all_questions_list.append([textTok, None, label])

    def _getLabels(self):
        for each_data in self.all_questions_list:
            _, _, label = each_data
            if label not in self.target_labels:
                self.target_labels.append(label)

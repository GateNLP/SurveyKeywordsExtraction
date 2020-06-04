from .IFPRI_Reader_base import IFPRIReader
import nltk
import pandas as pd

class IFPRI_ExcelReader_TypeA(IFPRIReader):
    def __init__(self, excel_file, batch_size=32):
        super().__init__(batch_size=batch_size)
        self.ignor_type = ['note', 'begin group', 'start', 'calculate', 'end group', 'end']
        self.selecte_type = ['select_one', 'select_multiple']
        self.readQuestions(excel_file)

    def readQuestions(self, excel_file):
        survy_pd_frame = pd.read_excel(excel_file, sheet_name='survey', header=0)
        choice_pd_frame = pd.read_excel(excel_file, sheet_name='choices', header=0, index_col=0)
        column_list = list(survy_pd_frame.columns)
        print(column_list)
        print(list(survy_pd_frame['type'].unique()))

        begin_read = False
        for eachrow in survy_pd_frame.iterrows():
            row_type = eachrow[1]['type']
            if row_type == 'begin group':
                begin_read = True
            elif row_type == 'end group':
                begin_read = False
            else:
                if begin_read:
                    if row_type not in self.ignor_type:
                        question = str(eachrow[1]['label::English (en)']).strip()
                        if not pd.isna(question):
                            row_type_tok = row_type.split()
                            question_tok = nltk.word_tokenize(question.lower())
                            list_choices = []
                            if row_type_tok[0] in self.selecte_type:
                                choice_type = row_type_tok[1]
                                list_choices = list(choice_pd_frame.loc[choice_type]['name'])
                                list_choices = [str(each_choice).lower() for each_choice in list_choices]

                            self.all_questions_list.append([question_tok, list_choices, None])

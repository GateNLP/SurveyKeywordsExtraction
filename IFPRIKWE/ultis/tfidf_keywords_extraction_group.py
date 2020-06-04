import sys
import pandas as pd
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import nltk
import copy


def get_top_words(toks, score_list):
    sorted_tokes = []
    for word, score in score_list:
        if word in toks:
            sorted_tokes.append([word, score])
    return sorted_tokes

excel_file = sys.argv[1]
output_tsv = sys.argv[2]

survy_pd_frame = pd.read_excel(excel_file, sheet_name='survey', header=0)
choice_pd_frame = pd.read_excel(excel_file, sheet_name='choices', header=0, index_col=0)

column_list = list(survy_pd_frame.columns)
print(column_list)
#print(survy_pd_frame['label::English (en)'])
ignor_type = ['note', 'begin group', 'start', 'calculate', 'end group', 'end']
selecte_type = ['select_one', 'select_multiple']
print(list(survy_pd_frame['type'].unique()))

all_questions_dict = {}
group_id = -1
begin_read = False
for eachrow in survy_pd_frame.iterrows():
    row_type = eachrow[1]['type']
    if row_type == 'begin group':
        group_id += 1
        all_questions_dict[group_id] = {}
        all_questions_dict[group_id]['questions'] = []
        all_questions_dict[group_id]['questions_tok'] = []
        all_questions_dict[group_id]['choice_tok'] = []
        all_questions_dict[group_id]['questions_and_choice_tok'] = []
        begin_read = True
    elif row_type == 'end group':
        begin_read = False

    else:
        if begin_read:
            if row_type not in ignor_type:
                question = str(eachrow[1]['label::English (en)']).strip()
                if not pd.isna(question):
                    all_questions_dict[group_id]['questions'].append(question)
                    row_type_tok = row_type.split()
                    question_tok = nltk.word_tokenize(question.lower())
                    all_questions_dict[group_id]['questions_tok'].append(copy.deepcopy(question_tok))
                    if row_type_tok[0] in selecte_type:
                        choice_type = row_type_tok[1]
                        list_choices = list(choice_pd_frame.loc[choice_type]['name'])
                        list_choices = [str(each_choice).lower() for each_choice in list_choices]
                        all_questions_dict[group_id]['choice_tok'].append(list_choices)
                        question_tok += list_choices
                    else:
                        all_questions_dict[group_id]['choice_tok'].append([])

                    all_questions_dict[group_id]['questions_and_choice_tok'].append(question_tok)

num_question_groups = len(all_questions_dict)
all_questions = []
for question_group_id in range(num_question_groups):
    all_questions.append([])
    for each_question in all_questions_dict[question_group_id]['questions_and_choice_tok']:
        all_questions[question_group_id] += each_question

#print(all_questions)




dct = Dictionary(all_questions)
corpus = [dct.doc2bow(line) for line in all_questions]
##print(corpus)
model = TfidfModel(corpus)

with open(output_tsv, 'w') as fo:
    output_line = 'question\ttop3words\tchoices\ttop3wordsIncuChoice\n'
    fo.write(output_line)
    for question_group_id, each_question_group in enumerate(corpus):
        vector = model[each_question_group]
        #print(all_questions[question_group_id])
        sorted_by_second = sorted(vector, key=lambda tup: tup[1], reverse=True)
        sorted_by_second = [[dct[word], score] for word,score in sorted_by_second]
        #print(sorted_by_second)
        for question_id in range(len(all_questions_dict[question_group_id]['questions'])):
            question = all_questions_dict[question_group_id]['questions'][question_id]
            questionTok = all_questions_dict[question_group_id]['questions_tok'][question_id]
            choiceTok = all_questions_dict[question_group_id]['choice_tok'][question_id]
            questionnchoiceTok = all_questions_dict[question_group_id]['questions_and_choice_tok'][question_id]
            topQuestionTok = get_top_words(questionTok, sorted_by_second)
            print(question)
            print(topQuestionTok)
            topQnChoiceTok = get_top_words(questionnchoiceTok, sorted_by_second)
    
            top3wordsQuestion = [topTok[0] for topTok in topQuestionTok[:3]]
            top3wordsQnChoice = [topTok[0] for topTok in topQnChoiceTok[:3]]
    
            output_line = question+'\t'+ ','.join(top3wordsQuestion)+'\t'+','.join(choiceTok)+'\t'+','.join(top3wordsQnChoice)+'\n'
            print(output_line)
            fo.write(output_line)
    
        




















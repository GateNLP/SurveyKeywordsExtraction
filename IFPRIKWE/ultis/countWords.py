import sys
from nltk import word_tokenize


inputFile = sys.argv[1]
print(inputFile)

countDict = {}

with open(inputFile, 'r') as fi:
    for line in fi:
        #print(line)
        lineTok = line.split()
        tokens = lineTok[1:]
        for token in tokens:
            if token not in countDict:
                countDict[token] = 0
            countDict[token]+=1


sorted_list = sorted(countDict.items(), key = lambda kv:(kv[1], kv[0]))
print(sorted_list)




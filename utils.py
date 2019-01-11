import pickle
from langconv import Converter

from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def tokenize(line):
    line = Converter('zh-hans').convert(line)
    res = nlp.annotate(line, properties={
       'annotators': 'pos',
       'outputFormat': 'json'
       })

    tokens = []
    if type(res) == type({}):
        for sentence in res['sentences']:
            for token in sentence['tokens']:
                tokens.append(Converter('zh-hant').convert(token['word']))

    return tokens

def convert(input, dictionary):
    idx_to_word = {dictionary[word]: word for word in dictionary}
    res = []
    for line in input:
        tmp = ''
        for idx in line:
            if idx == 1:
                break
            elif idx == 0:
                continue
            else:
                tmp += idx_to_word[idx]
        res.append(tmp)
    return res

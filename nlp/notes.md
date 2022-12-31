tutorial = 'https://www.machinelearningplus.com/nlp/text-summarization-approaches-nlp-example/'

# file=r'C:\Users\Dell\learnPython\nlp\sample.txt'

# with open(file) as f:
#     data = f.read()

two methods:
(a) Extractive : keep important sentence
(b) Abstractive : create new sentence

tokenize = split sentences into words using nltk or spacy :: more cleverly than simply splitting on space

from nltk.tokenize import sent_tokenize, word_tokenize

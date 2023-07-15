import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
# import csv
import time
from nltk.stem.wordnet import WordNetLemmatizer

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
from num2words import num2words


# 1. Import Vocabulary from Task1
# 2. Normalise terms of the vocabulary:
#    - Case conversion to lower() 
#    - removing punctuation
#    - removing white spaces (strip())
# 3. Stop words removal
# 4. Stemming/ Lemmatisation


# In[47]:


def lower_case(text):
    return text.lower()

def punctuation_removal(text):
    punctuations=['!','@','#','$','%','^','&','*','(',')','-','_','`','~','+','=','[',']','{','}','|',';',':','<','>','?','/',',','.','"','<<','>>']
    for character in text:
        if(character in punctuations) or (character in ['—','\n',"\\"]) :  #including em-dash, forward slash and enter seperately
            text = text.replace(character," ")
    return text 

def remove_apostrophe(text):
    text = str(np.char.replace(text, "'", " "))
    return text

def num_to_words(text):
    if text.isdigit()==True:
        text = num2words(text)
    else:
        text = text
    return text

def remove_URLs(text):
    text = ' '.join(word for word in text.split() if word[:4] not in('www:','http'))
    return text

def remove_short_words(text):
    text = ' '.join(word for word in text.split() if len(word)>2)
    #text = re.sub(r'bw{1,2}b', '', text) 
    return text

def remove_long_words(text):
    text = ' '.join(word for word in text.split() if len(word)<15)
    return text

def remove_white_space(text):
    text = text.strip()
    return text

def stop_words_removal(vocabulary): 
    FilteredVocabulary = []
    for term in vocabulary:
        if term not in stops:
            FilteredVocabulary.append(term)
    return FilteredVocabulary

def tokenisation(text):
    tokens = [term for term in text.split()]         #tokenisation 
    return tokens

def lemmetizing(text):
    lemmatiser = WordNetLemmatizer()
    lemmetized_word = lemmatiser.lemmatize(text)
    #lemmetized_word = Lemmatizer.lemmatize
    return lemmetized_word

def stemming(term):
    suffixes = ['ed', 'ing' , 's','es']#,'ers', 'ion', 'ize', 'ise', 'ive', 'en', 'ly', 'ish', 'ian','ese']
    for suffix in suffixes:
        if term.endswith(suffix):
            term =  term[:-len(suffix)]
        else:
            term = term
    return term


def preprocessing(text):
    text = lower_case(text)
    text = punctuation_removal(text)
    text = remove_apostrophe(text)
    text = remove_URLs(text)
    text = remove_short_words(text) 
    text = remove_long_words(text)
    text = remove_white_space(text)
    return text



passage_collection = "passage-collection.txt"
with open(passage_collection) as file:
    lines = file.readlines()

start = time.time()

clean_lines = [preprocessing(line) for line in lines]
text_tokens = [tokenisation(line) for line in clean_lines]
vocabulary = [word_token for line_token in text_tokens for word_token in line_token]
vocabulary = [lemmetizing(word) for word in vocabulary]
vocabulary_wo_stopwords = stop_words_removal(vocabulary)
vocab = list(dict.fromkeys(vocabulary_wo_stopwords))
print("Length of vocab: ", len(vocab))
print("Time taken to create vocab: ", time.time() - start)


candidate_passage = "candidate-passages-top1000.tsv"
passage_data = pd.read_csv(candidate_passage, sep='\t',names=['qid','pid','query','passage'])
data = passage_data.drop_duplicates()

st1 = time.time()
data['passage'] = data['passage'].apply(preprocessing)
data['passage'] = data['passage'].apply(lambda x: x.split())
data['passage'] = data['passage'].apply(stop_words_removal)
print("Time taken to preprocess passages: ", time.time() - st1)

data_passage = data.drop(['qid','query'],axis=1)
data_passage = data_passage.drop_duplicates(subset=['pid'])
data_passage = data_passage.reset_index()

inverted_index_dict = {}
term_count = 0
count = 0

st2 = time.time()
word_count = 0
for idx, row in data_passage.iterrows():
    freq_tokens = nltk.FreqDist(row['passage'])
    words_passage = len(row['passage'])
    for word, freq in freq_tokens.items():
        if word not in inverted_index_dict:
            inverted_index_dict[word] = [(int(row['pid']), freq, words_passage)] 
        else:
            inverted_index_dict[word].append((int(row['pid']), freq, words_passage))

print('Time taken to created inverted dict from passages: ', time.time() - st2)




new_words = set(vocab) - set(inverted_index_dict.keys())
print("Count of words present in passage-collection but not in candidate-passages: ", len(new_words))


words_not_in_vocab = set(inverted_index_dict.keys()) -  set(vocab) 
print("Count of words present in candidate-passages but not in passage-collection: ", len(words_not_in_vocab))


# Modifying the inverted dictionary accordingly
for n in new_words:
    inverted_index_dict[n] = []

for w in words_not_in_vocab:
    del inverted_index_dict[w]

print("Total keys in final inverted index dictionary: ", len(inverted_index_dict.keys()))
print("Total time taken in Task 2: ", time.time() - start)

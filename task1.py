import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time

import nltk

#Stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))

#Lemmitizer
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from collections import defaultdict


from num2words import num2words

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

def stop_words_removal(vocabulary): 
    FilteredVocabulary = []
    for term in vocabulary:
        if term not in stops:
            FilteredVocabulary.append(term)
    return FilteredVocabulary

def preprocessing(text):
    text = lower_case(text)
    text = punctuation_removal(text)
    text = remove_apostrophe(text)
    text = remove_URLs(text)
    text = remove_short_words(text) 
    text = remove_long_words(text)
    text = remove_white_space(text)
    #print(type(text))
    #text = NumtoWords(text)
    return text

def word_counter(vocabulary):
    TermCount = {}  
    for term in vocabulary:
        if (term in TermCount):  #Count term frequency for each term in one_grams list 
            TermCount[term] = TermCount[term]+1
        else:
            TermCount[term]=1
    return TermCount

# k - term frequency
# s - set of documents
# N - Length of vocabulary
def ZipfDistribution(k,s,N): # Function takes tf-rank, calculates denominator sum and return the ratio for every term
    zipf=[]
    den = 0
    for i in range(1,N+1):
        den = den + (i**(-s))  
    for k in range(1,N+1):
        zipf.append(((k)**(-s))/den)
    return zipf


filename = "passage-collection.txt"
with open(filename) as file:
    lines = file.readlines()

start = time.time()
clean_lines = [preprocessing(line) for line in lines]

text_tokens = [tokenisation(line) for line in clean_lines]

vocabulary = [word_token for line_token in text_tokens for word_token in line_token] 

vocabulary = [lemmetizing(word) for word in vocabulary]
print("Time taken to extract vocab: ", time.time() - start)
print("Length of vocab: ", len(vocabulary))

vocabulary_wo_stopwords = stop_words_removal(vocabulary)
print("Length of vocab w/o stopwords: ", len(vocabulary_wo_stopwords))

vocab = list(dict.fromkeys(vocabulary_wo_stopwords))

frequencies = word_counter(vocabulary)
frequencies = sorted(frequencies.items(), key=lambda i: i[1], reverse=True)
frequencies_df = pd.DataFrame.from_dict(frequencies,orient='columns')
frequencies_df.columns=['Term','Term Frequency']
frequencies_df.index = np.arange(1, len(frequencies_df) + 1)
frequencies_df['Rank'] = frequencies_df.index
frequencies_df.set_index('Term', inplace=True)
frequencies_df['Normalised Frequency'] = frequencies_df['Term Frequency']/sum(frequencies_df['Term Frequency'])
frequencies_df['Rank*Frequency'] = (frequencies_df['Rank']) * (frequencies_df['Term Frequency'])

frequencies_df['Zipf'] = ZipfDistribution(frequencies_df['Term Frequency'],1,len(frequencies_df))

rank = frequencies_df['Rank']
norm_freq = frequencies_df['Normalised Frequency']
zipf = frequencies_df['Zipf']

plt.plot(rank,norm_freq)
plt.plot(rank, zipf, linestyle='dashed')
plt.title("Emprical Distribution vs Zipf's Distribution (with stop words)")
plt.legend(['Data','Zipf'])
plt.xlabel("Term frequency ranking")
plt.ylabel("Term probability of occurrence")
# plt.savefig('zipf.png')


log_rank = frequencies_df['Rank']
log_norm_freq = frequencies_df['Normalised Frequency']
log_zipf = frequencies_df['Zipf']

plt.loglog(log_rank,log_norm_freq)
plt.loglog(log_rank, log_zipf, linestyle='dashed')
plt.title("Emprical Distribution vs Zipf's Distribution (with stop words)")
plt.legend(['Data','Zipf'])
plt.xlabel("Term frequency ranking (log)")
plt.ylabel("Term probability of occurrence (log)")
# plt.savefig('zipflog.png')



frequencies = word_counter(vocabulary_wo_stopwords)
frequencies = sorted(frequencies.items(), key=lambda i: i[1], reverse=True)
frequencies_df = pd.DataFrame.from_dict(frequencies,orient='columns')
frequencies_df.columns=['Term','Term Frequency']
frequencies_df.index = np.arange(1, len(frequencies_df) + 1)
frequencies_df['Rank'] = frequencies_df.index
frequencies_df.set_index('Term', inplace=True)
frequencies_df['Normalised Frequency'] = frequencies_df['Term Frequency']/sum(frequencies_df['Term Frequency'])
frequencies_df['Rank*Frequency'] = (frequencies_df['Rank']) * (frequencies_df['Term Frequency'])


frequencies_df['Zipf'] = ZipfDistribution(frequencies_df['Term Frequency'],1,len(frequencies_df))


log_rank = frequencies_df['Rank']
log_norm_freq = frequencies_df['Normalised Frequency']
log_zipf = frequencies_df['Zipf']

plt.loglog(log_rank,log_norm_freq)
plt.loglog(log_rank, log_zipf, linestyle='dashed')
plt.title("Emprical Distribution vs Zipf's Distribution (without stop words)")
plt.legend(['Data','Zipf'])
plt.xlabel("Term frequency ranking (log)")
plt.ylabel("Term probability of occurrence (log)")
# plt.savefig('zipflogwostop.png')

print("Total time taken in Task 1: ", time.time() - start)
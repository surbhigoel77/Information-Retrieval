import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import time
import nltk
import csv
from num2words import num2words

nltk.download('stopwords')
# from tqdm import tqdm
stops = set(stopwords.words('english'))

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

def preprocessing(text):
    text = lower_case(text)
    text = punctuation_removal(text)
    text = remove_apostrophe(text)
    text = remove_URLs(text)
    text = remove_short_words(text) 
    text = remove_long_words(text)
    text = remove_white_space(text)
    return text

start = time.time()

passage_file = "candidate-passages-top1000.tsv"
raw_data = pd.read_csv(passage_file, sep='\t',names=['qid','pid','query','passage'])
data = raw_data.drop_duplicates()
print(data.head())

st1 = time.time()
data['query'] = data['query'].apply(preprocessing)
data['query'] = data['query'].apply(lambda x: x.split())
data['query'] = data['query'].apply(stop_words_removal)
print("Time taken to preprocess query: ", time.time() - st1)

st2 = time.time()
data['passage'] = data['passage'].apply(preprocessing)
data['passage'] = data['passage'].apply(lambda x: x.split())
data['passage'] = data['passage'].apply(stop_words_removal)
print("Time taken to preprocess passages: ", time.time() - st2)

print("Preprocessed Data: ")
print(data.head())

passage_df = data.drop_duplicates(subset=['pid'], keep='first')[['pid', 'passage']] 

inverted_index_dict = {}

st3 = time.time()
for idx, row in passage_df.iterrows():
    freq_tokens = nltk.FreqDist(row['passage'])
    words_passage = len(row['passage'])
    for word, freq in freq_tokens.items():
        if word not in inverted_index_dict:
            inverted_index_dict[word] = [(int(row['pid']), freq, words_passage)] 
        else:
            inverted_index_dict[word].append((int(row['pid']), freq, words_passage))
print('Time taken to calculate inverted index dict: ', time.time() - st3)

vocab = list(inverted_index_dict.keys())
V = len(vocab)

def laplace_score(query, passage):
    D = len(passage)
    passage_fdist = nltk.FreqDist(passage)
    score = 0
    for token in query:
        score += np.log((passage_fdist[token]+1)/(D+V))     
    return score

st4 = time.time()
data['laplace_score'] = data.apply(lambda x: laplace_score(x['query'], x['passage']), axis=1)
print('Time taken to calculate laplace score: ', time.time() - st4)

def lidstone_score(query, passage):
    D = len(passage)
    passage_fdist = nltk.FreqDist(passage)
    score = 0
    epsilon = 0.1
    for token in query:
        score += np.log((passage_fdist[token]+epsilon)/(D+epsilon*V))
    return score

st5 = time.time()
data['lidstone_score'] = data.apply(lambda x: lidstone_score(x['query'], x['passage']), axis=1)
print('Time taken to calculate lidstone score: ', time.time() - st5)

def total_frequency(word):
    freq = 0
    try:
        for tup in inverted_index_dict[word]:
            freq += tup[1]
    except:
        pass
    return freq

def dirichlet_score(pid, query, passage):
    passage_freq_distribution = nltk.FreqDist(passage)
    N = len(passage)
    mu = 50
    score = 0
    lamb = N / (N + mu)
    one_lamb = mu / (N + mu)
    for token in query:
        freq = total_frequency(token)
        ft = lamb * (passage_freq_distribution[token] / N)
        st = one_lamb * (freq / V)
        if (ft + st == 0):
            continue
        score += np.log(ft + st)
    return score

st6 = time.time()
data['dirichlet_score'] = data.apply(lambda x: dirichlet_score(x['pid'], x['query'], x['passage']), axis=1)
print('Time taken to calculate dirichlet score: ', time.time() - st6)

def write_results(queries_file, model, df):
    queries_data = pd.read_csv(queries_file, sep='\t',names=['qid','query']) # We need to put the result in this sequence
    querylist = list(queries_data.qid)
    filename = '%s.csv' % model
    col = '%s_score' % model
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for q in querylist:
            sorted_df = df[(df.qid==q)].sort_values(by=[col],ascending=False)
            if len(sorted_df)>=100:
                limit=100
            else:
                limit=len(sorted_df)
            for i, row in sorted_df.iloc[:limit].iterrows():
                writer.writerow((int(row.qid),int(row.pid),row[col]))

queries_file = "test-queries.tsv"
write_results(queries_file, 'laplace', data)
write_results(queries_file, 'lidstone', data)
write_results(queries_file, 'dirichlet', data)

print("Total time taken in Task 4: ", time.time() - start)

import numpy as np
import pandas as pd
from num2words import num2words
import csv
from datetime import datetime
import time
timenow =  datetime.now()

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
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
    text = str(np.char.replace(text, "'", ""))
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

###########################################################################################
print("Calculating tf-idf of passages")
###########################################################################################


st3 = time.time()
data_passage = data.drop(['qid','query'],axis=1)
data_passage = data_passage.drop_duplicates(subset=['pid'])
data_passage = data_passage.reset_index()


inverted_index_dict = {}

st4 = time.time()
for idx, row in data_passage.iterrows():
    freq_tokens = nltk.FreqDist(row['passage'])
    words_passage = len(row['passage'])
    for word, freq in freq_tokens.items():
        if word not in inverted_index_dict:
            inverted_index_dict[word] = [(int(row['pid']), freq, words_passage)] 
        else:
            inverted_index_dict[word].append((int(row['pid']), freq, words_passage))
print('Time taken to calculate inverted index dict: ', time.time() - st4)

tf=[]
st5 = time.time()
for idx, row in data_passage.iterrows():
    tokens = set(row.passage)
    for term in tokens:
        count = row.passage.count(term) 
#         term_freq = 1 + np.log(count/len(row.passage))
        term_freq = count/len(row.passage)
        tf.append((row.pid,term,term_freq))   #qid,pid,term,count,length of doc

print("Time taken to calculate term frequency for passages: ", time.time() - st5)

passage_tf = pd.DataFrame(tf)
passage_tf.columns=['pid','term','tf']
print("Passage tf:")
print(passage_tf.head())

idf=[]
N = len(data_passage)
st5 = time.time()
for key,value in inverted_index_dict.items(): 
    docfreq = len(inverted_index_dict[key])
    idf.append((key, np.log10(N/docfreq)))
print("Time taken to calc term idf for passages: ", time.time() - st5)

passage_idf = pd.DataFrame(idf)
passage_idf.columns = ['term','idf']
print("Passage idf:")
print(passage_idf.head())

passage_tfidf = passage_tf.merge(passage_idf, on='term', how='inner')
passage_tfidf['tfidf'] = passage_tfidf['tf']*passage_tfidf['idf']
print("Passage tfidf:")
print(passage_tfidf.head())

passage_tfidf['magnitude'] = passage_tfidf.tfidf.apply(lambda x: x**2)
passage_norm = passage_tfidf.groupby(['pid'])['magnitude'].agg('sum')
passage_norm = passage_norm.apply(np.sqrt)

passage_norm = passage_norm.to_frame().reset_index()
passage_norm = passage_norm.rename(columns={"magnitude": "norm"})


passage_tfidf_norm = pd.merge(passage_tfidf, passage_norm, on='pid')
passage_tfidf_norm = passage_tfidf_norm.drop(['tf', 'idf', 'magnitude'], axis=1)
print("Passage tfidf with norm:")
print(passage_tfidf_norm.head())
print("Time taken to calculate tfidf for passages: ", time.time() - st3)

###########################################################################################
print("Calculating tf-idf of queries")
###########################################################################################

st6 = time.time()


data_query = data

st7 = time.time()
tf = []
for idx, row in data_query.iterrows():
    query_len = len(row.query)
    for term in row.query:
        term_count = row.query.count(term)/query_len 
        tf.append((row.qid,row.pid,term,term_count))    #qid,pid,term,count,length of doc    

query_tf = pd.DataFrame(tf)
query_tf.columns =['qid','pid','term','termcount']
print('Time taken to calculate tf of queries: ', time.time() - st7)
print("Queries tf:")
print(query_tf.head())

st8 = time.time()
idf=[]
for idx, row in query_tf.iterrows():
    if row.term in inverted_index_dict:
        idf.append(np.log10(N/(len(inverted_index_dict[row.term]))))
    else:
        idf.append(0)
query_idf = pd.DataFrame(idf)
query_idf.columns = ['idf']
print('Time taken to calculate idf of queries: ', time.time() - st8)

query_tfidf = query_tf
query_tfidf['idf'] = query_idf
query_tfidf['tfidf'] = query_tfidf['termcount']*query_tfidf['idf']
print("Queries tfidf: ")
print(query_tfidf.head())



query_tfidf['magnitude'] = query_tfidf.tfidf.apply(lambda x: x**2)
query_norm = query_tfidf.groupby(['qid', 'pid'])['magnitude'].agg('sum')
query_norm = query_norm.apply(np.sqrt)
query_norm = query_norm.to_frame().reset_index()
query_norm = query_norm.rename(columns={"magnitude": "norm"})
query_norm = query_norm.drop(['pid'], axis=1).drop_duplicates()

query_tfidf_norm = pd.merge(query_tfidf, query_norm, on='qid')
query_tfidf_norm = query_tfidf_norm.drop(['termcount', 'idf', 'magnitude'], axis=1)
print("Queries tfidf with norm:")
print(query_tfidf_norm.head())
print("Time taken to calculate tfidf for queries: ", time.time() - st6)


###########################################################################################
print("Calculating cosign similarity")
###########################################################################################
cos_df = query_tfidf_norm.merge(passage_tfidf_norm, on=['pid','term'], how='left').fillna(0)
print("Cosign similarity: ")
print(cos_df.head())

cos_df['score'] = (cos_df.tfidf_x * cos_df.tfidf_y)/ (cos_df.norm_x * cos_df.norm_y)
cos_df = cos_df.fillna(0)
print("Cosign similarity with norm: ")
print(cos_df.head())


cos_df_group = cos_df.groupby(['qid','pid']).agg({'score':sum}).reset_index()
print("Cosign score aggregated: ")
print(cos_df_group.head())

queries_file = "test-queries.tsv"
test_queries = pd.read_csv(queries_file, sep='\t',names=['qid','query'])

querylist = list(test_queries.qid)

filename = "tfidf.csv"
with open(filename, 'w') as csvfile:
    writer = csv.writer(csvfile)
    for q in querylist:
        df = (cos_df_group[(cos_df_group.qid==q)].sort_values(by=['score'],ascending=False)).values       
        if len(df)>=100:
            limit=100
        else:
            limit=len(df)            
        for i in range(0,limit):
            writer.writerow((int(df[i][0]),int(df[i][1]),df[i][2]))

print("Time taken to calculate top 100 passages for queries using cosign similarity: ", time.time() - start)

###########################################################################################
print("Calculating bm25 model scores")
###########################################################################################
st9 = time.time()

def write_results(model, df):
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

N = len(data_passage)
# Average Document Length
words = 0
for i in range(len(data_passage)):
    words += len(data_passage.passage[i])

avg_doc_len = words/N

R = 0
r = 0
k1 = 1.2
k2 = 100
b = 0.75


def bm25_score(query, passage):    
    passage_freq_distribution = nltk.FreqDist(passage)
    query_freq_distribution = nltk.FreqDist(query)
    len_doc = len(passage)
    K = k1*((1-b) + b *(float(len_doc)/float(avg_doc_len)))
    
    score = 0
    for token in query:
        try:
            n = len(inverted_index_dict[token])
        except:
            n = 0
        f = passage_freq_distribution[token]
        qf = query_freq_distribution[token]
        inter = np.log(((r + 0.5)/(R - r + 0.5))/((n-r+0.5)/(N-n-R+r+0.5)))
        score += inter * ((k1 + 1) * f)/(K+f) * ((k2+1) * qf)/(k2+qf)
    return score

bm25 = data
bm25['bm25_score'] = data.apply(lambda x: bm25_score(x['query'], x['passage']), axis=1)
print('Time taken to calculate BM25 scores: ', time.time() - st9)
write_results('bm25', bm25)


print("Total time taken in Task 3: ", time.time() - start)

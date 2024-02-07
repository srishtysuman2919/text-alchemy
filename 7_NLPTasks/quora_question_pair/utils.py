import pandas as pd
import re
from nltk.corpus import stopwords
import gensim
import numpy as np
from torch.utils.data import Dataset
import torch

def collate(batch):
    q1_text_list = []
    q2_text_list = []
    q1_list = []
    q2_list = []
    labels = []
    for item in batch:
        q1_text_list.append(item['q1'])
        q2_text_list.append(item['q2'])
        q1_list.append(item['q1_token'])
        q2_list.append(item['q2_token'])
        labels.append(item['labels'])
          
        
    q1_lengths = [len(q) for q in q1_list]
    q2_lengths = [len(q) for q in q2_list]
    
    return {
        'q1_text': q1_text_list,
        'q2_text': q2_text_list, 
        'q1_token': q1_list, 
        'q2_token': q2_list,
        'q1_lengths': q1_lengths, 
        'q2_lengths': q2_lengths,
        'labels': labels
    }

def convert_data_to_tuples(df):
    question_list=df.apply(lambda x: (x["question1"], x["question2"]), axis=1).tolist()
    question_indices=df.apply(lambda x: (x["token_question1"], x["token_question2"]), axis=1)
    labels=df["is_duplicate"].tolist()
    return question_list, question_indices, labels

class QuoraDataset(Dataset):
    def __init__(self, questions_list, question_indices, labels):
        """
        Params:
        -------
        questions_list : list
                         list with tuples of all the questions pairs 
        
        word2index : dict
                     vocbulary of the dataset
        labels : list 
                 list of the corrsponding labels to the question pairs 
        
        """
        self.questions_list = questions_list
        self.questions_indices = question_indices
        self.labels = labels
        
    def __len__(self):
        return len(self.questions_list)
    
    def __getitem__(self, index):
        questions_pair_indices = self.questions_indices[index]
        q1_indices = questions_pair_indices[0]            
        q2_indices = questions_pair_indices[1]
        questions_pair = self.questions_list[index]
        q1=" ".join(word for word in questions_pair[0])
        q2=" ".join(word for word in questions_pair[0])
            
        # q1_indices and q2_indices are lists of indices against words used in the sentence 
        return {
            'q1': q1,
            'q2': q2,
            'q1_token': q1_indices, 
            'q2_token': q2_indices, 
            'labels': self.labels[index], 
        }


def pad_sequence(df, max_len=None): 
    max_len=max_len
    def pad_seq(sequence, max_len):
        if len(sequence)>max_len:
            return sequence[:max_len]
        len_x=len(sequence)
        return [0]*(max_len-len_x)+sequence
    df["padded_q1"]=df["question1"].apply(lambda x: pad_seq(x, max_len))
    df["padded_q2"]=df["question2"].apply(lambda x: pad_seq(x, max_len))

    q1_data=df["padded_q1"].values.tolist()
    q2_data=df["padded_q2"].values.tolist()
    labels=np.array(df['is_duplicate'], dtype=int)
    return q1_data, q2_data, labels
    

def pad_sequences(df, max_len=None):
    max_len=max_len
    def pad_sequences(sequences, max_len):
        num_samples = len(sequences)
        lengths = []
        sample_shape = ()
        flag = True
        for x in sequences:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False

        if max_len is None:
            max_len = np.max(lengths)

        x = np.full((num_samples, max_len) + sample_shape, 0.0, dtype="int32")
        for idx, s in enumerate(sequences):
            trunc = s[-max_len:]
            trunc = np.asarray(trunc, dtype="int32")
            x[idx, -len(trunc) :] = trunc
        return x
    
    q1_data = pad_sequences(df['question1'], max_len)
    q2_data = pad_sequences(df['question2'], max_len)
    labels = np.array(df['is_duplicate'], dtype=int)
    return q1_data, q2_data, labels

def load_pretrained_embeddings():
    embeddings=gensim.models.KeyedVectors.load_word2vec_format('/Users/srishtysuman/Downloads/GoogleNewsvectorsnegative300.bin', binary=True)
    return embeddings

def read_data(path, size=2000):
    df=pd.read_csv(path)
    df=df.iloc[:size]
    return df

def preprocess(df, columns):
    df[columns[0]]=df[columns[0]].apply(lambda x: clean(x))
    df[columns[1]]=df[columns[1]].apply(lambda x: clean(x))
    return df

def map_words(df, columns, pre_trained_embedding_vocab):
    vocabulary=dict()
    appended_unique_words=['<unk>']   
    stop_words = stopwords.words('english')
    count=0
    token_columns={"token_question1":[], "token_question2":[]}
    for index,row in df.iterrows():
        for question in columns:
            q2n = []  
            for word in row[question]:
                if word in stop_words or word not in pre_trained_embedding_vocab:
                    continue
                if word not in vocabulary:
                    count+=1
                    vocabulary[word] = count
                    q2n.append(count)
                    appended_unique_words.append(word)
                else:
                    q2n.append(vocabulary[word])
            token_columns["token_"+question].append(q2n)
    for col_name, value in token_columns.items():
        df[col_name]=value
    return df, vocabulary, appended_unique_words

# def map_words(df, columns, pre_trained_embedding_vocab):
#     vocabulary=dict()
#     appended_unique_words=['<unk>']   
#     stop_words = stopwords.words('english')
#     count=0
#     for index,row in df.iterrows():
#         for question in columns:
#             q2n = []  
#             for word in row[question]:
#                 if word in stop_words and word not in pre_trained_embedding_vocab:
#                     continue
#                 if word not in vocabulary:
#                     count+=1
#                     vocabulary[word] = count
#                     q2n.append(count)
#                     appended_unique_words.append(word)
#                 else:
#                     q2n.append(vocabulary[word])
#             df.at[index, question] = q2n 
#     return df, vocabulary, appended_unique_words


def get_embeddings_matrix(vocabulary, embedding_dim, word2vec):
    embeddings=torch.randn(len(vocabulary) + 1, embedding_dim)
    embeddings[0] = torch.zeros(embedding_dim)
    for word, index in vocabulary.items():
        if word in word2vec.index_to_key:
            embeddings[index]=torch.FloatTensor(word2vec.word_vec(word))
    return embeddings

def clean(text):
    if type(text)!=str or text=='' or pd.isnull(text):
        return ''
    text = text.lower()
    text = re.sub("\'s", " ", text) 
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("n't", " not ", text)
    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " are ", text)
    text = re.sub("\'d", " would ", text)
    text = re.sub("\'ll", " will ", text)
    text = re.sub("e\.g\.", " eg ", text, flags=re.IGNORECASE)
    text = re.sub("b\.g\.", " bg ", text, flags=re.IGNORECASE)
    text = re.sub("(\d+)(kK)", " \g<1>000 ", text)
    text = re.sub("e-mail", " email ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?U\.S\.A\.", " America ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?United State(s)?", " America ", text, flags=re.IGNORECASE)
    text = re.sub("\(s\)", " ", text, flags=re.IGNORECASE)
    text = re.sub("[c-fC-F]\:\/", " disk ", text)
    text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)
    text = re.sub('\$', " dollar ", text)
    text = re.sub('\%', " percent ", text)
    text = re.sub('\&', " and ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r'\d+', '',text)
    text = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', text)
    text = text.replace("?","")
    text = text.replace("(","")
    text = text.replace(")","")
    text = text.replace('"',"")
    text = text.replace(",","")
    text = text.replace("#","")   
    text = text.replace("-","")    
    text = text.replace("..","")
    text = text.replace("/","")
    text = text.replace("\\","")
    text = text.replace(":","")
    text = text.replace("the","") 
    text = text.split()   
    return text




#jieguo.py
import os

import numpy as np
import tensorflow as tf
#import tensorflow_hub as hub
from collections import Counter
import Summarizer
import summarizer_data_utils
import summarizer_model_utils
import pandas as pd
import jieba
#import tensorflow.keras as keras
max_features=10000
maxlen=300
dt=pd.read_csv("sgnewfull.csv",sep='\t',skiprows=1,names=['title','content'],nrows=10000)
#按行读取文件，返回文件的行字符串列表,读取stopwords.dat
def read_file(file_name):
    fp = open(file_name, "r", encoding="utf-8")
    content_lines = fp.readlines()
    fp.close()
    #去除行末的换行符，否则会在停用词匹配的过程中产生干扰
    for i in range(len(content_lines)):
        content_lines[i] = content_lines[i].rstrip("\n")
    return content_lines
def fenci(selist):
    stopwords = read_file("stopwords.dat")#读取停用词
    l=[]
    for i in selist:
        k=[]
        seg_list = jieba.cut(i)  # 默认是精确模式
        outstr=''
        i=0
        for word in seg_list:  #去除停顿词
            i+=1
            if word not in stopwords:  #如果去除停用词的话，把注释去掉，同时把下面三行加tab
                if word != '\t' and i<150 :  
                    k.append(word)
        l.append(k)
    return l

#tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_features,lower=True)
dt['title']=dt['title'].fillna("")
dt['content']=dt['content'].fillna("")
title=dt['title']
#title=[('s'+i+'e')for i in title]

content=fenci(list(dt['content']))
title=fenci(title)
#统计词
words=[]
for text in content:
    for word in text:
        words.append(word)
for text in title:
    for word in text:
        words.append(word)
words_counted = Counter(words).most_common() 


specials = ["<EOS>", "<SOS>","<PAD>","<UNK>"]
word2ind, ind2word,  missing_words = summarizer_data_utils.create_word_inds_dicts(words_counted,
                                                                       specials = specials)


processed_texts=content
converted_texts, unknown_words_in_texts = summarizer_data_utils.convert_to_inds(processed_texts,
                                                                                word2ind,
                                                                                eos = False)


processed_summaries=title
converted_summaries, unknown_words_in_summaries = summarizer_data_utils.convert_to_inds(processed_summaries,word2ind, eos = True,  sos = True)



num_layers_encoder = 2
num_layers_decoder = 2
rnn_size_encoder = 128
rnn_size_decoder = 128

batch_size = 64
epochs = 10
clip = 5
keep_probability = 0.5
learning_rate = 0.001
max_lr=0.005
learning_rate_decay_steps = 700
learning_rate_decay = 0.90


pretrained_embeddings_path = './tf_hub_embedding.npy'
summary_dir = os.path.join('./tensorboard', str('Nn_' + str(rnn_size_encoder) + '_Lr_' + str(learning_rate)))


use_cyclic_lr = True
inference_targets=True


d=round(len(converted_summaries)*0.9)

summarizer_model_utils.reset_graph()
summarizer = Summarizer.Summarizer(word2ind,
                                   ind2word,
                                   save_path='./models/sogou/my_model',
                                   mode='TRAIN',
                                   num_layers_encoder = num_layers_encoder,
                                   num_layers_decoder = num_layers_decoder,
                                   rnn_size_encoder = rnn_size_encoder,
                                   rnn_size_decoder = rnn_size_decoder,
                                   batch_size = 32,
                                   clip = clip,
                                   keep_probability = keep_probability,
                                   learning_rate = learning_rate,
                                   max_lr=max_lr,
                                   learning_rate_decay_steps = learning_rate_decay_steps,
                                   learning_rate_decay = learning_rate_decay,
                                   epochs = epochs,
                                   pretrained_embeddings_path = None, #pretrained_embeddings_path,
                                   use_cyclic_lr = use_cyclic_lr,
                                   summary_dir = None)#summary_dir)           

summarizer.build_graph()
summarizer.train(converted_texts[:d], 
                 converted_summaries[:d],
                 validation_inputs=converted_texts[d:],
                 validation_targets=converted_summaries[d:])


summarizer_model_utils.reset_graph()
summarizer = Summarizer.Summarizer(word2ind,
                                   ind2word,
                                   './models/sogou/my_model',
                                   'INFER',
                                   num_layers_encoder = num_layers_encoder,
                                   num_layers_decoder = num_layers_decoder,
                                   batch_size = len(converted_texts[:50]),
                                   clip = clip,
                                   keep_probability = 1.0,
                                   learning_rate = 0.0,
                                   beam_width = 5,
                                   rnn_size_encoder = rnn_size_encoder,
                                   rnn_size_decoder = rnn_size_decoder,
                                   inference_targets = True,
                                   pretrained_embeddings_path = None)#pretrained_embeddings_path)

summarizer.build_graph()
preds = summarizer.infer(converted_texts[:50],
                         restore_path =  './models/sogou/my_model',
                         targets = converted_summaries[:50])



summarizer_model_utils.sample_results(preds,
                                      ind2word,
                                      word2ind,
                                      converted_summaries[:50],
                                      converted_texts[:50])



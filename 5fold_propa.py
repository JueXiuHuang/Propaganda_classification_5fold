#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from tqdm import tqdm
from transformers import *
from official import nlp
import official.nlp.optimization
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings("ignore")


# In[2]:


def load_propa_data():
    dfUDN = pd.read_csv('originalDataset/propa/UDN-bootstrap-checked.tsv', sep='\t', names=['宣傳手法', '標記詞', '句子'])
    dfLT = pd.read_csv('originalDataset/propa/LT-bootstrap-checked.tsv', sep='\t', names=['宣傳手法', '標記詞', '句子'])
    dfUDN_new = pd.read_csv('originalDataset/propa/UDN-bootstrap-checked-20210209.tsv', sep='\t', names=['宣傳手法', '標記詞', '句子'])
    dfLT_new = pd.read_csv('originalDataset/propa/LT-bootstrap-checked-20210209.tsv', sep='\t', names=['宣傳手法', '標記詞', '句子'])
    dfPropa_new = pd.concat([dfUDN, dfLT, dfUDN_new, dfLT_new], ignore_index=True)
    dfPropa_old = pd.concat([dfUDN, dfLT], ignore_index=True)
    return dfPropa_new, dfPropa_old

def encode_data(contexts, tokenizer, sent_len):
    input_ids, attention_mask = [], []
    for i in range(len(contexts)):
        inputs = tokenizer.encode_plus(contexts[i],add_special_tokens=True, max_length=sent_len, pad_to_max_length=True,
                    return_attention_mask=True, return_token_type_ids=False, truncation=True)
        input_ids.append(inputs['input_ids'])
        attention_mask.append(inputs['attention_mask'])
    
    return np.asarray(input_ids, dtype='int32'), np.asarray(attention_mask, dtype='int32')

def remove_not_Ch_Eng(cont):
    # chinese unicode range: [0x4E00,0x9FA5]
    rule = u'[\u4E00-\u9FA5\w]'
    for i in range(len(cont)):
        pChEng = re.compile(rule).findall(cont[i])
        ChEngText = "".join(pChEng)
        cont[i] = ChEngText
    return cont

def plot_learning_curve(hist):
    pd.DataFrame(hist.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
    return


# In[3]:


# Rapid and slow 在表現上沒有太大的差別，直接用rapid就好
def build_model_propa_slow(base, lr, epochs, batchSize, train_data_size, sent_len):
    model = TFBertModel.from_pretrained(base, return_dict=True)
    inp1 = tf.keras.Input(shape=(sent_len,), dtype=tf.int32, name='input_ids')
    inp2 = tf.keras.Input(shape=(sent_len,), dtype=tf.int32, name='attention_mask')
    outDict = model([inp1, inp2])
    #print(outDict.keys())
    # >>> odict_keys(['last_hidden_state', 'pooler_output'])
    pooler_out = outDict['pooler_output']
    hid = tf.keras.layers.Dense(512, activation='relu')(pooler_out)
    hid = tf.keras.layers.Dense(128, activation='relu')(hid)
    hid = tf.keras.layers.Dense(32, activation='relu')(hid)
    hid = tf.keras.layers.Dense(8, activation='relu')(hid)
    result = tf.keras.layers.Dense(1, activation='sigmoid')(hid)
    #spe = steps_per_epoch
    spe = int(train_data_size/batchSize)
    #nts = num_train_steps
    nts = spe * epochs
    #ws = warmup_steps
    ws = int(epochs * train_data_size * 0.1 / batchSize)
    opWarm = nlp.optimization.create_optimizer(lr, num_train_steps=nts,
                                                  num_warmup_steps=ws)
    model = tf.keras.Model(inputs=[inp1, inp2], outputs=result)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    metric = tf.keras.metrics.BinaryAccuracy('accuracy')
    model.compile(optimizer=opWarm, loss=loss_fn, metrics=[metric])
    
    model.summary()
    
    return model

def build_model_propa_rapid(base, lr, epochs, batchSize, train_data_size, sent_len):
    model = TFBertModel.from_pretrained(base, return_dict=True)
    inp1 = tf.keras.Input(shape=(sent_len,), dtype=tf.int32, name='input_ids')
    inp2 = tf.keras.Input(shape=(sent_len,), dtype=tf.int32, name='attention_mask')
    outDict = model([inp1, inp2])
    #print(outDict.keys())
    # >>> odict_keys(['last_hidden_state', 'pooler_output'])
    pooler_out = outDict['pooler_output']
    hid = tf.keras.layers.Dense(64, activation='relu')(pooler_out)
    result = tf.keras.layers.Dense(1, activation='sigmoid')(hid)
    #spe = steps_per_epoch
    spe = int(train_data_size/batchSize)
    #nts = num_train_steps
    nts = spe * epochs
    #ws = warmup_steps
    ws = int(epochs * train_data_size * 0.1 / batchSize)
    opWarm = nlp.optimization.create_optimizer(lr, num_train_steps=nts,
                                                  num_warmup_steps=ws)
    model = tf.keras.Model(inputs=[inp1, inp2], outputs=result)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    metric = tf.keras.metrics.BinaryAccuracy('accuracy')
    model.compile(optimizer=opWarm, loss=loss_fn, metrics=[metric])
    
    model.summary()
    
    return model


# In[ ]:





# ### Main function

# In[4]:


base = 'bert-base-chinese'
tok = BertTokenizer.from_pretrained(base)
sent_len = 16


# In[5]:


# generate propa training data - new
dfPropa, dfPropa_old = load_propa_data()
dfPropa['宣傳手法'] = dfPropa['宣傳手法'].apply(lambda x: 0 if x == 'X' else 1)
#neg, pos = np.bincount(dfPropa['宣傳手法'])
propa_lab = dfPropa['宣傳手法'].to_numpy().astype(np.float32)
#propa_lab = tf.keras.utils.to_categorical(propa_lab, num_classes=2)
propa_contexts = dfPropa['句子'].to_numpy()
# remove all not chinese, english and number tokens
filted_propa_contexts = remove_not_Ch_Eng(propa_contexts)

# prepare training & testing data
sh_propa_cont, sh_propa_lab = shuffle(filted_propa_contexts, propa_lab, random_state=10)
#X_train, X_test, y_train, y_test = train_test_split(sh_propa_cont, sh_propa_lab, test_size=0.2)
#train_ii, train_am = encode_data(X_train, tokenizer, sentence_len)
#test_ii, test_am = encode_data(X_test, tokenizer, sentence_len)


# In[6]:


# generate propa training data - old
dfPropa_old['宣傳手法'] = dfPropa_old['宣傳手法'].apply(lambda x: 0 if x == 'X' else 1)
propa_lab_old = dfPropa_old['宣傳手法'].to_numpy().astype(np.float32)
propa_contexts_old = dfPropa_old['句子'].to_numpy()
# remove all not chinese, english and number tokens
filted_propa_contexts_old = remove_not_Ch_Eng(propa_contexts_old)

# prepare training & testing data
sh_propa_cont_old, sh_propa_lab_old = shuffle(filted_propa_contexts_old, propa_lab_old, random_state=10)


# #### 5-fold

# In[7]:


# implementing cross validation - 1
k = 5
kf = KFold(n_splits=k, random_state=None, shuffle=True)

#Hist_li_slow = []
#Score_li_slow = []

for train_index, test_index in kf.split(sh_propa_cont):
    X_tr, X_te = sh_propa_cont[train_index], sh_propa_cont[test_index]
    y_tr, y_te = sh_propa_lab[train_index], sh_propa_lab[test_index]
    
    # encode training testing data
    tr_ii, tr_am = encode_data(X_tr, tok, sent_len)
    te_ii, te_am = encode_data(X_te, tok, sent_len)
    
    # build and train model
    lr = 3e-5
    epochs = 4
    batch_size = 16
    train_data_size = len(y_tr)
    
    model = build_model_propa_slow(base, lr, epochs, batch_size, train_data_size, sent_len)
    history = model.fit([tr_ii, tr_am], y_tr, epochs=epochs, batch_size=batch_size)
    #Hist_li_slow.append(history)
    plot_learning_curve(history)
    
    # score the model
    y_pred = model.predict([te_ii, te_am])
    y_pred_bool = y_pred.round()
    #Score_li_slow.append(classification_report(y_te, y_pred_bool))
    print(classification_report(y_te, y_pred_bool))


# In[8]:


'''
# 比較rapid 與 slow之間的差異

#Hist_li_rapid = []
#Score_li_rapid = []

for train_index, test_index in kf.split(sh_propa_cont):
    X_tr, X_te = sh_propa_cont[train_index], sh_propa_cont[test_index]
    y_tr, y_te = sh_propa_lab[train_index], sh_propa_lab[test_index]
    
    # encode training testing data
    tr_ii, tr_am = encode_data(X_tr, tok, sent_len)
    te_ii, te_am = encode_data(X_te, tok, sent_len)
    
    # build and train model
    lr = 3e-5
    epochs = 4
    batch_size = 16
    train_data_size = len(y_tr)
    
    model = build_model_propa_rapid(base, lr, epochs, batch_size, train_data_size, sent_len)
    history = model.fit([tr_ii, tr_am], y_tr, epochs=epochs, batch_size=batch_size)
    #Hist_li_rapid.append(history)
    plot_learning_curve(history)
    
    # score the model
    y_pred = model.predict([te_ii, te_am])
    y_pred_bool = y_pred.round()
    #Score_li_rapid.append(classification_report(y_te, y_pred_bool))
    print(classification_report(y_te, y_pred_bool))
'''


# In[9]:


# 比較新舊資料集的差異 (使用rapid模型)
# implementing cross validation - 3

#Hist_li_slow = []
#Score_li_slow = []

for train_index, test_index in kf.split(sh_propa_cont_old):
    X_tr, X_te = sh_propa_cont_old[train_index], sh_propa_cont_old[test_index]
    y_tr, y_te = sh_propa_lab_old[train_index], sh_propa_lab_old[test_index]
    
    # encode training testing data
    tr_ii, tr_am = encode_data(X_tr, tok, sent_len)
    te_ii, te_am = encode_data(X_te, tok, sent_len)
    
    # build and train model
    lr = 3e-5
    epochs = 4
    batch_size = 16
    train_data_size = len(y_tr)
    
    model = build_model_propa_rapid(base, lr, epochs, batch_size, train_data_size, sent_len)
    history = model.fit([tr_ii, tr_am], y_tr, epochs=epochs, batch_size=batch_size)
    #Hist_li_slow.append(history)
    plot_learning_curve(history)
    
    # score the model
    y_pred = model.predict([te_ii, te_am])
    y_pred_bool = y_pred.round()
    #Score_li_slow.append(classification_report(y_te, y_pred_bool))
    print(classification_report(y_te, y_pred_bool))


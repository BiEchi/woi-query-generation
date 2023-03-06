#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import json 
import openai
import backoff
import requests
import pandas as pd
from tqdm import tqdm

from utils import *


# In[6]:


train_sub1 = pd.read_json('./data/synthesized/train_sub1.jsonl', lines=True, orient='records')
train_sub1_ids = train_sub1['id'].unique()

cnt = 0
# we restart from dialog 2200
for train_sub1_id in tqdm(train_sub1_ids[6000:]):
    cnt += 1
    subset = train_sub1[train_sub1['id'] == train_sub1_id]
    start_idx = subset.index[0]
    for idx, utterance in subset.iterrows():
        cur_uter_idx = idx - start_idx
        if utterance['query'] == True:
            # get the entity
            entity = get_entity(cur_uter_idx, subset)
            train_sub1.loc[idx, 'entity'] = entity
            # get the cosmo response
            cosmo_utterance = get_cosmo_uter(cur_uter_idx, subset, entity)
            train_sub1.loc[idx, 'cosmo_utterance'] = cosmo_utterance
            # get the query
            query = get_query(cur_uter_idx, subset, cosmo_utterance, entity)
            train_sub1.loc[idx, 'query_gen'] = query
    if cnt % 1 == 0: 
        train_sub1.to_json('./data/synthesized/train_sub1.jsonl', orient='records', lines=True)


# In[ ]:





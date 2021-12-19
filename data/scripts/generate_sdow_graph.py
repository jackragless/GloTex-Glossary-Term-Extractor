#!/usr/bin/env python
# coding: utf-8

# In[1]:


data_loc = ''
import pandas as pd
from tqdm import tqdm
import itertools
import pickle
from csv import DictReader
import sys
import csv


# In[2]:


def dfToDict(df):
    outdict = {}
    for index, row in tqdm(df.iterrows(), total=len(df)):
        row = dict(row)
        outdict[row[list(row.keys())[0]]] = dict(itertools.islice(row.items(),1,None))
    return outdict


# In[3]:


pages = pd.read_csv(data_loc+'pages.csv')
page_dict = {}
graph_dict = {}
for index, row in tqdm(pages.iterrows(), total=len(pages)):
    row = dict(row)
    page_dict[row[list(row.keys())[1]]] = row[list(row.keys())[0]]
    graph_dict[row[list(row.keys())[0]]] = {list(row.keys())[2]:row[list(row.keys())[2]]}
del pages


# In[4]:


pickle.dump(page_dict,open(data_loc+"title2id.pkl", "wb" ))
del page_dict


# In[5]:


redirects = dfToDict(pd.read_csv(data_loc+'redirects.csv'))


# In[10]:


for k,v in tqdm(graph_dict.items()):
    if k in redirects.keys() and v['is_redirect'] == 1:
        v['redirect'] = redirects[k]['target_id']
    else:
        v['redirect'] = None
    del v['is_redirect']
    graph_dict[k] = v
del redirects


# In[12]:


csv.field_size_limit(sys.maxsize)
with open(data_loc+'links.csv', 'r') as read_obj:
    csv_dict_reader = DictReader(read_obj)
    count = 0
    for row in csv_dict_reader:
        row = dict(row)
        graph_dict[int(row['id'])].update(dict(itertools.islice(row.items(),1,None)))
        if not graph_dict[int(row['id'])]['outgoing_links']:
            graph_dict[int(row['id'])]['outgoing_links'] = None
        if not graph_dict[int(row['id'])]['incoming_links']:
            graph_dict[int(row['id'])]['incoming_links'] = None


# In[13]:


pickle.dump(graph_dict,open(data_loc+"sdow_graph.pkl", "wb" ))


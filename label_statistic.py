import pandas as pd
import random
import torch
import time
import datetime
import numpy as np
from gensim.models.word2vec import Word2Vec
from model import BatchProgramClassifier
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import sys

root = 'data/'
train_data = pd.read_pickle(root+'train/blocks.pkl')
val_data = pd.read_pickle(root+'dev/blocks.pkl')
test_data = pd.read_pickle(root+'test/blocks.pkl')

dict = {}
for i in range(0,18):
    dict[i] = 0
for _, item in train_data.iterrows():
    dict[item[2]] += 1
for _, item in val_data.iterrows():
    dict[item[2]] += 1
for _, item in test_data.iterrows():
    dict[item[2]] += 1
    
print(dict)
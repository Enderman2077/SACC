import pandas as pd
import random
import torch
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models.word2vec import Word2Vec
from model import BatchProgramClassifier
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import sys

mshape = [1,4,21,21]
ATTN = torch.zeros(mshape, dtype=torch.float32)

if __name__ == '__main__':
    root = 'data/'
    test_data = pd.read_pickle(root+'test/blocks.pkl')
    test_ap = pd.read_pickle(root+'ast_pattern/test.pkl')
    demo_epoch = 12
    model_path = root + 'model/2023-08-15_12-09-09/model_' + str(demo_epoch) + '.pth'
    demo_id = 2

    word2vec = Word2Vec.load(root+"train/embedding/node_w2v_128").wv
    embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    HIDDEN_DIM = 128
    ENCODE_DIM = 128
    LABELS = 18
    BATCH_SIZE = 1
    USE_GPU = True
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]

    model = BatchProgramClassifier(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                                   USE_GPU, embeddings)
    if USE_GPU:
        model.cuda()

    model.load_state_dict(torch.load(model_path))
    model.eval()

    def get_attention_weights(module, input, output):
        global ATTN
        ATTN = output[1][1]

    '''def get_batch(dataset, matrix, idx, bs):
        tmp = dataset.iloc[idx: idx+bs]
        tmp2 = matrix.iloc[idx: idx+bs]
        data, labels, ap = [], [], []
        for _, item in tmp.iterrows():
            data.append(item[1])
            labels.append(item[2])
        for _, item in tmp2.iterrows():
            ap.append(item[1])
        return [data, ap], torch.LongTensor(labels)'''

    target_submodel = model.transformer
    target_layer = target_submodel.encoder

    hook_handle = target_layer.register_forward_hook(get_attention_weights)

    # 获取input
    data, labels, ap = [], [], []
    data.append(test_data.iloc[demo_id][1])
    labels.append(test_data.iloc[demo_id][2])
    ap.append(test_ap.iloc[demo_id][1])
    test_inputs = [data, ap]
    test_labels = torch.LongTensor(labels)
    if USE_GPU:
        test_inputs, test_labels = test_inputs, test_labels.cuda()

    output = model(test_inputs)

    attn_np = ATTN.detach().cpu().numpy()
    print(attn_np.shape)

    def plot(matrix, y_labels):
        y_ticks = np.arange(len(y_labels)) + 0.5
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, cmap="YlOrRd", annot=False)
        plt.title("Full Attention Weights Heatmap")
        plt.yticks(y_ticks, y_labels, rotation=0)
        plt.gca().invert_yaxis()
        plt.savefig("heatmap_fa1.svg", format="svg")
    
    tick = []
    vocab_indices = word2vec.index2word
    for block in test_data.iloc[demo_id][1]:
        tick.append(vocab_indices[block[0]])
    
    plot(attn_np[0][1], tick)

    hook_handle.remove()

    _, predicted = torch.max(output.data, 1)
    print(predicted, test_labels)


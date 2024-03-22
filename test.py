import pandas as pd
import random
import torch
import time
import numpy as np
from gensim.models.word2vec import Word2Vec
from model import BatchProgramClassifier
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchmetrics import Precision, Recall, F1Score
import os
import sys

def get_batch(dataset, matrix, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    tmp2 = matrix.iloc[idx: idx+bs]
    data, labels, ap = [], [], []
    for _, item in tmp.iterrows():
        data.append(item[1])
        labels.append(item[2])
    for _, item in tmp2.iterrows():
        ap.append(item[1])
    return [data, ap], torch.LongTensor(labels)


if __name__ == '__main__':
    root = 'data/'
    test_data = pd.read_pickle(root+'test/blocks.pkl')
    test_ap = pd.read_pickle(root+'ast_pattern/test.pkl')
    test_epoch = 12
    model_path = root + 'model/2023-08-15_12-09-09/model_' + str(test_epoch) + '.pth'

    word2vec = Word2Vec.load(root+"train/embedding/node_w2v_128").wv
    embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    HIDDEN_DIM = 128
    ENCODE_DIM = 128
    LABELS = 18
    BATCH_SIZE = 16
    USE_GPU = True
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]

    precision = Precision(num_classes=18, average='macro').cuda()
    recall = Recall(num_classes=18, average='macro').cuda()
    f1 = F1Score(num_classes=18, average='macro').cuda()


    model = BatchProgramClassifier(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                                   USE_GPU, embeddings)
    if USE_GPU:
        model.cuda()

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # parameters = model.parameters()
    # optimizer = torch.optim.Adamax(parameters)
    # loss_function = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    print('Start testing...')
    # testing procedure
    
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    i = 0
    while i < len(test_data):
        batch = get_batch(test_data, test_ap, i, BATCH_SIZE)
        i += BATCH_SIZE
        test_inputs, test_labels = batch
        if USE_GPU:
            test_inputs, test_labels = test_inputs, test_labels.cuda()

        model.batch_size = len(test_labels)
        # model.hidden = model.init_hidden()
        output = model(test_inputs)
        # loss = loss_function(output, Variable(test_labels))

        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == test_labels).sum()
        total += len(test_labels)
        # total_loss += loss.item() * len(test_inputs)
        precision.update(predicted, test_labels)
        recall.update(predicted, test_labels)
        f1.update(output, test_labels)

    prec = precision.compute()
    rec = recall.compute()
    f1_score = f1.compute()
    print("Precision:", prec.item())
    print("Recall:", rec.item())
    print("F1 Score:", f1_score.item())
    print("Testing results(Acc):", total_acc.item() / total)
    print("Results from model_%d" % test_epoch)
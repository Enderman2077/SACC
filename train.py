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
from tqdm import tqdm
import os
import sys


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    #tmp2 = matrix.iloddc[idx: idx+bs]
    data, labels = [], []
    for _, item in tmp.iterrows():
        data.append(item[1])
        labels.append(item[2])
    #for _, item in tmp2.iterrows():
    #    ap.append(item[1])
    return data, torch.LongTensor(labels)


if __name__ == '__main__':
    root = 'data/'
    train_data = pd.read_pickle(root+'train/blocks.pkl')
    val_data = pd.read_pickle(root+'dev/blocks.pkl')
    test_data = pd.read_pickle(root+'test/blocks.pkl')
    #train_ap = pd.read_pickle(root+'ast_pattern/train.pkl')
    #val_ap = pd.read_pickle(root+'ast_pattern/dev.pkl')
    #test_ap = pd.read_pickle(root+'ast_pattern/test.pkl')
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
    #save_dir = root + 'model/' + timestamp
    #if not os.path.exists(save_dir):
    #    os.makedirs(save_dir)

    word2vec = Word2Vec.load(root+"train/embedding/node_w2v_128").wv
    embeddings = np.zeros((word2vec.vectors.shape[0] + 1, word2vec.vectors.shape[1]), dtype="float32")
    embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors

    HIDDEN_DIM = 128
    ENCODE_DIM = 128
    LABELS = 18
    EPOCHS = 30
    BATCH_SIZE = 64
    USE_GPU = True
    MAX_TOKENS = word2vec.vectors.shape[0]
    EMBEDDING_DIM = word2vec.vectors.shape[1]

    model = BatchProgramClassifier(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                                   USE_GPU, embeddings)
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters, lr=0.002)
    loss_function = torch.nn.CrossEntropyLoss()

    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []
    best_acc = 0.0
    print('Start training...')
    # training procedure
    best_model = model.state_dict()
    for epoch in range(EPOCHS):
        start_time = time.time()

        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for i in tqdm(range(0, len(train_data), BATCH_SIZE)):
            batch = get_batch(train_data, i, BATCH_SIZE)
            train_inputs, train_labels = batch
            if USE_GPU:
                train_inputs, train_labels = train_inputs, train_labels.cuda()

            model.zero_grad()
            model.batch_size = len(train_labels)
            # model.hidden = model.init_hidden()
            output = model(train_inputs)

            loss = loss_function(output, Variable(train_labels))
            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_labels).sum()
            total += len(train_labels)
            total_loss += loss.item()*len(train_inputs)

        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc.item() / total)
        # validation epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for i in tqdm(range(0, len(val_data), BATCH_SIZE)):
            batch = get_batch(val_data, i, BATCH_SIZE)
            val_inputs, val_labels = batch
            if USE_GPU:
                val_inputs, val_labels = val_inputs, val_labels.cuda()

            model.batch_size = len(val_labels)
            # model.hidden = model.init_hidden()
            output = model(val_inputs)

            loss = loss_function(output, Variable(val_labels))

            # calc valing acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == val_labels).sum()
            total += len(val_labels)
            total_loss += loss.item()*len(val_inputs)
        val_loss_.append(total_loss / total)
        val_acc_.append(total_acc.item() / total)
        end_time = time.time()
        if total_acc/total > best_acc:
            best_acc = total_acc/total
            best_model = model.state_dict()
        print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
              ' Training Acc: %.3f, Validation Acc: %.3f, Time Cost: %.3f s'
              % (epoch + 1, EPOCHS, train_loss_[epoch], val_loss_[epoch],
                 train_acc_[epoch], val_acc_[epoch], end_time - start_time))
        #if (epoch + 1) % 5 == 0:
        #    torch.save(model.state_dict(), save_dir + '/model_%d.pth' % (epoch + 1))
        #    print("Checkpoint saved")

    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    model.load_state_dict(best_model)

    for i in tqdm(range(0, len(test_data), BATCH_SIZE)):
        batch = get_batch(test_data, i, BATCH_SIZE)
        test_inputs, test_labels = batch
        if USE_GPU:
            test_inputs, test_labels = test_inputs, test_labels.cuda()

        model.batch_size = len(test_labels)
        # model.hidden = model.init_hidden()
        output = model(test_inputs)

        loss = loss_function(output, Variable(test_labels))

        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == test_labels).sum()
        total += len(test_labels)
        total_loss += loss.item() * len(test_inputs)
    print("Testing results(Acc):", total_acc.item() / total)
    #with open('output.txt', 'a') as file:
    #    file.write("model dir:%s, gp + lp\n" % timestamp)
    #    file.write("Testing results(Acc):%d\n\n\n" % total_acc.item() / total)


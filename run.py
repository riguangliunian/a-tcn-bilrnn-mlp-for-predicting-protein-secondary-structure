from typing import Iterable
from collections import Counter
# model是large model，Finalmodel是samll model
import gensim
from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm
import os,logging,pickle,random,torch
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from DL_ClassifierModel import *
from utils import *
dataClass = DataClass('/content/drive/MyDrive/protein/ts115.txt', '/content/drive/MyDrive/protein/ts115_Q3.txt', k=1, validSize=0.3, minCount=0)
trainStream = dataClass.random_batch_data_stream(batchSize=128, type='train', device="cuda", augmentation=0.5)
# 词向量预训练
dataClass.vectorize(method='char2vec', feaSize=25, sg=1)
dataClass.vectorize(method='feaEmbedding')
model = FinalModel(classNum=dataClass.classNum, embedding=dataClass.vector['embedding'], feaEmbedding=dataClass.vector['feaEmbedding'],useFocalLoss=True, device=torch.device('cuda'))
# 开始训练
model.cv_train( dataClass, trainSize=64, batchSize=64, epoch=1000, stopRounds=100, earlyStop=30, saveRounds=1,
                savePath='model/FinalModel', lr=3e-4, augmentation=0.1, kFold=3)

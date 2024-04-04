from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm
import os,logging,pickle,random,torch
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


featureDict = {
    'G':[57,   75,  156,   0.102,  0.085,  0.190,  0.152, 75.06714, 6.06, 2.34, 9.60, 48, 249.9, 0], # Glycine 甘氨酸
    'P':[57,   55,  152,   0.102,  0.301,  0.034,  0.068, 115.13194, 6.30, 1.99, 10.96, 90, 1620.0, 10.87], # Proline 脯氨酸
    'T':[83,  119,   96,   0.086,  0.108,  0.065,  0.079, 119.12034, 5.60, 2.09, 9.10, 93, 13.2, 1.67], # Threonine 苏氨酸
    'E':[151,   37,   74,   0.056,  0.060,  0.077,  0.064, 147.13074, 3.15, 2.10, 9.47, 109, 8.5, 2.09], # Glutamic Acid 谷氨酸
    'S':[77,   75,  143,   0.120,  0.139,  0.125,  0.106, 105.09344, 5.68, 2.21, 9.15, 73, 422.0, 1.25], # Serine 丝氨酸
    'K':[114,   74,  101,   0.055,  0.115,  0.072,  0.095, 146.18934, 9.60, 2.16, 9.06, 135, 739.0, 5.223888888888889], # Lysine 赖氨酸
    'C':[70,  119,  119,   0.149,  0.050,  0.117,  0.128, 121.15404, 5.05, 1.92, 10.70, 86, 280, 4.18], # Cysteine 半胱氨酸
    'L':[121,  130,   59,   0.061,  0.025,  0.036,  0.070, 131.17464, 6.01, 2.32, 9.58, 124, 21.7, 9.61], # Leucine 亮氨酸
    'M':[145,  105,   60,   0.068,  0.082,  0.014,  0.055, 149.20784, 5.74, 2.28, 9.21, 124, 56.2, 5.43], # Methionine 蛋氨酸
    'V':[106,  170,   50,   0.062,  0.048,  0.028,  0.053, 117.14784, 6.00, 2.29, 9.74, 105, 58.1, 6.27], # Valine 缬氨酸
    'D':[67,   89,  156,   0.161,  0.083,  0.191,  0.091, 133.10384, 2.85, 1.99, 9.90, 96, 5.0, 2.09], # Asparagine 天冬氨酸
    'A':[142,   83,   66,   0.06,   0.076,  0.035,  0.058, 89.09404, 6.01, 2.35, 9.87, 67, 167.2, 2.09], # Alanine 丙氨酸
    'R':[98,   93,   95,   0.070,  0.106,  0.099,  0.085, 174.20274, 10.76, 2.17, 9.04, 148, 855.6, 5.223888888888889], # Arginine 精氨酸
    'I':[108,  160,   47,   0.043,  0.034,  0.013,  0.056, 131.17464, 6.05, 2.32, 9.76, 124, 34.5, 12.54], # Isoleucine 异亮氨酸
    'N':[101,   54,  146,   0.147,  0.110,  0.179,  0.081, 132.11904, 5.41, 2.02, 8.80, 91, 28.5, 0], # Aspartic Acid 天冬酰胺
    'H':[100,   87,   95,   0.140,  0.047,  0.093,  0.054, 155.15634, 7.60, 1.80, 9.33, 118, 41.9, 2.09], # Histidine 组氨酸
    'F':[113,  138,   60,   0.059,  0.041,  0.065,  0.065, 165.19184, 5.49, 2.20, 9.60, 135, 27.6, 10.45], # Phenylalanine 苯丙氨酸
    'W':[108,  137,   96,   0.077,  0.013,  0.064,  0.167, 204.22844, 5.89, 2.46, 9.41, 163, 13.6, 14.21], # Tryptophan 色氨酸
    'Y':[69,  147,  114,   0.082,  0.065,  0.114,  0.125, 181.19124, 5.64, 2.20, 9.21, 141, 0.4, 9.61], # Tyrosine 酪氨酸
    'Q':[111,  110,   98,   0.074,  0.098,  0.037,  0.098, 146.14594, 5.65, 2.17, 9.13, 114, 4.7, -0.42], # Glutamine 谷氨酰胺
    'X':[99.9, 102.85, 99.15, 0.0887, 0.08429999999999999, 0.0824, 0.0875, 136.90127, 6.027, 2.1690000000000005, 0.0875, 109.2, 232.37999999999997, 5.223888888888889],
    'U':[99.9, 102.85, 99.15, 0.08870000000000001, 0.08430000000000001, 0.0824, 0.08750000000000001, 169.06, 6.026999999999999, 2.1690000000000005, 9.081309523809526, 109.19999999999999, 232.37999999999997, 5.223888888888889],
    'Z':[99.9, 102.85, 99.15, 0.08870000000000001, 0.08430000000000001, 0.0824, 0.08750000000000001, 136.90126999999998, 6.026999999999999, 2.1690000000000005, 9.081309523809526, 109.19999999999999, 232.37999999999997, 5.223888888888889],
}

class DataClass:
    def __init__(self, seqPath, secPath, validSize=0.3, k=3, minCount=10):
        # Open files and load data
        with open(seqPath,'r') as f:
            seqData = [' '*(k//2)+i[:-1]+' '*(k//2) for i in f.readlines()]
        with open(secPath,'r') as f:
            secData = [i[:-1] for i in f.readlines()]
        self.tmp,self.k = seqData,k
        seqData = [[seq[i-k//2:i+k//2+1] for i in range(k//2,len(seq)-k//2)] for seq in seqData]
        # Dropping uncommon items
        itemCounter = {}
        for seq in seqData:
            for i in seq:
                itemCounter[i] = itemCounter.get(i,0)+1
        seqData = [[i if itemCounter[i]>=minCount else "<UNK>" for i in seq] for seq in seqData]
        self.rawSeq,self.rawSec = seqData,secData
        self.minCount = minCount
        # Get mapping variables
        self.seqItem2id,self.id2seqItem = {"<EOS>":0, "<UNK>":1},["<EOS>", "<UNK>"]
        self.secItem2id,self.id2secItem = {"<EOS>":0},["<EOS>"]
        cnt = 2
        for seq in seqData:
            for i in seq:
                if i not in self.seqItem2id:
                    self.seqItem2id[i] = cnt
                    self.id2seqItem.append(i)
                    cnt += 1
        self.seqItemNum = cnt
        cnt = 1
        for sec in secData:
            for i in sec:
                if i not in self.secItem2id:
                    self.secItem2id[i] = cnt
                    self.id2secItem.append(i)
                    cnt += 1
        self.classNum = cnt
        # Tokenized the seq
        self.tokenizedSeq,self.tokenizedSec = np.array([[self.seqItem2id[i] for i in seq] for seq in seqData]),np.array([[self.secItem2id[i] for i in sec] for sec in secData])
        self.seqLen,self.secLen = np.array([len(seq)+1 for seq in seqData]),np.array([len(sec)+1 for sec in secData])
        self.trainIdList,self.validIdList = train_test_split(range(len(seqData)), test_size=validSize) if validSize>0.0 else (list(range(seqData)),[])
        self.trainSampleNum,self.validSampleNum = len(self.trainIdList),len(self.validIdList)
        self.totalSampleNum = self.trainSampleNum+self.validSampleNum
        self.vector = {}
        print('classNum:',self.classNum)
        print(f'seqItemNum:{self.seqItemNum}')
        print('train sample size:',len(self.trainIdList))
        print('valid sample size:',len(self.validIdList))
    def describe(self):
        pass
        '''
        trainSec,validSec = np.hstack(self.tokenizedSec[self.trainIdList]),np.hstack(self.tokenizedSec[self.validIdList])
        trainPad,validPad = self.trainSampleNum*self.seqLen.max()-len(trainSec),self.validSampleNum*self.seqLen.max()-len(validSec)
        trainSec,validSec = np.hstack([trainSec,[0]*trainPad]),np.hstack([validSec,[0]*validPad])
        print('===========DataClass Describe===========')
        print(f'{"CLASS":<16}{"TRAIN":<8}{"VALID":<8}')
        for i,c in enumerate(self.id2secItem):
            trainIsC = sum(trainSec==i)/self.trainSampleNum if self.trainSampleNum>0 else -1.0
            validIsC = sum(validSec==i)/self.validSampleNum if self.validSampleNum>0 else -1.0
            print(f'{c:<16}{trainIsC:<8.3f}{validIsC:<8.3f}')
        print('========================================')
        '''
    def vectorize(self, method="char2vec", feaSize=128, window=13, sg=1, 
                        workers=8, loadCache=True):
        if method=='feaEmbedding': loadCache = False
        vecPath = f'cache/{method}_k{self.k}_d{feaSize}.pkl'
        if os.path.exists(vecPath) and loadCache:
            with open(vecPath, 'rb') as f:
                self.vector['embedding'] = pickle.load(f)
            print(f'Loaded cache from cache/{vecPath}.')
            return
        if method == 'char2vec':
            doc = [list(i)+['<EOS>'] for i in self.rawSeq]
            model = Word2Vec(doc, min_count=self.minCount, window=window, vector_size=feaSize, workers=workers, sg=sg, epochs=10)
            char2vec = np.random.random((self.seqItemNum, feaSize))
            for i in range(self.seqItemNum):
                if self.id2seqItem[i] in model.wv:
                    char2vec[i] = model.wv[self.id2seqItem[i]]
                else:
                    print(self.id2seqItem[i],'not in training docs...')
            self.vector['embedding'] = char2vec
            with open(vecPath, 'wb') as f:
                pickle.dump(self.vector['embedding'], f, protocol=4)
        elif method == 'feaEmbedding':
            oh = np.eye(self.seqItemNum)
            feaAppend = []
            for i in range(self.seqItemNum):
                item = self.id2seqItem[i]
                if item in featureDict:
                    feaAppend.append( featureDict[item] )
                else:
                    feaAppend.append( np.random.random(14) )
            emb = np.hstack([oh, np.array(feaAppend)]).astype('float32')
            mean,std = emb.mean(axis=0),emb.std(axis=0)
            self.vector['feaEmbedding'] = (emb-mean)/(std+1e-10)

    def vector_merge(self, vecList, mergeVecName='mergeVec'):
        self.vector[mergeVec] = np.hstack([self.vector[i] for i in vecList])
        print(f'Get a new vector "{mergeVec}" with shape {self.vector[mergeVec].shape}...')

    def random_batch_data_stream(self, batchSize=128, type='train', device=torch.device('cpu'), augmentation=0.05):
        idList = [i for i in self.trainIdList] if type=='train' else [i for i in self.validIdList]
        X,XLen,Y = self.tokenizedSeq,self.seqLen,self.tokenizedSec
        seqMaxLen = XLen.max()
        while True:
            random.shuffle(idList)
            for i in range((len(idList)+batchSize-1)//batchSize):
                samples = idList[i*batchSize:(i+1)*batchSize]
                yield {
                        "seqArr":torch.tensor([[i if random.random()>augmentation else self.seqItem2id['<UNK>'] for i in seq]+[0]*(seqMaxLen-len(seq)) for seq in X[samples]], dtype=torch.long).to(device), \
                        "seqLenArr":torch.tensor(XLen[samples], dtype=torch.int).to(device)
                      }, torch.tensor([i+[0]*(seqMaxLen-len(i)) for i in Y[samples]], dtype=torch.long).to(device)

    def one_epoch_batch_data_stream(self, batchSize=128, type='valid', device=torch.device('cpu')):
        idList = [i for i in self.trainIdList] if type=='train' else [i for i in self.validIdList]
        X,XLen,Y = self.tokenizedSeq,self.seqLen,self.tokenizedSec
        seqMaxLen = XLen.max()
        for i in range((len(idList)+batchSize-1)//batchSize):
            samples = idList[i*batchSize:(i+1)*batchSize]
            yield {
                    "seqArr":torch.tensor([i+[0]*(seqMaxLen-len(i)) for i in X[samples]], dtype=torch.long).to(device), \
                    "seqLenArr":torch.tensor(XLen[samples], dtype=torch.int).to(device)
                  }, torch.tensor([i+[0]*(seqMaxLen-len(i)) for i in Y[samples]], dtype=torch.long).to(device)


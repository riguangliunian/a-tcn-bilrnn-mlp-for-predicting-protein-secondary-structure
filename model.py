from layers import *
from utils import *

##student model
class FinalModel(BaseClassifier):
    def __init__(self, classNum, embedding, feaEmbedding, feaSize=64,
                 filterNum=128, contextSizeList=[1,9,81],
                 hiddenSize=512, num_layers=3,
                 hiddenList=[2048],
                 embDropout=0.2, BiGRUDropout=0.2, fcDropout=0.4,
                 useFocalLoss=False, weight=-1, device=torch.device("cuda:0")):
        self.textEmbedding = TextEmbedding( torch.tensor(embedding, dtype=torch.float),dropout=embDropout ).to(device)
        self.feaEmbedding = TextEmbedding( torch.tensor(feaEmbedding, dtype=torch.float),dropout=embDropout//2,name='feaEmbedding',freeze=True ).to(device)
        self.textCNN = TextTCN( feaSize, contextSizeList, filterNum ).to(device)
        self.textBiGRU = TextBiGRULSTM(len(contextSizeList)*filterNum, hiddenSize, num_layers=num_layers, dropout=BiGRUDropout).to(device)
        self.fcLinear = MLP(len(contextSizeList)*filterNum+hiddenSize*2, classNum, hiddenList, fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.textEmbedding,self.feaEmbedding,self.textCNN,self.textBiGRU,self.fcLinear])
        self.classNum = classNum
        self.device = device
        self.feaSize = feaSize
        self.criterion = nn.CrossEntropyLoss() if not useFocalLoss else FocalCrossEntropyLoss(weight=weight).to(device)
    def calculate_y_logit(self, X):
        X = X['seqArr']
        X = torch.cat([self.textEmbedding(X),self.feaEmbedding(X)], dim=2) # => batchSize × seqLen × feaSize
        X_conved = self.textCNN(X) # => batchSize × seqLen × scaleNum*filterNum
        X_BiGRUed = self.textBiGRU(X_conved, None) # => batchSize × seqLen × hiddenSize*2
        X = torch.cat([X_conved,X_BiGRUed], dim=2) # => batchSize × seqLen × (scaleNum*filterNum+hiddenSize*2)
        return self.fcLinear(X) # => batchSize × seqLen × classNum
    def calculate_y_prob(self, X):
        Y_pre = self.calculate_y_logit(X)
        return torch.softmax(Y_pre, dim=2)
    def calculate_y(self, X):
        Y_pre = self.calculate_y_prob(X)
        return torch.argmax(Y_pre, dim=2)
    def calculate_y_by_iterator(self, dataStream):
        Y_preArr, YArr = self.calculate_y_prob_by_iterator(dataStream)
        return Y_preArr.argmax(axis=2), YArr
    def calculate_loss(self, X, Y):
        Y_logit = self.calculate_y_logit(X)
        Y = Y.reshape(-1)
        Y_logit = Y_logit.reshape(len(Y),-1)
        return self.criterion(Y_logit, Y)
    def calculate_y_prob_by_iterator(self, dataStream):
        YArr,Y_preArr = [],[]
        while True:
            try:
                X,Y = next(dataStream)
            except:
                break
            Y_pre,Y = self.calculate_y_prob(X).cpu().data.numpy(),Y.cpu().data.numpy()
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.vstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        return Y_preArr, YArr





















##teacher model
##加载下载ProtT5模型
#@title Install requirements. { display-mode: "form" }
# Install requirements
!pip install torch transformers sentencepiece h5py

#@title Set up working directories and download files/checkpoints. { display-mode: "form" }
# Create directory for storing model weights (2.3GB) and example sequences.
# Here we use the encoder-part of ProtT5-XL-U50 in half-precision (fp16) as
# it performed best in our benchmarks (also outperforming ProtBERT-BFD).
# Also download secondary structure prediction checkpoint to show annotation extraction from embeddings
!mkdir protT5 # root directory for storing checkpoints, results etc
!mkdir protT5/protT5_checkpoint # directory holding the ProtT5 checkpoint
!mkdir protT5/sec_struct_checkpoint # directory storing the supervised classifier's checkpoint
!mkdir protT5/output # directory for storing your embeddings & predictions
# Huge kudos to the bio_embeddings team here! We will integrate the new encoder, half-prec ProtT5 checkpoint soon
!wget -nc -P protT5/sec_struct_checkpoint http://data.bioembeddings.com/public/embeddings/feature_models/t5/secstruct_checkpoint.pt


# In the following you can define your desired output. Current options:
# per_residue embeddings
# per_protein embeddings
# secondary structure predictions


# whether to retrieve embeddings for each residue in a protein
# --> Lx1024 matrix per protein with L being the protein's length
# as a rule of thumb: 1k proteins require around 1GB RAM/disk
per_residue = True
per_residue_path = "./protT5/output/per_residue_embeddings.h5" # where to store the embeddings

# whether to retrieve per-protein embeddings
# --> only one 1024-d vector per protein, irrespective of its length
per_protein = True
per_protein_path = "./protT5/output/per_protein_embeddings.h5" # where to store the embeddings

# whether to retrieve secondary structure predictions
# This can be replaced by your method after being trained on ProtT5 embeddings
sec_struct = True
sec_struct_path = "./protT5/output/ss3_preds.fasta" # file for storing predictions

# make sure that either per-residue or per-protein embeddings are stored
assert per_protein is True or per_residue is True or sec_struct is True, print(
    "Minimally, you need to active per_residue, per_protein or sec_struct. (or any combination)")

#检查GPU能否用
#@title Import dependencies and check whether GPU is available. { display-mode: "form" }
from transformers import T5EncoderModel, T5Tokenizer
import torch
import h5py
import time
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using {}".format(device))

##预测模型
#@title Network architecture for secondary structure prediction. { display-mode: "form" }
# Convolutional neural network (two convolutional layers) to predict secondary structure
class ConvNet( torch.nn.Module ):
    def __init__( self ):
        super(ConvNet, self).__init__()
        # This is only called "elmo_feature_extractor" for historic reason
        # CNN weights are trained on ProtT5 embeddings
        self.elmo_feature_extractor = torch.nn.Sequential(
                        torch.nn.Conv2d( 1024, 32, kernel_size=(7,1), padding=(3,0) ), # 7x32
                        torch.nn.ReLU(),
                        torch.nn.Dropout( 0.25 ),
                        )
        n_final_in = 32
        self.dssp3_classifier = torch.nn.Sequential(
                        torch.nn.Conv2d( n_final_in, 3, kernel_size=(7,1), padding=(3,0)) # 7
                        )

        self.dssp8_classifier = torch.nn.Sequential(
                        torch.nn.Conv2d( n_final_in, 8, kernel_size=(7,1), padding=(3,0))
                        )
        self.diso_classifier = torch.nn.Sequential(
                        torch.nn.Conv2d( n_final_in, 2, kernel_size=(7,1), padding=(3,0))
                        )


    def forward( self, x):
        # IN: X = (B x L x F); OUT: (B x F x L, 1)
        x = x.permute(0,2,1).unsqueeze(dim=-1)
        x         = self.elmo_feature_extractor(x) # OUT: (B x 32 x L x 1)
        d3_Yhat   = self.dssp3_classifier( x ).squeeze(dim=-1).permute(0,2,1) # OUT: (B x L x 3)
        d8_Yhat   = self.dssp8_classifier( x ).squeeze(dim=-1).permute(0,2,1) # OUT: (B x L x 8)
        diso_Yhat = self.diso_classifier(  x ).squeeze(dim=-1).permute(0,2,1) # OUT: (B x L x 2)
        return d3_Yhat, d8_Yhat, diso_Yhat
    

##@title Load the checkpoint for secondary structure prediction. { display-mode: "form" }
def load_sec_struct_model():
  checkpoint_dir="./protT5/sec_struct_checkpoint/secstruct_checkpoint.pt"
  state = torch.load( checkpoint_dir )
  model = ConvNet()
  model.load_state_dict(state['state_dict'])
  model = model.eval()
  model = model.to(device)
  print('Loaded sec. struct. model from epoch: {:.1f}'.format(state['epoch']))

  return model

#@title Load encoder-part of ProtT5 in half-precision. { display-mode: "form" }
# Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50)
def get_T5_model():
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    return model, tokenizer


##读取TXT文本
seq_path2="/content/data1.txt"
sec_path2="/content/data2.txt"
def normalize_sequence(seq):
    '''
        Normalize a protein sequence by converting to uppercase,
        replacing non-standard amino acids, and removing gaps.
    '''
    seq = seq.upper()
    seq = seq.replace("-", "")
    seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X")
    return seq

def read_text(txt_path):
    """
    Reads sequences from a text file.
    Each line in the text file is treated as a separate sequence.

    Args:
        txt_path (str): Path to the input text file.

    Returns:
        list: A list containing the normalized sequences read from the file.
    """
    sequences = []
    with open(txt_path, 'r') as txt_file:
        current_sequence = ''
        for line in txt_file:
            line = line.strip()
            if line:
                current_sequence += line
                sequences.append(normalize_sequence(current_sequence))
                current_sequence = ''

    if current_sequence:
        sequences.append(normalize_sequence(current_sequence))

    return sequences

def convert_predictions_to_dict(predictions):
    '''
    将预测结果列表转换为字典对象。

    参数：
    predictions (list): 包含预测结果的列表。

    返回：
    dict: 包含转换后的字典对象。
    '''
    converted_predictions = {
        str(idx): pred
        for idx, pred in enumerate(predictions)
    }
    return converted_predictions

seq2=read_text(seq_path2)
sec2=read_text(sec_path2)

Y=convert_predictions_to_dict(sec2)

#@title Generate embeddings with Learning Rate and Optimizer { display-mode: "form" }
# Generate embeddings via batch-processing
# per_residue indicates that embeddings for each residue in a protein should be returned.
# per_protein indicates that embeddings for a whole protein should be returned (average-pooling)
# max_residues gives the upper limit of residues within one batch
# max_seq_len gives the upper sequences length for applying batch-processing
# max_batch gives the upper number of sequences per batch
def get_embeddings( model, tokenizer, seqs, per_residue, per_protein, sec_struct,
                   max_residues=4000, max_seq_len=1000, max_batch=100, learning_rate=0.001 ):

    if sec_struct:
      sec_struct_model = load_sec_struct_model()

    results = {"residue_embs" : dict(),
               "protein_embs" : dict(),
               "sec_structs" : dict()
               }

    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    seq_dict   = sorted( seqs.items(), key=lambda kv: len( seqs[kv[0]] ), reverse=True )
    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict,1):
        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id,seq,seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed
        n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # add_special_tokens adds extra token at the end of each sequence
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            if sec_struct: # in case you want to predict secondary structure from embeddings
              d3_Yhat, d8_Yhat, diso_Yhat = sec_struct_model(embedding_repr.last_hidden_state)


            for batch_idx, identifier in enumerate(pdb_ids): # for each protein in the current mini-batch
                s_len = seq_lens[batch_idx]
                # slice off padding --> batch-size x seq_len x embedding_dim
                emb = embedding_repr.last_hidden_state[batch_idx,:s_len]
                if sec_struct: # get classification results
                    results["sec_structs"][identifier] = torch.max( d8_Yhat[batch_idx,:s_len], dim=1 )[1].detach().cpu().numpy().squeeze()

                if per_residue: # store per-residue embeddings (Lx1024)
                    results["residue_embs"][ identifier ] = emb.detach().cpu().numpy().squeeze()
                if per_protein: # apply average-pooling to derive per-protein embeddings (1024-d)
                    protein_emb = emb.mean(dim=0)
                    results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()

    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 使用Adam优化器，学习率为指定的learning_rate

    passed_time=time.time()-start
    avg_time = passed_time/len(results["residue_embs"]) if per_residue else passed_time/len(results["protein_embs"])
    print('\n############# EMBEDDING STATS #############')
    print('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
    print('Total number of per-protein embeddings: {}'.format(len(results["protein_embs"])))
    print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
        passed_time/60, avg_time ))
    print('\n############# END #############')
    return results



##设置对应输出键值
#八维
def convert_predictions_to_letters(predictions):
    class_mapping = {0:"G",1:"H",2:"I",3:"B",4:"E",5:"S",6:"T",7:"C"}
#    class_mapping = {0:"H",1:"E",2:"C"} 三维情况
    converted_predictions = {
        seq_id: [class_mapping[j] for j in yhat]
        for seq_id, yhat in predictions.items()
    }
    return converted_predictions


#加载计算模型
# Load the encoder part of ProtT5-XL-U50 in half-precision (recommended)
model, tokenizer = get_T5_model()

# Load example fasta.
#seqs = read_fasta( seq_path )


# 这里写入您要运行的代码块

# Compute embeddings and/or secondary structure predictions
results = get_embeddings( model, tokenizer, seqs,
                         per_residue, per_protein, sec_struct,learning_rate=0.0001)


##蒸馏
from teacher import *
from student import *
import random
import torch
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn.functional as F

seed = 42
random.seed(seed)

#开始进行知识蒸馏算法
#model1小模型；
#蒸馏温度
T=7
hard_loss=nn.CrossEntropyLoss()
alpha=0.3
soft_loss=nn.KLDivLoss(reduction="batchmean")
optim=torch.optim.Adam(model.parameters(),lr=0.0001)

all_values = [value for value in sorted_dict.values()]
Y_hat = [[value] for value in all_values]

for i in range(len(Y_hat)):
# 获取NumPy数组
  arr = Y_hat[i]
# 使用 ndim 属性查看数组的维度
  dimensions = np.ndim(arr)
  if dimensions==1:
    print(i)
    
formatted_data = [[list(arr) for arr in sublist] for sublist in Y_hat]

new = []

for i in range(len(Y_true)):
    flat_list = [item for sublist in Y_true[i] for item in sublist]
    new.append(flat_list)
new2 = []

for i in range(len(Y_hat)):
    flat_list = [item for sublist in Y_hat[i] for item in sublist]
    new2.append(flat_list)
    
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ParallelSequenceDataLoader:
    def __init__(self, data1, data2, batch_size):
        assert len(data1) == len(data2), "Data lengths must be the same"

        self.data1 = data1
        self.data2 = data2
        self.batch_size = batch_size
        self.num_samples = len(data1)
        self.indices = list(range(self.num_samples))
        self.current_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx >= self.num_samples:
            raise StopIteration

        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        batch_data1 = [self.data1[idx] for idx in batch_indices]
        batch_data2 = [self.data2[idx] for idx in batch_indices]

        self.current_idx += self.batch_size

        # Padding
        max_length1 = max([len(seq) for seq in batch_data1])
        max_length2 = max([len(seq) for seq in batch_data2])

        padded_batch_data1 = [seq + [0] * (max_length1 - len(seq)) for seq in batch_data1]
        padded_batch_data2 = [seq + [0] * (max_length2 - len(seq)) for seq in batch_data2]

        # Convert to tensors
        padded_batch_tensor1 = torch.tensor(padded_batch_data1)
        padded_batch_tensor2 = torch.tensor(padded_batch_data2)

        return padded_batch_tensor1, padded_batch_tensor2

batch_size = 64

data_loader = ParallelSequenceDataLoader(new, new2, batch_size)

# Get a batch of X and X2 as Tensors
batch_X, batch_X2 = next(data_loader)

##第二种蒸馏
seq_len = []

# 遍历 Y_hat 列表中的每个 NumPy 数组
for i in range(5000):  # 请注意这里应该是 range(5000)，不是 range(0:4999)
    # 计算当前 NumPy 数组的长度
    array_lengths = [len(arr) for arr in Y_true[i]]
    seq_len.extend(array_lengths)

# 输出结果
print("Lengths of arrays:", seq_len)

# 找到最大的子数组长度
max_length = max(seq_len)

# 创建一个列表来存储转换后的 PyTorch 张量
tensor_list = []

# 将每个 NumPy 数组转换为 PyTorch 张量并保存到列表中
for arr in Y_hat:
    tensor_arr = torch.tensor(arr, dtype=torch.int64)
    tensor_list.append(tensor_arr)

# 参数设置
batch_size = 64
class_num = 9   # 替换为你的实际类别数量

def process_tensor_list(tensor_list, class_num):
    processed_list = []

    for prediction in tensor_list:
        seq_len = len(prediction)  # 获取当前张量的长度

        # 将预测结果 reshape 成 (1, seq_len)
        prediction_reshaped = prediction.view(1, -1)  # 使用 -1 自动计算维度

        # 将预测结果转换为 one-hot 编码
        prediction_onehot = F.one_hot(prediction_reshaped, num_classes=class_num)

        processed_list.append(prediction_onehot)

    return processed_list

processed_tensor_list = process_tensor_list(tensor_list, class_num)

def process_and_softmax(tensor_list, class_num):
    processed_list = []

    for prediction in tensor_list:
        seq_len = len(prediction)  # 获取当前张量的长度

        # 将预测结果 reshape 成 (1, seq_len)
        prediction_reshaped = prediction.view(1, -1)  # 使用 -1 自动计算维度

        # 将预测结果转换为 one-hot 编码
        prediction_onehot = F.one_hot(prediction_reshaped, num_classes=class_num)

        # 将 one-hot 编码的张量转换为浮点数
        prediction_onehot_float = prediction_onehot.float()

        # 使用 softmax 函数将预测值转换为概率分布
        probabilities = F.softmax(prediction_onehot_float, dim=2)

        processed_list.append(probabilities)

    return processed_list

# 示例使用
class_num = 9

# 调用函数进行处理和 softmax 转换
processed_softmax_list = process_and_softmax(tensor_list, class_num)

# 输出结果
for probabilities in processed_softmax_list:
    print(probabilities)

# 找到最大的序列长度
max_seq_len = max([tensor.size(2) for tensor in processed_softmax_list])

# 手动进行 padding
padded_tensors = []
for tensor in processed_softmax_list:
    pad_size = max_seq_len - tensor.size(2)
    padded_tensor = torch.nn.functional.pad(tensor, (0, pad_size))
    padded_tensors.append(padded_tensor)

# 创建一个 TensorDataset，用于构建 DataLoader
dataset = TensorDataset(*padded_tensors)

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

############蒸馏实现
class BaseClassifier:
    def __init__(self):
        pass
    def calculate_y_logit(self, X, XLen):
        pass
    def cv_train(self, dataClass, trainSize=256, batchSize=256, epoch=100, stopRounds=10, earlyStop=10, saveRounds=1, augmentation=0.05,
                 optimType='Adam', lr=0.001, weightDecay=0, kFold=5, isHigherBetter=True, metrics="Score", report=["ACC", "MaF", "Score"],
                 savePath='model'):
        kf = KFold(n_splits=kFold)
        validRes = []
        for i,(trainIndices,validIndices) in enumerate(kf.split(range(dataClass.totalSampleNum))):
            print(f'CV_{i+1}:')
            self.reset_parameters()
            dataClass.trainIdList,dataClass.validIdList = trainIndices,validIndices
            dataClass.trainSampleNum,self.validSampleNum = len(trainIndices),len(validIndices)
            dataClass.describe()
            res = self.train(dataClass,trainSize,batchSize,epoch,stopRounds,earlyStop,saveRounds,augmentation,optimType,lr,weightDecay,
                             isHigherBetter,metrics,report,f"{savePath}_cv{i+1}")
            validRes.append(res)
        Metrictor.table_show(validRes, report)
    def train(self, dataClass, trainSize=256, batchSize=256, epoch=100, stopRounds=10, earlyStop=10, saveRounds=1, augmentation=0.05,
              optimType='Adam', lr=0.001, weightDecay=0, isHigherBetter=True, metrics="Score", report=["ACC", "MaF", "Score"],
              savePath='model'):
        assert batchSize%trainSize==0
        metrictor = Metrictor(dataClass.classNum)
        self.stepCounter = 0
        self.stepUpdate = batchSize//trainSize
        optimizer = torch.optim.Adam(self.moduleList.parameters(), lr=lr, weight_decay=weightDecay)
        trainStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='train', device=self.device, augmentation=augmentation)
        itersPerEpoch = (dataClass.trainSampleNum+trainSize-1)//trainSize
        mtc,bestMtc,stopSteps = 0.0,0.0,0
        if dataClass.validSampleNum>0: validStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='valid', device=self.device, augmentation=augmentation)
        st = time.time()
        for e in range(epoch):
            for i in range(itersPerEpoch):
                self.to_train_mode()
                X,Y = next(trainStream)
                # 蒸馏
                student_loss = self.calculate_loss(X,Y)
                # 计算教师模型的loss
                yh,yt=next(data_loader)
                teacher_loss=hard_loss(yh.float(),yt.float())
                # 综合教师和学生损失来进行反向传播和优化
                optimizer.zero_grad()
                total_loss = alpha * teacher_loss + (1 - alpha) * student_loss
                total_loss.backward()
                optimizer.step()
                if stopRounds>0 and (e*itersPerEpoch+i+1)%stopRounds==0:
                    self.to_eval_mode()
                    print(f"After iters {e*itersPerEpoch+i+1}: [train] loss= {loss:.3f};", end='')
                    if dataClass.validSampleNum>0:
                        X,Y = next(validStream)
                        loss = self.calculate_loss(X,Y)
                        print(f' [valid] loss= {loss:.3f};', end='')
                    restNum = ((itersPerEpoch-i-1)+(epoch-e-1)*itersPerEpoch)*trainSize
                    speed = (e*itersPerEpoch+i+1)*trainSize/(time.time()-st)
                    print(" speed: %.3lf items/s; remaining time: %.3lfs;"%(speed, restNum/speed))
            if dataClass.validSampleNum>0 and (e+1)%saveRounds==0:
                self.to_eval_mode()
                print(f'========== Epoch:{e+1:5d} ==========')
                #Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='train', device=self.device))
                #metrictor.set_data(Y_pre, Y)
                #print(f'[Total Train]',end='')
                #metrictor(report)
                print(f'[Total Valid]',end='')
                Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='valid', device=self.device))
                metrictor.set_data(Y_pre, Y)
                res = metrictor(report)
                mtc = res[metrics]
                print('=================================')
                if (mtc>bestMtc and isHigherBetter) or (mtc<bestMtc and not isHigherBetter):
                    print(f'Bingo!!! Get a better Model with val {metrics}: {mtc:.3f}!!!')
                    bestMtc = mtc
                    self.save("%s.pkl"%savePath, e+1, bestMtc, dataClass)
                    stopSteps = 0
                else:
                    stopSteps += 1
                    if stopSteps>=earlyStop:
                        print(f'The val {metrics} has not improved for more than {earlyStop} steps in epoch {e+1}, stop training.')
                        break
        self.load("%s.pkl"%savePath)
        os.rename("%s.pkl"%savePath, "%s_%s.pkl"%(savePath, ("%.3lf"%bestMtc)[2:]))
        print(f'============ Result ============')
        Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='train', device=self.device))
        metrictor.set_data(Y_pre, Y)
        print(f'[Total Train]',end='')
        metrictor(report)
        Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='valid', device=self.device))
        metrictor.set_data(Y_pre, Y)
        print(f'[Total Valid]',end='')
        res = metrictor(report)
        metrictor.each_class_indictor_show(dataClass.id2secItem)
        print(f'================================')
        return res
    def reset_parameters(self):
        for module in self.moduleList:
            for subModule in module.modules():
                if hasattr(subModule, "reset_parameters"):
                    subModule.reset_parameters()
    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc}
        for module in self.moduleList:
            stateDict[module.name] = module.state_dict()
        if dataClass is not None:
            stateDict['trainIdList'],stateDict['validIdList'] = dataClass.trainIdList,dataClass.validIdList
            stateDict['seqItem2id'],stateDict['id2seqItem'] = dataClass.seqItem2id,dataClass.id2seqItem
            stateDict['secItem2id'],stateDict['id2secItem'] = dataClass.secItem2id,dataClass.id2secItem
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.moduleList:
            module.load_state_dict(parameters[module.name])
        if dataClass is not None:
            dataClass.trainIdList,dataClass.validIdList = parameters['trainIdList'],parameters['validIdList']
            dataClass.seqItem2id,dataClass.id2seqItem = parameters['seqItem2id'],parameters['id2seqItem']
            dataClass.secItem2id,dataClass.id2secItem = parameters['secItem2id'],parameters['id2secItem']
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc']))
    def calculate_y_prob(self, X):
        Y_pre = self.calculate_y_logit(X)
        return torch.softmax(Y_pre, dim=1)
    def calculate_y(self, X):
        Y_pre = self.calculate_y_prob(X)
        return torch.argmax(Y_pre, dim=1)
    def calculate_loss(self, X, Y):
        Y_logit = self.calculate_y_logit(X)
        return self.criterion(Y_logit, Y)
    def calculate_indicator_by_iterator(self, dataStream, classNum, report):
        metrictor = Metrictor(classNum)
        Y_prob_pre,Y = self.calculate_y_prob_by_iterator(dataStream)
        metrictor.set_data(Y_prob_pre, Y)
        return metrictor(report)
    def calculate_y_prob_by_iterator(self, dataStream):
        YArr,Y_preArr = [],[]
        while True:
            try:
                X,Y = next(dataStream)
            except:
                break
            Y_pre,Y = self.calculate_y_prob(X).cpu().data.numpy(),Y.cpu().data.numpy()
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.hstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        return Y_preArr, YArr
    def calculate_y_by_iterator(self, dataStream):
        Y_preArr, YArr = self.calculate_y_prob_by_iterator(dataStream)
        return Y_preArr.argmax(axis=1), YArr
    def to_train_mode(self):
        for module in self.moduleList:
            module.train()
    def to_eval_mode(self):
        for module in self.moduleList:
            module.eval()
    def _train_step(self, X, Y, optimizer):
        self.stepCounter += 1
        if self.stepCounter<self.stepUpdate:
            p = False
        else:
            self.stepCounter = 0
            p = True
        loss = self.calculate_loss(X, Y)/self.stepUpdate
        loss.backward()
        if p:
            optimizer.step()
            optimizer.zero_grad()
        return loss*self.stepUpdate
    
##第二种蒸馏
class BaseClassifier:
    def __init__(self):
        pass
    def calculate_y_logit(self, X, XLen):
        pass
    def cv_train(self, dataClass, trainSize=256, batchSize=256, epoch=100, stopRounds=10, earlyStop=10, saveRounds=1, augmentation=0.05,
                 optimType='Adam', lr=0.001, weightDecay=0, kFold=5, isHigherBetter=True, metrics="Score", report=["ACC", "MaF", "Score"],
                 savePath='model'):
        kf = KFold(n_splits=kFold)
        validRes = []
        for i,(trainIndices,validIndices) in enumerate(kf.split(range(dataClass.totalSampleNum))):
            print(f'CV_{i+1}:')
            self.reset_parameters()
            dataClass.trainIdList,dataClass.validIdList = trainIndices,validIndices
            dataClass.trainSampleNum,self.validSampleNum = len(trainIndices),len(validIndices)
            dataClass.describe()
            res = self.train(dataClass,trainSize,batchSize,epoch,stopRounds,earlyStop,saveRounds,augmentation,optimType,lr,weightDecay,
                             isHigherBetter,metrics,report,f"{savePath}_cv{i+1}")
            validRes.append(res)
        Metrictor.table_show(validRes, report)
    def train(self, dataClass, trainSize=256, batchSize=256, epoch=100, stopRounds=10, earlyStop=10, saveRounds=1, augmentation=0.05,
              optimType='Adam', lr=0.001, weightDecay=0, isHigherBetter=True, metrics="Score", report=["ACC", "MaF", "Score"],
              savePath='model'):
        assert batchSize%trainSize==0
        metrictor = Metrictor(dataClass.classNum)
        self.stepCounter = 0
        self.stepUpdate = batchSize//trainSize
        optimizer = torch.optim.Adam(self.moduleList.parameters(), lr=lr, weight_decay=weightDecay)
        trainStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='train', device=self.device, augmentation=augmentation)
        itersPerEpoch = (dataClass.trainSampleNum+trainSize-1)//trainSize
        mtc,bestMtc,stopSteps = 0.0,0.0,0
        if dataClass.validSampleNum>0: validStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='valid', device=self.device, augmentation=augmentation)
        st = time.time()
        for e in range(epoch):
            for i in range(itersPerEpoch):
                self.to_train_mode()
                X,Y = next(trainStream)
                # 蒸馏
                student_loss = self.calculate_loss(X,Y)
                # 计算教师模型的loss
                student_preds = self.calculate_y_logit(X)
                teacher_preds=next(data_iter)
                distillation_loss = soft_loss(
                F.softmax(student_preds/temp, dim=1),
                F.softmax(teacher_preds/temp, dim=1))
                # 综合教师和学生损失来进行反向传播和优化
                optimizer.zero_grad()
                total_loss = alpha * teacher_loss + (1 - alpha) * student_loss
                total_loss.backward()
                optimizer.step()
                if stopRounds>0 and (e*itersPerEpoch+i+1)%stopRounds==0:
                    self.to_eval_mode()
                    print(f"After iters {e*itersPerEpoch+i+1}: [train] loss= {loss:.3f};", end='')
                    if dataClass.validSampleNum>0:
                        X,Y = next(validStream)
                        loss = self.calculate_loss(X,Y)
                        print(f' [valid] loss= {loss:.3f};', end='')
                    restNum = ((itersPerEpoch-i-1)+(epoch-e-1)*itersPerEpoch)*trainSize
                    speed = (e*itersPerEpoch+i+1)*trainSize/(time.time()-st)
                    print(" speed: %.3lf items/s; remaining time: %.3lfs;"%(speed, restNum/speed))
            if dataClass.validSampleNum>0 and (e+1)%saveRounds==0:
                self.to_eval_mode()
                print(f'========== Epoch:{e+1:5d} ==========')
                #Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='train', device=self.device))
                #metrictor.set_data(Y_pre, Y)
                #print(f'[Total Train]',end='')
                #metrictor(report)
                print(f'[Total Valid]',end='')
                Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='valid', device=self.device))
                metrictor.set_data(Y_pre, Y)
                res = metrictor(report)
                mtc = res[metrics]
                print('=================================')
                if (mtc>bestMtc and isHigherBetter) or (mtc<bestMtc and not isHigherBetter):
                    print(f'Bingo!!! Get a better Model with val {metrics}: {mtc:.3f}!!!')
                    bestMtc = mtc
                    self.save("%s.pkl"%savePath, e+1, bestMtc, dataClass)
                    stopSteps = 0
                else:
                    stopSteps += 1
                    if stopSteps>=earlyStop:
                        print(f'The val {metrics} has not improved for more than {earlyStop} steps in epoch {e+1}, stop training.')
                        break
        self.load("%s.pkl"%savePath)
        os.rename("%s.pkl"%savePath, "%s_%s.pkl"%(savePath, ("%.3lf"%bestMtc)[2:]))
        print(f'============ Result ============')
        Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='train', device=self.device))
        metrictor.set_data(Y_pre, Y)
        print(f'[Total Train]',end='')
        metrictor(report)
        Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='valid', device=self.device))
        metrictor.set_data(Y_pre, Y)
        print(f'[Total Valid]',end='')
        res = metrictor(report)
        metrictor.each_class_indictor_show(dataClass.id2secItem)
        print(f'================================')
        return res
    def reset_parameters(self):
        for module in self.moduleList:
            for subModule in module.modules():
                if hasattr(subModule, "reset_parameters"):
                    subModule.reset_parameters()
    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc}
        for module in self.moduleList:
            stateDict[module.name] = module.state_dict()
        if dataClass is not None:
            stateDict['trainIdList'],stateDict['validIdList'] = dataClass.trainIdList,dataClass.validIdList
            stateDict['seqItem2id'],stateDict['id2seqItem'] = dataClass.seqItem2id,dataClass.id2seqItem
            stateDict['secItem2id'],stateDict['id2secItem'] = dataClass.secItem2id,dataClass.id2secItem
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.moduleList:
            module.load_state_dict(parameters[module.name])
        if dataClass is not None:
            dataClass.trainIdList,dataClass.validIdList = parameters['trainIdList'],parameters['validIdList']
            dataClass.seqItem2id,dataClass.id2seqItem = parameters['seqItem2id'],parameters['id2seqItem']
            dataClass.secItem2id,dataClass.id2secItem = parameters['secItem2id'],parameters['id2secItem']
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc']))
    def calculate_y_prob(self, X):
        Y_pre = self.calculate_y_logit(X)
        return torch.softmax(Y_pre, dim=1)
    def calculate_y(self, X):
        Y_pre = self.calculate_y_prob(X)
        return torch.argmax(Y_pre, dim=1)
    def calculate_loss(self, X, Y):
        Y_logit = self.calculate_y_logit(X)
        return self.criterion(Y_logit, Y)
    def calculate_indicator_by_iterator(self, dataStream, classNum, report):
        metrictor = Metrictor(classNum)
        Y_prob_pre,Y = self.calculate_y_prob_by_iterator(dataStream)
        metrictor.set_data(Y_prob_pre, Y)
        return metrictor(report)
    def calculate_y_prob_by_iterator(self, dataStream):
        YArr,Y_preArr = [],[]
        while True:
            try:
                X,Y = next(dataStream)
            except:
                break
            Y_pre,Y = self.calculate_y_prob(X).cpu().data.numpy(),Y.cpu().data.numpy()
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.hstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        return Y_preArr, YArr
    def calculate_y_by_iterator(self, dataStream):
        Y_preArr, YArr = self.calculate_y_prob_by_iterator(dataStream)
        return Y_preArr.argmax(axis=1), YArr
    def to_train_mode(self):
        for module in self.moduleList:
            module.train()
    def to_eval_mode(self):
        for module in self.moduleList:
            module.eval()
    def _train_step(self, X, Y, optimizer):
        self.stepCounter += 1
        if self.stepCounter<self.stepUpdate:
            p = False
        else:
            self.stepCounter = 0
            p = True
        loss = self.calculate_loss(X, Y)/self.stepUpdate
        loss.backward()
        if p:
            optimizer.step()
            optimizer.zero_grad()
        return loss*self.stepUpdate

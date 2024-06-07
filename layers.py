from torch import nn as nn
from torch.nn import functional as F
import torch,time,os
import numpy as np

class TextEmbedding(nn.Module):
    def __init__(self, embedding, dropout=0.3, freeze=False, name='textEmbedding'):
        super(TextEmbedding, self).__init__()
        self.name = name
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=freeze)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        # x: batchSize × seqLen
        return self.dropout(self.embedding(x))



############多尺度
class TextMSTCN(nn.Module):
    def __init__(self, feaSize, contextSizeList, filterNum, name='textMSTCN'):
        super(TextMSTCN, self).__init__()
        self.name = name
        moduleList = []
        # Define ResBlock
        self.ResBlock = nn.Sequential(
            nn.Conv1d(in_channels=feaSize, out_channels=feaSize * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(feaSize * 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=feaSize * 2, out_channels=feaSize * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(feaSize * 4),
            nn.ReLU(),
            nn.Conv1d(in_channels=feaSize * 4, out_channels=filterNum, kernel_size=3, padding=1),  # Change output channels to scaleNum * filterNum
            nn.BatchNorm1d(filterNum),
            nn.ReLU(),
        )

        # Create TCN modules
        for i in range(len(contextSizeList)):
            moduleList.append(
                nn.Sequential(
                    self.ResBlock,
                ))
        self.tcnList = nn.ModuleList(moduleList)
    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = x.transpose(1,2) # => batchSize × feaSize × seqLen
        x = [conv(x) for conv in self.tcnList] # => scaleNum * (batchSize × filterNum × seqLen)
        return torch.cat(x, dim=1).transpose(1,2) # => batchSize × seqLen × scaleNum*filterNum
    



########################双向实现
class TextTCN(nn.Module):
    def __init__(self, feaSize, contextSizeList, filterNum, name='textTCN'):
        super(TextTCN, self).__init__()
        self.name = name
        self.num_scales = len(contextSizeList)

        # Define ResBlock
        self.ResBlock = nn.Sequential(
            nn.Conv1d(in_channels=feaSize, out_channels=feaSize * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(feaSize * 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=feaSize * 2, out_channels=feaSize * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(feaSize * 4),
            nn.ReLU(),
            nn.Conv1d(in_channels=feaSize * 4, out_channels=filterNum, kernel_size=3, padding=1),
            nn.BatchNorm1d(filterNum),
            nn.ReLU(),
        )

        # Create forward TCN modules
        forward_module_list = []
        for i in range(len(contextSizeList)):
            forward_module_list.append(
                nn.Sequential(
                    self.ResBlock,
                ))
        self.forward_tcnList = nn.ModuleList(forward_module_list)

        # Create backward TCN modules
        backward_module_list = []
        for i in range(len(contextSizeList)):
            backward_module_list.append(
                nn.Sequential(
                    self.ResBlock,
                ))
        self.backward_tcnList = nn.ModuleList(backward_module_list)

        self.conv1x1 = nn.Conv1d(in_channels=filterNum * 2, out_channels=filterNum*self.num_scales, kernel_size=1)

    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = x.transpose(1, 2)  # => batchSize × feaSize × seqLen

        # Forward pass
        forward_outputs = [conv(x) for conv in self.forward_tcnList]  # => scaleNum * (batchSize × filterNum × seqLen)
        forward_output = torch.mean(torch.stack(forward_outputs), dim=0)

        # Backward pass
        backward_x = torch.flip(x, dims=[2])  # Reverse the input along the seqLen dimension
        backward_outputs = [conv(backward_x) for conv in self.backward_tcnList]  # => scaleNum * (batchSize × filterNum × seqLen)
        backward_output = torch.mean(torch.stack(backward_outputs), dim=0)

        # Concatenate forward and backward outputs
        output = torch.cat([forward_output, backward_output], dim=1)

        # Apply 1x1 convolution for output dimension consistency

        output = self.conv1x1(output)  # => batchSize × seqLen × filterNum
        output = output.transpose(1, 2)  # => batchSize × seqLen × (2 * filterNum)
        return output



##BiGRU
class TextBiGRU(nn.Module):
    def __init__(self, feaSize, hiddenSize, num_layers=1, dropout=0.0, name='textBiGRU'):
        super(TextBiGRU, self).__init__()
        self.name = name
        self.biGRU = nn.GRU(feaSize, hiddenSize, bidirectional=True, batch_first=True, num_layers=num_layers, dropout=dropout)
    def forward(self, x, xlen=None):
        # x: batchSizeh × seqLen × feaSize
        if xlen is not None:
            xlen, indices = torch.sort(xlen, descending=True)
            _, desortedIndices = torch.sort(indices, descending=False)

            x = nn.utils.rnn.pack_padded_sequence(x[indices], xlen, batch_first=True)
        output, hn = self.biGRU(x) # output: batchSize × seqLen × hiddenSize*2; hn: numLayers*2 × batchSize × hiddenSize
        if xlen is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            return output[desortedIndices]
        return output # output: batchSize × seqLen × hiddenSize*2
##BiGRU-LSTM


class TextBiGRULSTM(nn.Module):
    def __init__(self, feaSize, hiddenSize, num_layers=1, dropout=0.0, name='textBiGRU'):
        super(TextBiGRULSTM, self).__init__()
        self.name = name
        self.biGRU = nn.GRU(feaSize, hiddenSize, bidirectional=True, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.biLSTM = nn.LSTM(hiddenSize * 2, hiddenSize, bidirectional=True, batch_first=True)
        self.biGRU = nn.GRU(feaSize, hiddenSize, bidirectional=True, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.biLSTM = nn.LSTM(hiddenSize * 2, hiddenSize, bidirectional=True, batch_first=True)
        self.biGRU = nn.GRU(feaSize, hiddenSize, bidirectional=True, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.biLSTM = nn.LSTM(hiddenSize * 2, hiddenSize, bidirectional=True, batch_first=True)


    def forward(self, x, xlen=None):
        # x: batchSize × seqLen × feaSize
        if xlen is not None:
            xlen, indices = torch.sort(xlen, descending=True)
            _, desortedIndices = torch.sort(indices, descending=False)

            x = nn.utils.rnn.pack_padded_sequence(x[indices], xlen, batch_first=True)

        # Pass through BiGRU
        output, hn = self.biGRU(x)  # output: batchSize × seqLen × hiddenSize*2; hn: numLayers*2 × batchSize × hiddenSize

        # Pass through BiLSTM
        output, _ = self.biLSTM(output)  # output: batchSize × seqLen × hiddenSize*2

        if xlen is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            return output[desortedIndices]

        return output  # output: batchSize × seqLen × hiddenSize*2




class LinearRelu(nn.Module):
    def __init__(self, inSize, outSize, name='linearRelu'):
        super(LinearRelu, self).__init__()
        self.name = name
        self.layer = nn.Sequential(
                            nn.ReLU(),
                            nn.Linear(inSize, outSize),
                            nn.ReLU()
                         )
    def forward(self, x):
        return self.layer(x)

class MLP(nn.Module):
    def __init__(self, inSize, outSize, hiddenList=[], dropout=0.1, name='MLP', actFunc=nn.ReLU):
        super(MLP, self).__init__()
        self.name = name
        layers = nn.Sequential()
        for i,os in enumerate(hiddenList):
            layers.add_module(str(i*2), nn.Linear(inSize, os))
            layers.add_module(str(i*2+1), actFunc())
            inSize = os
        self.hiddenLayers = layers
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(inSize, outSize)
    def forward(self, x):
        x = self.hiddenLayers(x)
        return self.out(self.dropout(x))

##MLP_Mixer
class MLP_Mixer(nn.Module):
    def __init__(self, inSize, outSize, hiddenList=[], dropout=0.1, name='MLP-Mixer', actFunc=nn.GELU):
        super(MLP_Mixer, self).__init__()
        self.name = name
        layers = nn.Sequential()
        for i, os in enumerate(hiddenList):
            # Mixer Token-Mixing Layer
            layers.add_module(f'token_mixing_{i}', nn.Sequential(
                nn.Linear(inSize, os),
                actFunc(),
                nn.Linear(os, os),
            ))

            # Mixer Channel-Mixing Layer
            layers.add_module(f'channel_mixing_{i}', nn.Sequential(
                nn.LayerNorm(os),
                nn.Linear(os, os),
                actFunc(),
            ))
            
            inSize = os

        self.mixer_layers = layers
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(inSize, outSize)

    def forward(self, x):
        x = self.mixer_layers(x)
        return self.out(self.dropout(x))

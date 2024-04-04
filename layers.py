from utils import *

class TextBiLSTM(nn.Module):
    def __init__(self, feaSize, hiddenSize, num_layers=1, dropout=0.0, name='textBiLSTM'):
        super(TextBiLSTM, self).__init__()
        self.name = name
        # 直接初始化双向LSTM层，输入特征大小为feaSize
        self.biLSTM = nn.LSTM(feaSize, hiddenSize, bidirectional=True, batch_first=True, num_layers=num_layers, dropout=dropout)

    def forward(self, x, xlen=None):
        # x: batchSize × seqLen × feaSize
        if xlen is not None:
            # 如果提供了序列长度，对序列进行排序和打包
            xlen, indices = torch.sort(xlen, descending=True)
            _, desortedIndices = torch.sort(indices, descending=False)
            x = nn.utils.rnn.pack_padded_sequence(x[indices], xlen, batch_first=True)

        # 直接将输入传递给双向LSTM层
        output, _ = self.biLSTM(x)  # output: batchSize × seqLen × hiddenSize*2

        if xlen is not None:
            # 如果序列被打包，解包输出并恢复原始顺序
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            output = output[desortedIndices]

        return output  # output: batchSize × seqLen × hiddenSize*2
    

import math
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by the number of heads."

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query_transform = nn.Linear(hidden_size, hidden_size)
        self.key_transform = nn.Linear(hidden_size, hidden_size)
        self.value_transform = nn.Linear(hidden_size, hidden_size)

        self.fc_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, outputs):
        batch_size = outputs.size(0)

        # Transform and split the queries, keys, and values
        queries = self.query_transform(outputs).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key_transform(outputs).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value_transform(outputs).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute the attention scores
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = F.softmax(energy, dim=-1)

        # Apply the attention weights to the values
        x = torch.matmul(attention, values)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        # Final linear transformation
        x = self.fc_out(x)

        return x



class TextBiLSTMWithAttention(nn.Module):
    def __init__(self, feaSize, hiddenSize, num_heads, num_layers=1, dropout=0.0, name='textBiLSTMWithAttention'):
        super(TextBiLSTMWithAttention, self).__init__()
        self.name = name
        self.biLSTM = nn.LSTM(feaSize, hiddenSize, bidirectional=True, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.multihead_attention = MultiHeadAttention(hiddenSize * 2, num_heads)  # 使用隐藏层大小的两倍，因为是双向的

    def forward(self, x, xlen=None):
        if xlen is not None:
            xlen, indices = torch.sort(xlen, descending=True)
            _, desortedIndices = torch.sort(indices, descending=False)
            x = nn.utils.rnn.pack_padded_sequence(x[indices], xlen, batch_first=True)

        output, _ = self.biLSTM(x)  # output: batchSize × seqLen × (hiddenSize * 2)

        if xlen is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            output = output[desortedIndices]

        # 应用多头注意力机制
        output = self.multihead_attention(output)  # 注意这里我们只取了输出，没有取权重

        return output  # output: batchSize × seqLen × (hiddenSize * 2)


class TextTCN1(nn.Module):
    def __init__(self, feaSize, contextSizeList, filterNum, name='textTCN'):
        super(TextTCN, self).__init__()
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
    
class TextTCN2(nn.Module):
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
#1 Write a ai program to make better ai 
#창조 이포크 찾기 및 코드 생성 모델
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torch.utils.data import DataLoader, Dataset
from torchtext.legacy import data
from torchtext.legacy.data import BucketIterator, Field, Iterator
import nltk
import re
import spacy
import tensorflow as tf
import tensorflow_datasets as tfds
import datetime
import gradio as gr
from translate import Translator
import sys
import json
import math
import os
import urllib.request
import webbrowser
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import wikipedia
import wolframalpha
from nltk.stem.porter import PorterStemmer
import csv
#import evaluate
start = time.time()
def csv_writer2(time, name_list):
        with open('know_synapse.csv', mode='a', newline='', encoding='utf-8') as RESULT_writer_file:
            RESULT_writer = csv.writer(RESULT_writer_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            '''RESULT_writer.writerow(
                ["file name", "loss", "EPOCH", "working time"])'''
            for row in name_list: # 위의 name_list를 순차적으로 순회
                RESULT_writer.writerow([row[0],row[1],row[2],row[3]]) # 각 행을 순차적으로 .csv 파일에 저장
now = datetime.datetime.now()

f = open("improve.txt", "r", encoding='utf-8')
file_lines = f.readlines()


file_lines[:20]

dps = []
dp = None
for line in file_lines:
  if line[0] == "#":
    if dp:
      dp['solution'] = ''.join(dp['solution'])
      dps.append(dp)
    dp = {"question": None, "solution": []}
    dp['question'] = line[1:]
  else:
    dp["solution"].append(line)
i=0
for dp in dps:
  print("\n Question no: ", i+1)
  i+=1
  print(dp['question'][1:])
  print(dp['solution'])
  if i>49:
    break
print("Dataset size:", len(dps))
import io
from tokenize import tokenize, untokenize


def tokenize_python_code(python_code_str):
    python_tokens = list(tokenize(io.BytesIO(python_code_str.encode('utf-8')).readline))
    tokenized_output = []
    for i in range(0, len(python_tokens)):
        tokenized_output.append((python_tokens[i].type, python_tokens[i].string))
    return tokenized_output
tokenized_sample = tokenize_python_code(dps[1]['solution'])
print(tokenized_sample)
print(untokenize(tokenized_sample).decode('utf-8'))
import keyword

print(keyword.kwlist)
def augment_tokenize_python_code(python_code_str, mask_factor=0.3):
    

    var_dict = {} # Dictionary that stores masked variables

    # certain reserved words that should not be treated as normal variables and
    # hence need to be skipped from our variable mask augmentations
    skip_list = ['range', 'enumerate', 'print', 'ord', 'int', 'float', 'zip'
                 'char', 'list', 'dict', 'tuple', 'set', 'len', 'sum', 'min', 'max']
    skip_list.extend(keyword.kwlist)

    var_counter = 1
    python_tokens = list(tokenize(io.BytesIO(python_code_str.encode('utf-8')).readline))
    tokenized_output = []

    for i in range(0, len(python_tokens)):
      if python_tokens[i].type == 1 and python_tokens[i].string not in skip_list:
        
        if i>0 and python_tokens[i-1].string in ['def', '.', 'import', 'raise', 'except', 'class']: # avoid masking modules, functions and error literals
          skip_list.append(python_tokens[i].string)
          tokenized_output.append((python_tokens[i].type, python_tokens[i].string))
        elif python_tokens[i].string in var_dict:  # if variable is already masked
          tokenized_output.append((python_tokens[i].type, var_dict[python_tokens[i].string]))
        elif random.uniform(0, 1) > 1-mask_factor: # randomly mask variables
          var_dict[python_tokens[i].string] = 'var_' + str(var_counter)
          var_counter+=1
          tokenized_output.append((python_tokens[i].type, var_dict[python_tokens[i].string]))
        else:
          skip_list.append(python_tokens[i].string)
          tokenized_output.append((python_tokens[i].type, python_tokens[i].string))
      
      else:
        tokenized_output.append((python_tokens[i].type, python_tokens[i].string))
    
    return tokenized_output
tokenized_sample = augment_tokenize_python_code(dps[1]['solution'])
print(tokenized_sample)

python_problems_df = pd.DataFrame(dps)

python_problems_df.head()

python_problems_df.shape

import numpy as np

np.random.seed(0)
msk = np.random.rand(len(python_problems_df)) < 0.85 # Splitting data into 85% train and 15% validation

train_df = python_problems_df[msk]
val_df = python_problems_df[~msk]
train_df.shape
val_df.shape
SEED = 1234

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
Input = data.Field(tokenize = 'spacy',
            init_token='<sos>', 
            eos_token='<eos>', 
            lower=True)

Output = data.Field(tokenize = augment_tokenize_python_code,
                    init_token='<sos>', 
                    eos_token='<eos>', 
                    lower=False)
fields = [('Input', Input),('Output', Output)]
train_example = []
val_example = []

train_expansion_factor = 100
for j in range(train_expansion_factor):
  for i in range(train_df.shape[0]):
      try:
          ex = data.Example.fromlist([train_df.question[i], train_df.solution[i]], fields)
          train_example.append(ex)
      except:
          pass

for i in range(val_df.shape[0]):
    try:
        ex = data.Example.fromlist([val_df.question[i], val_df.solution[i]], fields)
        val_example.append(ex)
    except:
        pass       

train_data = data.Dataset(train_example, fields)
valid_data =  data.Dataset(val_example, fields)
Input.build_vocab(train_data, min_freq = 0)
Output.build_vocab(train_data, min_freq = 0)
Output.vocab
def save_vocab(vocab, path):
    import pickle
    output = open(path, 'wb')
    pickle.dump(vocab, output)
    output.close()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
train_data[0].Output

print(vars(train_data.examples[1]))
class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = 1000):
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, src len]
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        
        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        #src = [batch size, src len, hid dim]
            
        return src
class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
                
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention
class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = 10000):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        #pos = [batch size, trg len]

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
                
        #trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        output = self.fc_out(trg)
        
        #output = [batch size, trg len, output dim]
            
        return output, attention
class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
            
        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        # query, key, value
        
        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        #trg = [batch size, trg len, hid dim]
        
        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return trg, attention
class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention
INPUT_DIM = len(Input.vocab)
OUTPUT_DIM = len(Output.vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 16
DEC_HEADS = 16
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)

dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)
len(Output.vocab.__dict__['freqs'])
SRC_PAD_IDX = Input.vocab.stoi[Input.pad_token]
TRG_PAD_IDX = Output.vocab.stoi[Output.pad_token]

model3 = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

def count_parameters(model3):
    return sum(p.numel() for p in model3.parameters() if p.requires_grad)

print(f'The model3 has {count_parameters(model3):,} trainable parameters')
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
model3.apply(initialize_weights);

LEARNING_RATE = 0.0005

optimizer = torch.optim.Adam(model3.parameters(), lr = LEARNING_RATE)


import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """CrossEntropyLoss - with ability to recieve distrbution as targets, and optional label smoothing"""

    def __init__(self, weight=None, ignore_index=-100, reduction='mean', smooth_eps=None, smooth_dist=None, from_logits=True):
        super(CrossEntropyLoss, self).__init__(weight=weight,
                                               ignore_index=ignore_index, reduction=reduction)
        self.smooth_eps = smooth_eps
        self.smooth_dist = smooth_dist
        self.from_logits = from_logits

    def forward(self, input, target, smooth_dist=None):
        if smooth_dist is None:
            smooth_dist = self.smooth_dist
        return cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index,
                             reduction=self.reduction, smooth_eps=self.smooth_eps,
                             smooth_dist=smooth_dist, from_logits=self.from_logits)


def cross_entropy(inputs, target, weight=None, ignore_index=-100, reduction='mean',
                  smooth_eps=None, smooth_dist=None, from_logits=True):
    """cross entropy loss, with support for target distributions and label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0

    # ordinary log-liklihood - use cross_entropy from nn
    if _is_long(target) and smooth_eps == 0:
        if from_logits:
            return F.cross_entropy(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)
        else:
            return F.nll_loss(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)

    if from_logits:
        # log-softmax of inputs
        lsm = F.log_softmax(inputs, dim=-1)
    else:
        lsm = inputs

    masked_indices = None
    num_classes = inputs.size(-1)

    if _is_long(target) and ignore_index >= 0:
        masked_indices = target.eq(ignore_index)

    if smooth_eps > 0 and smooth_dist is not None:
        if _is_long(target):
            target = onehot(target, num_classes).type_as(inputs)
        if smooth_dist.dim() < target.dim():
            smooth_dist = smooth_dist.unsqueeze(0)
        target.lerp_(smooth_dist, smooth_eps)

    if weight is not None:
        lsm = lsm * weight.unsqueeze(0)

    if _is_long(target):
        eps_sum = smooth_eps / num_classes
        eps_nll = 1. - eps_sum - smooth_eps
        likelihood = lsm.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        loss = -(eps_nll * likelihood + eps_sum * lsm.sum(-1))
    else:
        loss = -(target * lsm).sum(-1)

    if masked_indices is not None:
        loss.masked_fill_(masked_indices, 0)

    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        if masked_indices is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / float(loss.size(0) - masked_indices.sum())

    return loss


def onehot(indexes, N=None, ignore_index=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    ignore_index will be zero in onehot representation
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().byte().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    if ignore_index is not None and ignore_index >= 0:
        output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
    return output

def _is_long(x):
    if hasattr(x, 'data'):
        x = x.data
    return isinstance(x, torch.LongTensor) or isinstance(x, torch.cuda.LongTensor)
def maskNLLLoss(inp, target, mask):
    # print(inp.shape, target.shape, mask.sum())
    nTotal = mask.sum()
    crossEntropy = CrossEntropyLoss(ignore_index = TRG_PAD_IDX, smooth_eps=0.20)
    loss = crossEntropy(inp, target)
    loss = loss.to(device)
    return loss, nTotal.item()

criterion = maskNLLLoss

from tqdm import tqdm


def make_trg_mask(trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != TRG_PAD_IDX).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

def train(model3, iterator, optimizer, criterion, clip):
    
    model3.train()
    
    n_totals = 0
    print_losses = []
    for i, batch in tqdm(enumerate(iterator), total=len(iterator)):
        # print(batch)
        loss = 0
        src = batch.Input.permute(1, 0)
        trg = batch.Output.permute(1, 0)
        trg_mask = make_trg_mask(trg)
        optimizer.zero_grad()
        
        output, _ = model3(src, trg[:,:-1])
                
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
            
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
            
        mask_loss, nTotal = criterion(output, trg, trg_mask)
        
        mask_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model3.parameters(), clip)
        
        optimizer.step()
        
        print_losses.append(mask_loss.item() * nTotal)
        n_totals += nTotal


        
    return sum(print_losses) / n_totals

def evaluate(model3, iterator, criterion):
    
    model3.eval()
    
    n_totals = 0
    print_losses = []
    
    with torch.no_grad():
    
        for i, batch in tqdm(enumerate(iterator), total=len(iterator)):

            src = batch.Input.permute(1, 0)
            trg = batch.Output.permute(1, 0)
            trg_mask = make_trg_mask(trg)

            output, _ = model3(src, trg[:,:-1])
            
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]
            
            mask_loss, nTotal = criterion(output, trg, trg_mask)

            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

        
    return sum(print_losses) / n_totals

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 1
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    import time
    start_time = time.time()
    
    train_example = []
    val_example = []

    for i in range(train_df.shape[0]):
        try:
            ex = data.Example.fromlist([train_df.question[i], train_df.solution[i]], fields)
            train_example.append(ex)
        except:
            pass

    for i in range(val_df.shape[0]):
        try:
            ex = data.Example.fromlist([val_df.question[i], val_df.solution[i]], fields)
            val_example.append(ex)
        except:
            pass       

    train_data = data.Dataset(train_example, fields)
    valid_data =  data.Dataset(val_example, fields)

    BATCH_SIZE = 16
    train_iterator, valid_iterator = BucketIterator.splits((train_data, valid_data), batch_size = BATCH_SIZE, 
                                                                sort_key = lambda x: len(x.Input),
                                                                sort_within_batch=True, device = device)

    train_loss = train(model3, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model3, valid_iterator, criterion)
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    #valid_loss 가 평가 메트릭으로 이건 성능을 평가한 정확률을 보여줄것이므로 이걸로 비교도 하게하여 개선할것
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model3.state_dict(), 'improve_py.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.0f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.0f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    f = open('loss.txt', 'w', encoding = 'utf-8')
    f.write(f'{valid_loss:.0f}')
    time = f'{epoch_mins}{epoch_secs}'

    

SRC = Input
TRG = Output
def translate_sentence(sentence, src_field, trg_field, model3, device, max_len = 50000):
    
    model3.eval()
        
    if isinstance(sentence, str):
        nlp = spacy.load('en')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    src_mask = model3.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model3.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model3.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model3.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:], attention


model3.load_state_dict(torch.load('improve_py.pt'))
def improve_py(src):
  src=src.split(" ")
  translation, attention = translate_sentence(src, SRC, TRG, model3, device)

  print(f'predicted trg: \n')
  # print(translation)
  print(untokenize(translation[:-1]).decode('utf-8'))

SRC = Input
TRG = Output


model3.load_state_dict(torch.load('improve_py.pt'))



f = open("program.txt", "r", encoding='utf-8')
file_lines = f.readlines()


file_lines[:20]

dps = []
dp = None
for line in file_lines:
  if line[0] == "#":
    if dp:
      dp['solution'] = ''.join(dp['solution'])
      dps.append(dp)
    dp = {"question": None, "solution": []}
    dp['question'] = line[1:]
  else:
    dp["solution"].append(line)
i=0
for dp in dps:
  print("\n Question no: ", i+1)
  i+=1
  print(dp['question'][1:])
  print(dp['solution'])
  if i>49:
    break
print("Dataset size:", len(dps))
from tokenize import tokenize, untokenize
import io


def tokenize_python_code(python_code_str):
    python_tokens = list(tokenize(io.BytesIO(python_code_str.encode('utf-8')).readline))
    tokenized_output = []
    for i in range(0, len(python_tokens)):
        tokenized_output.append((python_tokens[i].type, python_tokens[i].string))
    return tokenized_output
tokenized_sample = tokenize_python_code(dps[1]['solution'])
print(tokenized_sample)
print(untokenize(tokenized_sample).decode('utf-8'))
import keyword

print(keyword.kwlist)
def augment_tokenize_python_code(python_code_str, mask_factor=0.3):
    

    var_dict = {} # Dictionary that stores masked variables

    # certain reserved words that should not be treated as normal variables and
    # hence need to be skipped from our variable mask augmentations
    skip_list = ['range', 'enumerate', 'print', 'ord', 'int', 'float', 'zip'
                 'char', 'list', 'dict', 'tuple', 'set', 'len', 'sum', 'min', 'max']
    skip_list.extend(keyword.kwlist)

    var_counter = 1
    python_tokens = list(tokenize(io.BytesIO(python_code_str.encode('utf-8')).readline))
    tokenized_output = []

    for i in range(0, len(python_tokens)):
      if python_tokens[i].type == 1 and python_tokens[i].string not in skip_list:
        
        if i>0 and python_tokens[i-1].string in ['def', '.', 'import', 'raise', 'except', 'class']: # avoid masking modules, functions and error literals
          skip_list.append(python_tokens[i].string)
          tokenized_output.append((python_tokens[i].type, python_tokens[i].string))
        elif python_tokens[i].string in var_dict:  # if variable is already masked
          tokenized_output.append((python_tokens[i].type, var_dict[python_tokens[i].string]))
        elif random.uniform(0, 1) > 1-mask_factor: # randomly mask variables
          var_dict[python_tokens[i].string] = 'var_' + str(var_counter)
          var_counter+=1
          tokenized_output.append((python_tokens[i].type, var_dict[python_tokens[i].string]))
        else:
          skip_list.append(python_tokens[i].string)
          tokenized_output.append((python_tokens[i].type, python_tokens[i].string))
      
      else:
        tokenized_output.append((python_tokens[i].type, python_tokens[i].string))
    
    return tokenized_output
tokenized_sample = augment_tokenize_python_code(dps[1]['solution'])
print(tokenized_sample)

python_problems_df = pd.DataFrame(dps)

python_problems_df.head()

python_problems_df.shape

import numpy as np

np.random.seed(0)
msk = np.random.rand(len(python_problems_df)) < 0.85 # Splitting data into 85% train and 15% validation

train_df = python_problems_df[msk]
val_df = python_problems_df[~msk]
train_df.shape
val_df.shape
SEED = 1234

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
Input = data.Field(tokenize = 'spacy',
            init_token='<sos>', 
            eos_token='<eos>', 
            lower=True)

Output = data.Field(tokenize = augment_tokenize_python_code,
                    init_token='<sos>', 
                    eos_token='<eos>', 
                    lower=False)
fields = [('Input', Input),('Output', Output)]
train_example = []
val_example = []

train_expansion_factor = 100
for j in range(train_expansion_factor):
  for i in range(train_df.shape[0]):
      try:
          ex = data.Example.fromlist([train_df.question[i], train_df.solution[i]], fields)
          train_example.append(ex)
      except:
          pass

for i in range(val_df.shape[0]):
    try:
        ex = data.Example.fromlist([val_df.question[i], val_df.solution[i]], fields)
        val_example.append(ex)
    except:
        pass       

train_data = data.Dataset(train_example, fields)
valid_data =  data.Dataset(val_example, fields)
Input.build_vocab(train_data, min_freq = 0)
Output.build_vocab(train_data, min_freq = 0)
Output.vocab
def save_vocab(vocab, path):
    import pickle
    output = open(path, 'wb')
    pickle.dump(vocab, output)
    output.close()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
train_data[0].Output

print(vars(train_data.examples[1]))
class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = 1000):
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, src len]
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        
        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        #src = [batch size, src len, hid dim]
            
        return src
class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
                
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention
class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = 10000):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        #pos = [batch size, trg len]

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
                
        #trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        output = self.fc_out(trg)
        
        #output = [batch size, trg len, output dim]
            
        return output, attention
class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
            
        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        # query, key, value
        
        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        #trg = [batch size, trg len, hid dim]
        
        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return trg, attention
class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention
INPUT_DIM = len(Input.vocab)
OUTPUT_DIM = len(Output.vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 16
DEC_HEADS = 16
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)

dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)
len(Output.vocab.__dict__['freqs'])
SRC_PAD_IDX = Input.vocab.stoi[Input.pad_token]
TRG_PAD_IDX = Output.vocab.stoi[Output.pad_token]

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
model.apply(initialize_weights);

LEARNING_RATE = 0.0005
f = open('lr.txt', 'w', encoding='utf-8')
f.write(str(LEARNING_RATE))
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

f = open('optimizer.txt', 'w', encoding='utf-8')
f.write("Adam")#기본은 아담 이지만 개선할때 걸린 옵티마이저를 이다음 개선 됬을때부터는 사용한다

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.CrossEntropyLoss):
    """CrossEntropyLoss - with ability to recieve distrbution as targets, and optional label smoothing"""

    def __init__(self, weight=None, ignore_index=-100, reduction='mean', smooth_eps=None, smooth_dist=None, from_logits=True):
        super(CrossEntropyLoss, self).__init__(weight=weight,
                                               ignore_index=ignore_index, reduction=reduction)
        self.smooth_eps = smooth_eps
        self.smooth_dist = smooth_dist
        self.from_logits = from_logits

    def forward(self, input, target, smooth_dist=None):
        if smooth_dist is None:
            smooth_dist = self.smooth_dist
        return cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index,
                             reduction=self.reduction, smooth_eps=self.smooth_eps,
                             smooth_dist=smooth_dist, from_logits=self.from_logits)


def cross_entropy(inputs, target, weight=None, ignore_index=-100, reduction='mean',
                  smooth_eps=None, smooth_dist=None, from_logits=True):
    """cross entropy loss, with support for target distributions and label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0

    # ordinary log-liklihood - use cross_entropy from nn
    if _is_long(target) and smooth_eps == 0:
        if from_logits:
            return F.cross_entropy(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)
        else:
            return F.nll_loss(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)

    if from_logits:
        # log-softmax of inputs
        lsm = F.log_softmax(inputs, dim=-1)
    else:
        lsm = inputs

    masked_indices = None
    num_classes = inputs.size(-1)

    if _is_long(target) and ignore_index >= 0:
        masked_indices = target.eq(ignore_index)

    if smooth_eps > 0 and smooth_dist is not None:
        if _is_long(target):
            target = onehot(target, num_classes).type_as(inputs)
        if smooth_dist.dim() < target.dim():
            smooth_dist = smooth_dist.unsqueeze(0)
        target.lerp_(smooth_dist, smooth_eps)

    if weight is not None:
        lsm = lsm * weight.unsqueeze(0)

    if _is_long(target):
        eps_sum = smooth_eps / num_classes
        eps_nll = 1. - eps_sum - smooth_eps
        likelihood = lsm.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        loss = -(eps_nll * likelihood + eps_sum * lsm.sum(-1))
    else:
        loss = -(target * lsm).sum(-1)

    if masked_indices is not None:
        loss.masked_fill_(masked_indices, 0)

    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        if masked_indices is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / float(loss.size(0) - masked_indices.sum())

    return loss


def onehot(indexes, N=None, ignore_index=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    ignore_index will be zero in onehot representation
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().byte().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    if ignore_index is not None and ignore_index >= 0:
        output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
    return output

def _is_long(x):
    if hasattr(x, 'data'):
        x = x.data
    return isinstance(x, torch.LongTensor) or isinstance(x, torch.cuda.LongTensor)
def maskNLLLoss(inp, target, mask):
    # print(inp.shape, target.shape, mask.sum())
    nTotal = mask.sum()
    crossEntropy = CrossEntropyLoss(ignore_index = TRG_PAD_IDX, smooth_eps=0.20)
    loss = crossEntropy(inp, target)
    loss = loss.to(device)
    return loss, nTotal.item()

criterion = maskNLLLoss

from tqdm import tqdm

def make_trg_mask(trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != TRG_PAD_IDX).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    n_totals = 0
    print_losses = []
    for i, batch in tqdm(enumerate(iterator), total=len(iterator)):
        # print(batch)
        loss = 0
        src = batch.Input.permute(1, 0)
        trg = batch.Output.permute(1, 0)
        trg_mask = make_trg_mask(trg)
        optimizer.zero_grad()
        
        output, _ = model(src, trg[:,:-1])
                
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
            
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
            
        mask_loss, nTotal = criterion(output, trg, trg_mask)
        
        mask_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        print_losses.append(mask_loss.item() * nTotal)
        n_totals += nTotal


        
    return sum(print_losses) / n_totals

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    n_totals = 0
    print_losses = []
    
    with torch.no_grad():
    
        for i, batch in tqdm(enumerate(iterator), total=len(iterator)):

            src = batch.Input.permute(1, 0)
            trg = batch.Output.permute(1, 0)
            trg_mask = make_trg_mask(trg)

            output, _ = model(src, trg[:,:-1])
            
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]
            
            mask_loss, nTotal = criterion(output, trg, trg_mask)

            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

        
    return sum(print_losses) / n_totals

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def learn(N_EPOCHS):
    N_EPOCHS = N_EPOCHS
    CLIP = 1
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        import time
        start_time = time.time()
        
        train_example = []
        val_example = []

        for i in range(train_df.shape[0]):
            try:
                ex = data.Example.fromlist([train_df.question[i], train_df.solution[i]], fields)
                train_example.append(ex)
            except:
                pass

        for i in range(val_df.shape[0]):
            try:
                ex = data.Example.fromlist([val_df.question[i], val_df.solution[i]], fields)
                val_example.append(ex)
            except:
                pass       

        train_data = data.Dataset(train_example, fields)
        valid_data =  data.Dataset(val_example, fields)

        BATCH_SIZE = 1
        f = open('bs.txt', 'w', encoding='utf-8')
        f.write(str(BATCH_SIZE))
        train_iterator, valid_iterator = BucketIterator.splits((train_data, valid_data), batch_size = BATCH_SIZE, 
                                                                    sort_key = lambda x: len(x.Input),
                                                                    sort_within_batch=True, device = device)

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'code_create_model.pt')
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        f = open('loss.txt', 'w', encoding = 'utf-8')
        f.write(f'{valid_loss:.0f}')
learn(N_EPOCHS=1)

SRC = Input
TRG = Output
def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50000):
    
    model.eval()
        
    if isinstance(sentence, str):
        nlp = spacy.load('en')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:], attention

def listToString(str_list):
    
    result = ""

    for s in str_list:

        result += s + ""

    return result.strip()



now = datetime.datetime.now()
def csv_writer2(time, name_list):
    with open('judgment data.csv', mode='a', newline='', encoding='utf-8') as RESULT_writer_file:
        RESULT_writer = csv.writer(RESULT_writer_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        RESULT_writer.writerow(
            ['commend','code','correct','human judg','final judg','create correct'])
        for row in name_list: # 위의 name_list를 순차적으로 순회
            RESULT_writer.writerow([row[0],row[1],row[2],row[3]]) # 각 행을 순차적으로 .csv 파일에 저장

SRC = Input
TRG = Output


def code_create(src):
    
  src=src.split(" ")
  translation, attention = translate_sentence(src, SRC, TRG, model, device)

  print(f'predicted trg: \n')
  # print(translation)
  print(untokenize(translation[:-1]).decode('utf-8'))
  f = open('ai.txt', 'w', encoding='utf-8')

  f.write(untokenize(translation[:-1]).decode('utf-8'))


import gradio as gr
from translate import Translator
def listToString(str_list):
    
    result = ""

    for s in str_list:

        result += s + ""

    return result.strip()

def code_generation(text):
    from translate import Translator
    translator = Translator(from_lang="ko", to_lang="en")

    translation = translator.translate(text)

    src = translation

    
    code_create(src)

    f2 = open('ai.txt', 'r', encoding='utf-8')
    f3 = f2.readlines()

    f4 = listToString(f3)
    src2 = f4
    image = src2

    return image
    
    

    





def improve_self_model(text):
    f = open('feedback.txt', 'a', encoding = 'utf-8')
    
    from translate import Translator

    translator = Translator(from_lang="ko", to_lang="en")

    translation = translator.translate(text)

    translation
    f.write(translation)
    f.write('\n')
    
    return text, '\n', 'Thank you for your feedback'

f = open("n_epochs.txt", 'r', encoding='utf-8')
N_EPOCHS = listToString(f.readlines())
N_EPOCHS = int(N_EPOCHS)
model.load_state_dict(torch.load('code_create_model.pt'))
f = open("english_python_data (4).txt", "r", encoding='utf-8')
file_lines = f.readlines()

file_lines[:20]


dps = []
dp = None
for line in file_lines:
    if line[0] == "#":
        if dp:
            dp['solution'] = ''.join(dp['solution'])
            dps.append(dp)
        dp = {"question": None, "solution": []}
        dp['question'] = line[1:]
    else:
        dp["solution"].append(line)
i=0
for dp in dps:
    print("\n Question no: ", i+1)
    i+=1
    print(dp['question'][1:])
    print(dp['solution'])
    if i>49:
        break
print("Dataset size:", len(dps))
import io
from tokenize import tokenize, untokenize


def tokenize_python_code(python_code_str):
    python_tokens = list(tokenize(io.BytesIO(python_code_str.encode('utf-8')).readline))
    tokenized_output = []
    for i in range(0, len(python_tokens)):
        tokenized_output.append((python_tokens[i].type, python_tokens[i].string))
    return tokenized_output



python_problems_df = pd.DataFrame(dps)



import numpy as np

np.random.seed(0)
msk = np.random.rand(len(python_problems_df)) < 0.85 # Splitting data into 85% train and 15% validation

train_df = python_problems_df[msk]
question = train_df.question[0]




answer = train_df.solution[0]  
string3 = answer




while True:

    text = input("명령을 입력해주세요 : ")
    
        

    if text == "end" :
        break
        
        
    else:
        
    
        src = "Write a ai program to make better ai moudule import " 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'w',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')

        
        src = "Write a ai program to make better ai time caculate1" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai data load and preprocessing" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai data tokenize" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai dataframe generate, head, shape" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai numpy import" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai numpy random seed" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')

        src = "Write a ai program to make better ai msk variable generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai train_df variable generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai val_df variable generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai train_df variable shape" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai val_df variable shape" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai seed working" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai Input variable generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')

        src = "Write a ai program to make better ai Output variable generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai example list generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai train_data variable generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai valid_data variable generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai Input build_vocab" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai Output build_vocab" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')

        src = "Write a ai program to make better ai Output vocab" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai save_vocab fuction generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai device variable generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai train_data variable Output" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai train_data example print" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai Encoder class generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai EncoderLayer class generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')

        src = "Write a ai program to make better ai PositionwiseFeedforwardLayer class generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai MultiHeadAttentionLayer class generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai Decoder class generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai DecoderLayer class generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai Seq2Seq class generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai ready to model generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')

        src = "Write a ai program to make better ai model generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai initialize_weights function generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai model.apply" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai LEARNING_RATE variable generate, file save" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai optimizer variable generate, file save" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai CrossEntropyLoss class generate and cross_entropy working" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai criterion variable generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')

        src = "Write a ai program to make better ai tqdm moudule import " 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai make_trg_mask function generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai train function generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai evaluate function generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai epoch_time function generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai learn function generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')

        src = "Write a ai program to make better ai learn function execution" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai SRC, TRG variable generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai translate_sentence function generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai listToString function generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai now variable generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai csv_writer2 function generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')

        src = "Write a ai program to make better ai code_create function generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai gradio, translate moudule import" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai code_generation function generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai improve_self_model function generate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai n_epochs.txt file read and value load" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai code_create_model load" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')

        src = "Write a ai program to make better ai self model code data load and preprocessing" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai create a loop" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai get command input" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai if the command is end, stoping" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')
        src = "Write a ai program to make better ai otherwise code create and evaluate" 

        
        
        src1 = code_create(src)

        f = open('better_ai.py', 'a',encoding='utf-8')
        f.write(listToString(src1))
        f.write('\n')

        


        '''from improve_model import improve_model
        from check_model import check'''
        
        f2 = open('ai.txt', 'r', encoding='utf-8')

        f3 = f2.readlines()



        f4 = listToString(f3)

        src2 = f4
        code = src2
        print(code)
        src3 = len(code)
        print(src3)
        correct = ""
        human = ""
        correct = ""
        create = ""
        put = input("평가를 시작하시겠습니까? : ")
        if "start" in put:
            if src3 >= 7 :
                file_lines = listToString(file_lines)
                


                if code in file_lines:
                    print("correct program")
                    correct = "correct"
                    #correct = "wrong"
                    human = None
                    create = "wrong"
                    name_list = [[src, code, correct, human, correct, create]]
                    csv_writer2(now, name_list)

                else:
                    print("사용자의 추가확인(2차확인)이 필요합니다 현재 1차확인은 실패하였습니다")
                    check1, check2, check3, check4, check5 = input("데이터와 일치함, 맞춤법 올바름, 문법맞음, 띄어쓰기올바름, 명령취지맞음 : ").split(',')


                    if "정답" in check1:

                        correct = "correct"
                        human = "correct"
                        create = "wrong"
                        f = open("correct.txt", 'w', encoding='utf-8')
                        f.write(correct)
                        f.close()
                        f = open("human.txt", 'w', encoding='utf-8')
                        f.write(human)
                        f.close()
                        f = open("create.txt", 'w', encoding='utf-8')
                        f.write(create)
                        f.close()
                        name_list = [[src, code, correct, human, correct, create]]
                        csv_writer2(now, name_list)

                        #csv파일에 정답으로 입력한다(인간 판단 칸의 입력 및 최종 판단 칸에 입력)
                        print("최종 확인 결과 정답으로 판단되었습니다")
                    elif "오답" in check1:
                        #csv파일에 오답으로 입력한다(인간 판단 칸의 입력 및 최종 판단 칸에 입력)

                        f = open("check.py", "w", encoding='utf-8')
                        f.write(code)
                        f.close()
                        import time
                        time.sleep(30)
                        import check
                        # 나중에는 인공지능이 스스로 오류여부나 오답여부 등등을 인간에 판단데이터들을 바탕으로 직접 판단할수있게 만들기

                        correct = "wrong"
                        human = "wrong"
                        f = open("correct.txt", 'w', encoding='utf-8')
                        f.write(correct)
                        f.close()
                        f = open("human.txt", 'w', encoding='utf-8')
                        f.write(human)
                        f.close()
                        a = input("창조된 코드인가요? : ")
                        b = 0
                        if "yes" in a:
                            b += 1
                            b = str(b)
                            f = open('create_epoch' + b + '.txt', 'w', encoding='utf-8')
                            f.write(str(N_EPOCHS))
                            f.close()
                            create = 'correct'
                            name_list = [[src, code, correct, human, correct, create]]
                            csv_writer2(now, name_list)
                            f = open("create.txt", 'w', encoding='utf-8')
                            f.write(create)
                            f.close()
                            print("최종 확인 결과 오답으로 판단되었습니다")
                            print("오답인 코드이지만 창조된 코드입니다")
                            
                        elif "no" in a:
                            create = 'wrong'
                            name_list = [[src, code, correct, human, correct, create]]
                            csv_writer2(now, name_list)
                            f = open("create.txt", 'w', encoding='utf-8')
                            f.write(create)
                            f.close()
                            print("최종 확인 결과 오답으로 판단되었습니다")
                            print("오답인 코드의 오류 여부와 문제여부등을 확인하겠습니다(창조되지않음)")
                            
                    if "정답" in check2:
    
                        correct = "wrong"
                        human = "correct"
                        create = "wrong"
                        word = "correct"
                        
                        f = open("word.txt", 'w', encoding='utf-8')
                        f.write(word)
                        f.close()
                        #csv파일에 정답으로 입력한다(인간 판단 칸의 입력 및 최종 판단 칸에 입력)
                        
                    elif "오답" in check2:
                        word = "wrong"
                        f = open("word.txt", 'w', encoding='utf-8')
                        f.write(word)
                        f.close()
                    if "정답" in check3:
                        grammar = "correct"
                        
                        f = open("grammar.txt", 'w', encoding='utf-8')
                        f.write(grammar)
                        f.close()
                        #csv파일에 정답으로 입력한다(인간 판단 칸의 입력 및 최종 판단 칸에 입력)
                        
                    elif "오답" in check3:
                        grammar = "wrong"
                        f = open("grammar.txt", 'w', encoding='utf-8')
                        f.write(grammar)
                        f.close()
                    if "정답" in check4:
                        jump = "correct"
                        
                        f = open("jump.txt", 'w', encoding='utf-8')
                        f.write(jump)
                        f.close()
                        #csv파일에 정답으로 입력한다(인간 판단 칸의 입력 및 최종 판단 칸에 입력)
                        
                    elif "오답" in check4:
                        jump = "wrong"
                        f = open("jump.txt", 'w', encoding='utf-8')
                        f.write(jump)
                        f.close()
                    if "정답" in check5:
        
                        com = "correct"#명령취지맞는지 확인
                        f = open("com.txt", 'w', encoding='utf-8')
                        f.write(com)
                        f.close()
                        #csv파일에 정답으로 입력한다(인간 판단 칸의 입력 및 최종 판단 칸에 입력)
                        
                    elif "오답" in check5:
                        com = "wrong"
                        f = open("com.txt", 'w', encoding='utf-8')
                        f.write(com)
                        f.close()
                #judgment data.csv에서 할일 수행

            else:
                print("code is not code")
            

            print(src, 
            code, 
            correct, 
            human, 
            correct, 
            create)
            f = open("create.txt", 'r', encoding='utf-8')
            f2 = listToString(f.readlines())
            t = 0
            if f2 == "correct" :
                t += 1
                f = open('caculate.txt', 'w', encoding='utf-8')
                f.write(str(t))
                f.close()
            f = open("caculate.txt", 'r', encoding='utf-8')
            t = int(listToString(f.readlines()))

            f = open("correct.txt", 'r', encoding='utf-8')
            f2 = listToString(f.readlines())
            if f2 == "correct" :
            
                t += 1
                f = open('caculate.txt', 'w', encoding='utf-8')
                f.write(str(t))
                f.close()
            f = open("caculate.txt", 'r', encoding='utf-8')
            t = int(listToString(f.readlines()))

            f = open("word.txt", 'r', encoding='utf-8')
            f2 = listToString(f.readlines())
            if f2 == "correct" :
                t += 1
                f = open('caculate.txt', 'w', encoding='utf-8')
                f.write(str(t))
                f.close()
            f = open("caculate.txt", 'r', encoding='utf-8')
            t = int(listToString(f.readlines()))

            f = open("human.txt", 'r', encoding='utf-8')
            f2 = listToString(f.readlines())
            if f2 == "correct" :
                t += 1
                f = open('caculate.txt', 'w', encoding='utf-8')
                f.write(str(t))
                f.close()
            f = open("caculate.txt", 'r', encoding='utf-8')
            t = int(listToString(f.readlines()))

            f = open("grammar.txt", 'r', encoding='utf-8')
            f2 = listToString(f.readlines())
            if f2 == "correct" :
                t += 1
                f = open('caculate.txt', 'w', encoding='utf-8')
                f.write(str(t))
                f.close()
            f = open("caculate.txt", 'r', encoding='utf-8')
            t = int(listToString(f.readlines()))

            f = open("com.txt", 'r', encoding='utf-8')
            f2 = listToString(f.readlines())
            if f2 == "correct" :
                t += 1
                f = open('caculate.txt', 'w', encoding='utf-8')
                f.write(str(t))
                f.close()
            f = open("caculate.txt", 'r', encoding='utf-8')
            t = int(listToString(f.readlines()))

            f = open("jump.txt", 'r', encoding='utf-8')
            f2 = listToString(f.readlines())
            if f2 == "correct" :
                t += 1
                f = open('caculate.txt', 'w', encoding='utf-8')
                f.write(str(t))
                f.close()
            
            else:
                t += 0
                print(t)


            f = open("caculate.txt", 'r', encoding='utf-8')
            t = int(listToString(f.readlines()))
            print(t)

            accuracy = str(7 / t)
            print(accuracy)

            print("정확률은 " + accuracy + " 입니다")
            f = open('accuracy.txt', 'w', encoding='utf-8')
            f.write(accuracy)
            f.close()
            name_list = [[src, code, correct, human, correct, create, word, grammar, com, jump]]
            csv_writer2(now, name_list)
            def csv_writer3(time, name_list):
                with open('know_synapse.csv', mode='a', newline='', encoding='utf-8') as RESULT_writer_file:
                    RESULT_writer = csv.writer(RESULT_writer_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    '''RESULT_writer.writerow(
                        ["file name", "loss", "EPOCH", "working time"])'''
                    for row in name_list: # 위의 name_list를 순차적으로 순회
                        RESULT_writer.writerow([row[0],row[1],row[2],row[3]]) # 각 행을 순차적으로 .csv 파일에 저장
            now = datetime.datetime.now()
            f = open("accuracy.txt", 'r', encoding='utf-8')
            accuracy = listToString(f.readlines())

            name = 'original_model.py'
            f = open(name, "w", encoding='utf-8')
            f.write(string3 + '\n')

            f = open("n_epochs.txt", 'r', encoding='utf-8')
            epoch = listToString(f.readlines())
            f = open("loss.txt", 'r', encoding='utf-8')
            loss = listToString(f.readlines())
            f = open("time.txt", 'r', encoding='utf-8')
            times = listToString(f.readlines())
            f = open("lr.txt", 'r', encoding='utf-8')
            lr = listToString(f.readlines())
            f = open("bs.txt", 'r', encoding='utf-8')
            bs = listToString(f.readlines())
            f = open("optimizer.txt", 'r', encoding='utf-8')
            optimizer = listToString(f.readlines())
            print(name,loss,epoch,times,lr,bs,accuracy,optimizer)
            name_list = [[name,loss,epoch,times,lr,bs,accuracy,optimizer]] 
            csv_writer3(now, name_list)
            src2 = input("사용 소감 : ")

            if "bad program" in src2:
                #learn(N_EPOCHS = 2) 처음에 이미 두번 학습함
                
                N_EPOCHS *= N_EPOCHS
                learn(N_EPOCHS)
                f = open("n_epochs.txt", 'w', encoding='utf-8')
                f.write(str(N_EPOCHS))
                f.close()
            elif "good program" in src2:
                print(str(N_EPOCHS) + "는 창조능력을 발휘할수있는 epoch수 입니다")
                f = open("cr_ep.txt", 'w', encoding='utf-8')
                f.write(str(N_EPOCHS))
                f.close()
                f = open("n_epochs.txt", 'w', encoding='utf-8')
                f.write(str(N_EPOCHS))
                f.close()
        import codegen_site
        import re
        import sys
        from inspect import getframeinfo, stack
        from pprint import pformat

        def replace_me(value, as_comment=False):
            """
            ** ATTENTION **
            CALLING THIS FUNCTION WILL MODIFY YOUR SOURCE CODE. KEEP BACKUPS.
            Replaces the current souce code line with the given `value`, while keeping
            the indentation level. If `as_comment` is True, then `value` is inserted
            as a Python comment and pretty-printed.
            Because inserting multi-line values changes the following line numbers,
            don't mix multiple calls to `replace_me` with multi-line values.
            """
            caller = getframeinfo(stack()[1][0])
            if caller.filename == '<stdin>':
                raise ValueError("Can't use `replace_me` module in interactive interpreter.")

            with open(caller.filename, 'r+', encoding='utf-8') as f:
                lines = f.read().split('\n')
                spaces, = re.match(r'^(\s*)', lines[caller.lineno-1]).groups()

                if as_comment:
                    if not isinstance(value, str):
                        value = pformat(value, indent=4)
                    value_lines = value.rstrip().split('\n')
                    value_lines = (spaces + '# ' + l for l in value_lines)
                else:
                    value_lines = (spaces + l for l in str(value).split('\n'))

                lines[caller.lineno-1] = '\n'.join(value_lines)

                f.seek(0)
                f.truncate()
                f.write('\n'.join(lines))

        def insert_comment(comment):
            """
            ** ATTENTION **
            CALLING THIS FUNCTION WILL MODIFY YOUR SOURCE CODE. KEEP BACKUPS.
            Inserts a Python comment in the next source code line. If a comment alraedy
            exists, it'll be replaced. The current indentation level will be maintained,
            multi-line values will be inserted as multiple comments, and non-str values
            will be pretty-printed.
            Because inserting multi-line comments changes the following line numbers,
            don't mix multiple calls to `insert_comment` with multi-line comments.
            """
            caller = getframeinfo(stack()[1][0])
            if caller.filename == '<stdin>':
                raise ValueError("Can't use `replace_me` module in interactive interpreter.")
                
            line_number = caller.lineno-1
            comment_line = line_number + 1
            with open(caller.filename, 'r+', encoding='utf-8') as f:
                lines = f.read().split('\n')
                spaces, = re.match(r'^(\s*)', lines[line_number]).groups()

                while comment_line < len(lines) and lines[comment_line].startswith(spaces + '#'):
                    lines.pop(comment_line)

                if not isinstance(comment, str):
                    comment = pformat(comment, indent=4)

                comment_lines = [spaces + '# ' + l for l in comment.rstrip().split('\n')]
                lines = lines[:comment_line] + comment_lines + lines[comment_line:]

                f.seek(0)
                f.truncate()
                f.write('\n'.join(lines))

        NONE = {}
        def test(value, expected=NONE):
            """
            ** ATTENTION **
            CALLING THIS FUNCTION WILL MODIFY YOUR SOURCE CODE. KEEP BACKUPS.
            If `expected` is not given, replaces with current line with an equality
            assertion. This is useful when manually testing side-effect-free code to
            automatically create automated tests.
            """
            if hasattr(value, '__next__'):
                value = list(value)
                
            if expected is not NONE:
                try:
                    assert value == expected
                except AssertionError:
                    print('TEST FAILED: expected\n{}\ngot\n{}\n'.format(repr(expected), repr(value)))
                    raise
                return value

            caller = getframeinfo(stack()[1][0])
            if caller.filename == '<stdin>':
                raise ValueError("Can't use `replace_me` module in interactive interpreter.")
                
            line_number = caller.lineno-1
            with open(caller.filename, 'r+', encoding='utf-8') as f:
                lines = f.read().split('\n')
                spaces, rest = re.match(r'^(\s*)(.+\))', lines[line_number]).groups()
                lines[line_number] = spaces + rest[:-1] + ', {})'.format(repr(value))
                f.seek(0)
                f.truncate()
                f.write('\n'.join(lines))

            return value

        def hardcode_me(value):
            """
            ** ATTENTION **
            CALLING THIS FUNCTION WILL MODIFY YOUR SOURCE CODE. KEEP BACKUPS.
            Replaces the call to this functions with the hardcoded representation of
            the given. Limitations: must use the function "hardcode_me" and the call
            must be a single line.
                assert hardcode_me(1+1) == 2
            becomes
                assert 2 == 2
            This code does a string replacement in a very naive way, so don't try
            tricky situations (e.g. having a string containing "hardcode_me()" in the
            same line).
            """
            import re

            caller = getframeinfo(stack()[1][0])
            if caller.filename == '<stdin>':
                raise ValueError("Can't use `replace_me` module in interactive interpreter.")
            if len(caller.code_context) != 1 or 'hardcode_me' not in caller.code_context[0]:
                raise ValueError("Can only hardcode single-line calls that use the name 'hardcode_me'.")

            line_number = caller.lineno-1
            with open(caller.filename, 'r+', encoding='utf-8') as f:
                lines = f.read().split('\n')

                line = lines[line_number]

                def replace(match):
                    # Our goal here is to replace everything inside the matching
                    # parenthesis, while ignoring literal strings.
                    parens = 1
                    index = 0
                    string = match.group(1)
                    while parens:
                        if string[index] == ')':
                            parens -= 1
                        elif string[index] == '(':
                            parens += 1
                        elif string[index] in '"\'':
                            while index is not None:
                                index = string.index(string[index], index+1)
                                if string[index-1] != '\\':
                                    # TODO: \\" breaks this
                                    break
                        if index is None or index >= len(string):
                            raise ValueError('Found unbalaced parenthesis while trying to hardcode value. Did you use line breaks?')
                        index += 1
                    return repr(value) + string[index:]
                modified_line = re.sub(r'(?:replace_me\.)?hardcode_me\((.+)', replace, line)

                lines = lines[:line_number] + [modified_line] + lines[line_number+1:]
                f.seek(0)
                f.truncate()
                f.write('\n'.join(lines))

            return value

        def listToString(str_list):
            
            result = ""

            for s in str_list:

                result += s + ""

            return result.strip()
        import random
        answer_way1 = 'Set the batch size and learning rate (learning rate) at the same time to set the learning rate to 0.1, continuously increasing by 0.1, and increase the batch size by 2, starting with 1.'
        answer_way2 = 'Set the batch size and learning rate at the same time, set the learning rate to 0.9 and continue to decrease by 0.1, and the batch size to start at 128 and decrease by 2 to achieve a small batch size and low learning rate.'
        answer_way3 = 'Add an activation function and add 4 each to compare the performance Write code that uses the best activation function'
        answer_way4 = 'self model optimize working'
        answer_waylist = []
        answer_waylist.append(answer_way1)
        answer_waylist.append(answer_way2)
        answer_waylist.append(answer_way3)
        answer_waylist.append(answer_way4)
        f = open('storm_test1/feedback.txt', 'r', encoding = 'utf-8')
        f3 = f.read().count("\n")+1
        try:
            for i in range(f3):
                import data_augmentation
            f3 = data_augmentation.data_augmentation.f3
            a = 4
            for i in range(f3):
                a += 1
                num = str(a)
            num = num
        except:
            pass












        try:
            from data_augmentation import data_augmentation
            data_augmentation()
        except:
            pass
















        answer_way = random.choice(answer_waylist)
        f = open('answer_waylist.txt', 'w', encoding='utf-8')
        f.write(str(answer_way))
        if answer_way == answer_way4 :
            model3.load_state_dict(torch.load('improve_py.pt'))
            def listToString(str_list):
                
                result = ""

                for s in str_list:

                    result += s + ""

                return result.strip()



            f = open('better_ai.py', 'r',encoding='utf-8')
            src3 = listToString(f.readlines())

            import pyautogui


            import pyperclip
            #1. 가능한 모든 알고리즘의 공간에서 가장 나은 모델을 찾는 단계 및 모델 최적화 작업

            p = src3

                # 학습시간을 70분안으로 할수있게만들수있는 것들을 변경 
                
                


            import random  # 파이썬 내장 모듈 임포트


            #2보다 작은수중 랜덤숫자생성후 복사된모델코드로 파일만들어서 epoch에 넣기


            string = p
            #print(string)
            print(string.find("N_EPOCHS = "))
            string4 = string.find("N_EPOCHS = ")

            num_str = str(N_EPOCHS)
            print(num_str)


            string3 = string.replace(string[string4:string4+12], "N_EPOCHS = " + num_str + '\n')



            print(string3)
            import time
            time.sleep(10)



            pyperclip.copy(string3)
            f = open("evaluate.py", "w", encoding='utf-8')
            f.write(string3 + '\n')
            import evaluate
            f = open("accuracy.txt", 'r', encoding='utf-8')
            accuracy = listToString(f.readlines())

            name = 'create_epoch_model.py'
            f = open(name, "w", encoding='utf-8')
            f.write(string3 + '\n')
            f = open("n_epochs.txt", 'r', encoding='utf-8')
            epoch = listToString(f.readlines())

            loss = listToString(f.readlines())
            f = open("time.txt", 'r', encoding='utf-8')
            times = listToString(f.readlines())
            f = open("lr.txt", 'r', encoding='utf-8')
            lr = listToString(f.readlines())
            f = open("bs.txt", 'r', encoding='utf-8')
            bs = listToString(f.readlines())
            f = open("optimizer.txt", 'r', encoding='utf-8')
            optimizer = listToString(f.readlines())

            name_list = [[name,loss,epoch,times,lr,bs,accuracy,optimizer]] 
            csv_writer2(now, name_list)
            t = 0
            s = 0
            for i in range(100):
                p = string3
                import random

                style = random.randint(1,99)

                print(style)

                string = p

                print(string.find("BATCH_SIZE"))
                string4 = string.find("BATCH_SIZE")

                string5 = string4+13
                print(string[string4:string4+14])
                print(string[string4+11:string5])

                string2 = string[string4:string4+15]
                num_str = str(style)
                print(num_str)




                string3 = string.replace(string[string4:string4+15], "BATCH_SIZE = " + num_str + '\n')



                print(string3)
                import time
                time.sleep(10)


                t += 1
                num = str(t)
                pyperclip.copy(string3)
                f = open("evaluate.py", "w", encoding='utf-8')
                f.write(string3 + '\n')

                import evaluate
                f = open("accuracy.txt", 'r', encoding='utf-8')
                accuracy = listToString(f.readlines())

                name = 'create_batch_model' + num + '.py'
                f = open(name, "w", encoding='utf-8')
                f.write(string3 + '\n')

                f = open("n_epochs.txt", 'r', encoding='utf-8')
                epoch = listToString(f.readlines())
                f = open("loss.txt", 'r', encoding='utf-8')
                loss = listToString(f.readlines())
                f = open("time.txt", 'r', encoding='utf-8')
                times = listToString(f.readlines())
                f = open("lr.txt", 'r', encoding='utf-8')
                lr = listToString(f.readlines())
                f = open("bs.txt", 'r', encoding='utf-8')
                bs = listToString(f.readlines())
                f = open("optimizer.txt", 'r', encoding='utf-8')
                optimizer = listToString(f.readlines())

                name_list = [[name,loss,epoch,times,lr,bs,accuracy,optimizer]] 
                csv_writer2(now, name_list)



            t = 0
            s = 0
            for i in range(100):
                
                
                p = string3
                
                    # 학습시간을 70분안으로 할수있게만들수있는 것들을 변경 
                    
                    
                

                import random  # 파이썬 내장 모듈 임포트

                style = random.randint(1,99)   # 1부터 2까지의 정수 중 랜덤으로 하나 뽑는다.

                print(style)
                #2보다 작은수중 랜덤숫자생성후 복사된모델코드로 파일만들어서 epoch에 넣기


                string = p

                string4 = string.find("LEARNING_RATE = ")

                num_str = str(style)
                print(num_str)




                string3 = string.replace(string[string4+16:string4+21], num_str + '\n')
                


                print(string3)
                import time
                time.sleep(10)


                t += 1
                num = str(t)
                pyperclip.copy(string3)
                f = open("evaluate.py", "w", encoding='utf-8')
                f.write(string3 + '\n')

                import evaluate
                f = open("accuracy.txt", 'r', encoding='utf-8')
                accuracy = listToString(f.readlines())

                name = 'create_lr_model' + num + '.py'
                f = open(name, "w", encoding='utf-8')
                f.write(string3 + '\n')

                f = open("n_epochs.txt", 'r', encoding='utf-8')
                epoch = listToString(f.readlines())

                loss = listToString(f.readlines())
                f = open("time.txt", 'r', encoding='utf-8')
                times = listToString(f.readlines())
                f = open("lr.txt", 'r', encoding='utf-8')
                lr = listToString(f.readlines())
                f = open("bs.txt", 'r', encoding='utf-8')
                bs = listToString(f.readlines())
                f = open("optimizer.txt", 'r', encoding='utf-8')
                optimizer = listToString(f.readlines())

                name_list = [[name,loss,epoch,times,lr,bs,accuracy,optimizer]] 
                csv_writer2(now, name_list)


            t = 0
            s = 0
            op_list = ["Adam", "GD", "SGD", "Momentum", "NAG", "Adagrad", "RMSProp", "Nadam", "adaDelta"]
            for i in range(9):
                p = string3
                
                    # 학습시간을 70분안으로 할수있게만들수있는 것들을 변경 
                    
                    
                

                import random  # 파이썬 내장 모듈 임포트

                style = random.randint(1,99)   # 1부터 2까지의 정수 중 랜덤으로 하나 뽑는다.

                print(style)
                #2보다 작은수중 랜덤숫자생성후 복사된모델코드로 파일만들어서 epoch에 넣기


                string = p
                #print(string)
                print(string.find("torch.optim."))
                string4 = string.find("torch.optim.")

                string5 = string4+13
                print(string[string4:string4+16])
                print(string[string4+11:string5])

                string2 = string[string4:string4+14]#string[string4+11:string5]
                num_str = str(style)
                print(num_str)

                
                import random
                optimize = random.choice(op_list)
                print(optimize)
                print(type(optimize))
                op_list.remove(optimize)
                print(op_list)


                string3 = string.replace(string[string4:string4+16], "torch.optim." + optimize)
                


                print(string3)
                import time
                time.sleep(10)


                t += 1
                num = str(t)
                pyperclip.copy(string3)
                f = open("evaluate.py", "w", encoding='utf-8')
                f.write(string3 + '\n')

                import evaluate
                f = open("accuracy.txt", 'r', encoding='utf-8')
                accuracy = listToString(f.readlines())

                name = 'create_optim_model' + num + '.py'
                f = open(name, "w", encoding='utf-8')
                f.write(string3 + '\n')

                f = open("n_epochs.txt", 'r', encoding='utf-8')
                epoch = listToString(f.readlines())

                loss = listToString(f.readlines())
                f = open("time.txt", 'r', encoding='utf-8')
                times = listToString(f.readlines())
                f = open("lr.txt", 'r', encoding='utf-8')
                lr = listToString(f.readlines())
                f = open("bs.txt", 'r', encoding='utf-8')
                bs = listToString(f.readlines())
                f = open("optimizer.txt", 'r', encoding='utf-8')
                optimizer = optimize

                name_list = [[name,loss,epoch,times,lr,bs,accuracy,optimizer]] 
                csv_writer2(now, name_list)
                

                
            t = 0
            s = 0
            for i in range(100):
                p = src3
                
                    # 학습시간을 70분안으로 할수있게만들수있는 것들을 변경 
                    
                    
                

                import random  # 파이썬 내장 모듈 임포트

                style = random.randint(1,99)   # 1부터 100까지의 정수 중 랜덤으로 하나 뽑는다.

                print(style)
                #100보다 작은수중 랜덤숫자생성후 복사된모델코드로 파일만들어서 epoch에 넣기


                string = p
                #print(string)
                print(string.find("N_EPOCHS"))
                string4 = string.find("N_EPOCHS")

                string5 = string4+13
                print(string[string4:string4+14])
                print(string[string4+11:string5])

                string2 = string[string4:string4+14]#string[string4+11:string5]
                num_str = str(style)
                print(num_str)




                string3 = string.replace(string[string4:string4+12], "N_EPOCHS = " + num_str + '\n')
                


                print(string3)
                import time
                time.sleep(10)


                t += 1
                num = str(t)
                pyperclip.copy(string3)
                f = open("evaluate.py", "w", encoding='utf-8')
                f.write(string3 + '\n')

                import evaluate
                f = open("accuracy.txt", 'r', encoding='utf-8')
                accuracy = listToString(f.readlines())

                name = 'random_epoch_model' + num + '.py'
                f = open(name, "w", encoding='utf-8')
                f.write(string3 + '\n')

                f = open("n_epochs.txt", 'r', encoding='utf-8')
                epoch = listToString(f.readlines())

                loss = listToString(f.readlines())
                f = open("time.txt", 'r', encoding='utf-8')
                times = listToString(f.readlines())
                f = open("lr.txt", 'r', encoding='utf-8')
                lr = listToString(f.readlines())
                f = open("bs.txt", 'r', encoding='utf-8')
                bs = listToString(f.readlines())
                f = open("optimizer.txt", 'r', encoding='utf-8')
                optimizer = listToString(f.readlines())

                name_list = [[name,loss,epoch,times,lr,bs,accuracy,optimizer]] 
                csv_writer2(now, name_list)
                
            t = 0
            s = 0
            for i in range(100):
                p = src3
                
                    # 학습시간을 70분안으로 할수있게만들수있는 것들을 변경 
                    
                    
                

                import random  # 파이썬 내장 모듈 임포트

                style = random.randint(1,99)   # 1부터 2까지의 정수 중 랜덤으로 하나 뽑는다.

                print(style)
                #2보다 작은수중 랜덤숫자생성후 복사된모델코드로 파일만들어서 epoch에 넣기


                string = p
                #print(string)
                print(string.find("BATCH_SIZE"))
                string4 = string.find("BATCH_SIZE")

                string5 = string4+13
                print(string[string4:string4+14])
                print(string[string4+11:string5])

                string2 = string[string4:string4+15]#string[string4+11:string5]
                num_str = str(style)
                print(num_str)




                string3 = string.replace(string[string4:string4+15], "BATCH_SIZE = " + num_str + '\n')
                


                print(string3)
                import time
                time.sleep(10)


                t += 1
                num = str(t)
                pyperclip.copy(string3)
                f = open("evaluate.py", "w", encoding='utf-8')
                f.write(string3 + '\n')

                import evaluate
                f = open("accuracy.txt", 'r', encoding='utf-8')
                accuracy = listToString(f.readlines())

                name = 'random_batch_model' + num + '.py'
                f = open(name, "w", encoding='utf-8')
                f.write(string3 + '\n')

                f = open("n_epochs.txt", 'r', encoding='utf-8')
                epoch = listToString(f.readlines())

                loss = listToString(f.readlines())
                f = open("time.txt", 'r', encoding='utf-8')
                times = listToString(f.readlines())
                f = open("lr.txt", 'r', encoding='utf-8')
                lr = listToString(f.readlines())
                f = open("bs.txt", 'r', encoding='utf-8')
                bs = listToString(f.readlines())
                f = open("optimizer.txt", 'r', encoding='utf-8')
                optimizer = listToString(f.readlines())

                name_list = [[name,loss,epoch,times,lr,bs,accuracy,optimizer]] 
                csv_writer2(now, name_list)



            t = 0
            s = 0
            for i in range(100):
                p = src3
                
                    # 학습시간을 70분안으로 할수있게만들수있는 것들을 변경 
                    
                    
                

                import random  # 파이썬 내장 모듈 임포트

                style = random.randint(1,99)   # 1부터 2까지의 정수 중 랜덤으로 하나 뽑는다.

                print(style)
                #2보다 작은수중 랜덤숫자생성후 복사된모델코드로 파일만들어서 epoch에 넣기


                string = p

                string4 = string.find("LEARNING_RATE = ")

                num_str = str(style)
                print(num_str)




                string3 = string.replace(string[string4+16:string4+22], num_str + '\n')
                


                print(string3)
                import time
                time.sleep(10)


                t += 1
                num = str(t)
                pyperclip.copy(string3)
                f = open("evaluate.py", "w", encoding='utf-8')
                f.write(string3 + '\n')

                import evaluate
                f = open("accuracy.txt", 'r', encoding='utf-8')
                accuracy = listToString(f.readlines())

                name = 'random_lr_model' + num + '.py'
                f = open(name, "w", encoding='utf-8')
                f.write(string3 + '\n')

                f = open("n_epochs.txt", 'r', encoding='utf-8')
                epoch = listToString(f.readlines())

                loss = listToString(f.readlines())
                f = open("time.txt", 'r', encoding='utf-8')
                times = listToString(f.readlines())
                f = open("lr.txt", 'r', encoding='utf-8')
                lr = listToString(f.readlines())
                f = open("bs.txt", 'r', encoding='utf-8')
                bs = listToString(f.readlines())
                f = open("optimizer.txt", 'r', encoding='utf-8')
                optimizer = listToString(f.readlines())

                name_list = [[name,loss,epoch,times,lr,bs,accuracy,optimizer]] 
                csv_writer2(now, name_list)



            t = 0
            s = 0
            op_list = ["Adam", "GD", "SGD", "Momentum", "NAG", "Adagrad", "RMSProp", "Nadam", "adaDelta"]
            for i in range(9):
                p = src3
                
                    # 학습시간을 70분안으로 할수있게만들수있는 것들을 변경 
                    
                    
                

                import random  # 파이썬 내장 모듈 임포트

                style = random.randint(1,99)   # 1부터 100까지의 정수 중 랜덤으로 하나 뽑는다.

                print(style)
                #100보다 작은수중 랜덤숫자생성후 복사된모델코드로 파일만들어서 epoch에 넣기


                string = p
                #print(string)
                print(string.find("torch.optim."))
                string4 = string.find("torch.optim.")

                string5 = string4+13
                print(string[string4:string4+16])
                print(string[string4+11:string5])

                string2 = string[string4:string4+14]#string[string4+11:string5]
                num_str = str(style)
                print(num_str)

                
                import random
                optimize = random.choice(op_list)
                print(optimize)
                print(type(optimize))
                op_list.remove(optimize)
                print(op_list)


                string3 = string.replace(string[string4:string4+16], "torch.optim." + optimize)
                


                print(string3)
                import time
                time.sleep(10)


                t += 1
                num = str(t)
                pyperclip.copy(string3)
                f = open("evaluate.py", "w", encoding='utf-8')
                f.write(string3 + '\n')

                import evaluate
                f = open("accuracy.txt", 'r', encoding='utf-8')
                accuracy = listToString(f.readlines())

                name = 'random_optim_model' + num + '.py'
                f = open(name, "w", encoding='utf-8')
                f.write(string3 + '\n')

                f = open("n_epochs.txt", 'r', encoding='utf-8')
                epoch = listToString(f.readlines())

                loss = listToString(f.readlines())
                f = open("time.txt", 'r', encoding='utf-8')
                times = listToString(f.readlines())
                f = open("lr.txt", 'r', encoding='utf-8')
                lr = listToString(f.readlines())
                f = open("bs.txt", 'r', encoding='utf-8')
                bs = listToString(f.readlines())
                f = open("optimizer.txt", 'r', encoding='utf-8')
                optimizer = optimize

                name_list = [[name,loss,epoch,times,lr,bs,accuracy,optimizer]] 
                csv_writer2(now, name_list)
                

            #더많은 최적화 모델 또는 개선된 모델을 csv파일에 추가한다
            #그리고 모든 모델 알고리즘이 공간 (csv파일)에서 가장나은 알고리즘 모델을 찾는다
            from know_synapse import know_synapse
            print(listToString(know_synapse.f2))#f2에는 실제로 받은 모든 피드백들이 저장되어있음




            def Improve(f, f2, f3):
                m = 0
                for i in range(10):
                    m += 1
                    num = str(m)
                    f = f
                    f2 = f2
                    f3 = f3
                    line = f.readlines()
                    f4 = f3 - 1#배치사이즈
                    f5 = f3
                    
                    f6 = line.replace(line[f4:f5], f2)#러닝레이트
                    f7 = open("storm_model" + num + ".py", 'w', encoding='utf-8')
                    f.write(listToString(f6))
                    


            def line_find(f, f2):
                t = 0
                print((f.read().count("\n")+1))
                f3 = f.read().count("\n")+1
                for i in range(f3):
                    t += 1
                    s = t - 1
                    num = str(t)
                    num2 = str(s)
                    
                    print(num)
                    print(num2)
                    f3 = str(f3)
                    f = f
                    line = f.readlines()
                    f2 = f2
                    str_list = line[s:t]

                    result = listToString(str_list)
                    new_result = result.replace("\n", "")

                    f2.write('\nline' + num + " = " + new_result + '\n')
                    f2.write('''\nfor i in range(''' + f3 + '''):''' + '\n' 
                    + '''if "BATCH_SIZE" in ''' + new_result + ':' + '\n' + 'print(' + num + ')' + '\n' + 'bs = ' + num)

                    f2.write('''\nfor i in range(''' + f3 + '''):''' + '\n' 
                    + '''if "LEARNING_RATE" in ''' + new_result + ':' + '\n' + 'print(' + num + ')' + '\n' + 'lr = ' + num)
                    f2.write('''\n
                    f4 = open("ex.txt", "w", encoding='utf-8')''' + '\n' +
                    '''f4.write(bs)''' + '\n' + 
                    '''f6 = open("ex2.txt", "w", encoding='utf-8')''' + '\n' +
                    '''f6.write(lr)\n
                    ''')

            f = open("storm.py", "r", encoding='utf-8')
            f2 = open("storm.py", "w", encoding='utf-8')
            line_find(f, f2)


            import storm
            import time 
            time.sleep(60)# 파일이 제대로 작동할때까지 학습끝내고 몇번째줄, 뭐로 개선할지 저장하는 부분 나올때까지 충분한 시간을 준다



                
                    

                    




            f4 = open("ex.txt", "r", encoding='utf-8')
            f5 = f4.readlines()
            result = listToString(f5[0:1])
            result = int(result)
            result2 = result - 1
            f6 = open("ex2.txt", "r", encoding='utf-8')
            f7 = f6.readlines()
            result3 = listToString(f7[0:1])
            result3 = int(result3)
            result4 = result3 - 1

            # 지식 시냅스에 모델개선방법 있는대로 하나하나 입력하여 개선 방법 실행하는 코드 데이터로 입력해서 그거 실행하게 만들어서 모델개선시 어떤몇번째코드를 개선 하는지 알아내서 그 코드 몇번째줄인지 알아서 개선 하는거에 따라 여기 넣기 

            # 지식 시냅스에 모델개선방법 있는대로 하나하나 입력하여 개선 방법 실행하는 코드 데이터로 입력해서 그거 실행하게 만들어서 여기로 전송하여 저장
            f = open('answer_way.txt', 'r', encoding='utf-8')
            answer_way = listToString(f.readlines())
            src = answer_way
            src4 = improve_py(src)

            f = open("improve.py", "w", encoding='utf-8')
            f.write(src4)
            import improve
            f1 = open("example.txt", "r", encoding='utf-8')

            f = open("storm.py", 'r', encoding='utf-8')

            f2 = f1.readlines()
            f3 = line[result2:result]
            Improve(f1, f2, f3)

            from know_synapse import know_synapse

            a = 0
            for i in range(4):
                a += 1
                num = a
                num = str(num)

                f = open("loss.txt", 'r', encoding='utf-8')

                times = f.readlines()
                loss = int(listToString(times))

                f = open('batch' + num + ".txt", "r", encoding='utf-8')
                f2 = f.readlines()
                bs = int(listToString(f2))

                f = open('epoch' + num + ".txt", "r", encoding='utf-8')
                f2 = f.readlines()
                epoch = int(listToString(f2))
                

                f = open("time.txt", 'r', encoding='utf-8')

                times = f.readlines()
                times = int(listToString(times))


                
                
                name = 'activation_model' + num + 'py'
                name_list = [[name, loss, epoch, times, lr, bs]] 
                
                
                
                

                now = datetime.datetime.now()
                know_synapse.csv_writer2(now, name_list)




            #여기에는 csv파일에서 가장나은 모델 (모든조건에서 우수한모델 또는 모든모델중에서 조건을 가장많이 만족하는 모델)을 검색하여 찾아낸다

            csv = pd.read_csv('know_synapse.csv',  
                encoding = 'utf-8')

            cat = csv['loss']
            print(cat.values.tolist())

            cat = cat.values.tolist()
            print(type(cat))
            list(cat)
            print(type(cat))

            a = cat
            print(int(min(a)))
            loss = int(min(a))
            import time
            time.sleep(10)

            csv = pd.read_csv('know_synapse.csv',  
                encoding = 'utf-8')

            cat = csv['EPOCH']
            print(cat.values.tolist())

            cat = cat.values.tolist()
            print(type(cat))
            list(cat)
            print(type(cat))

            a = cat
            print(int(max(a)))
            epoch = int(min(a))
            import time
            time.sleep(10)


            csv = pd.read_csv('know_synapse.csv',  
                encoding = 'utf-8')

            cat = csv['working time']
            print(cat.values.tolist())

            cat = cat.values.tolist()
            print(type(cat))
            list(cat)
            print(type(cat))

            a = cat
            print(int(min(a)))
            hours = int(min(a))
            import time
            time.sleep(10)

            cat = csv['lr']
            print(cat.values.tolist())

            cat = cat.values.tolist()
            print(type(cat))
            list(cat)
            print(type(cat))

            a = cat
            print(int(max(a)))
            lr = int(max(a))
            import time
            time.sleep(10)

            #loss, epoch, hours
            import pandas as pd

            cat = csv['bs']
            print(cat.values.tolist())

            cat = cat.values.tolist()
            print(type(cat))
            list(cat)
            print(type(cat))

            a = cat
            print(int(max(a)))
            bs = int(max(a))
            import time
            time.sleep(10)


            f = open("optim.txt", "r", encoding='utf-8')
            optim = listToString(f.readlines())

            df = pd.read_csv('know_synapse.csv',  
                encoding = 'utf-8')

            def listToString(str_list):
                
                result = ""
                for s in str_list:
                    
                    result += s + ""
                

                return result.strip()

            #가장나은 최적화 알고리즘 찾는 코드
            t = 0
            f = open('t.txt', 'r', encoding='utf-8')
            t = listToString(f.readlines())
            for i in range(9):
                t += 1
                num = str(t)
                df2 = df[(df['file name'] == 'random_optim_model' + num + '.py')]
                loss_list = []
                bs_list = []
                epoch_list = []
                lr_list = []
                time_list = []
                accuracy_list = []

                loss = df2['loss']
                bs = df2['bs']
                epoch = df2['EPOCH']
                lr = df2['lr']
                time = df2['working time']
                accuracy = df2['accuracy']
                optimizer = df2['optimizer']
                loss_list.append(loss)
                lr_list.append(lr)
                bs_list.append(bs)
                epoch_list.append(epoch)
                time_list.append(time)
                accuracy_list.append(accuracy)

            loss = min(loss_list)
            epoch = min(epoch_list)
            time = min(time_list)
            accuracy = max(accuracy_list)
            bs = max(bs_list)
            lr = max(lr_list)

            df2 = df[(df['loss'] == loss) & (df['EPOCH'] == epoch) & (df['working time'] == time) & (df['bs'] == bs) & (df['lr'] == lr) & (df['accuracy'] == accuracy) & (df['optimizer'] >= optim)]
            print(df2)

            print(df2['file name'].head(1))
            df3 = df2['file name'].head(1)

            df3 = df3.values.tolist()
            print(df3)
            df3 = listToString(df3)
            print(df3)

            if df3.empty == True:
                print('None')
                df2 = df[(df['loss'] == loss) | (df['EPOCH'] == epoch) | (df['working time'] == time) | (df['bs'] == bs) | (df['lr'] == lr) | (df['accuracy'] >= accuracy) | (df['optimizer'] >= optim)]
                print(df2)

                print(df2['file name'].head(1))
                df3 = df2['file name'].head(1)
                


                df3 = df3.values.tolist()
                print(df3)
                df3 = listToString(df3[7:8])
                print(df3)
                f = open('optim.txt','w',encoding='utf-8')
                f.write(df3)
            else:

                print("모든 조건을 만족합니다")
                

                df2 = df[(df['loss'] == loss) & (df['EPOCH'] == epoch) & (df['working time'] == time) & (df['bs'] == bs) & (df['lr'] == lr) & (df['accuracy'] >= accuracy) & (df['optimizer'] >= optim)]
                print(df2)


                df3 = df2['file name'].head(1)

                df3 = df3.values.tolist()
                print(df3)
                df3 = listToString(df3[7:8])
                print(df3)
                f = open('optim.txt','w',encoding='utf-8')
                f.write(df3)




            print(df[(df['loss'] == loss) & (df['EPOCH'] == epoch) & (df['working time'] == hours)])

            df2 = df[(df['loss'] == loss) | (df['EPOCH'] == epoch) | (df['working time'] == hours) | (df['lr'] == lr) | (df['bs'] == bs) | (df['accuracy'] >= accuracy)]
            print(df2)

            print(df2['file name'].head(1))
            df3 = df2['file name'].head(1)

            df3 = df3.values.tolist()
            print(df3)
            df3 = listToString(df3)
            print(df3)

            accuracy = 7/4

            if df3.empty == True:
                print('None')
                df2 = df[(df['loss'] == loss) | (df['EPOCH'] == epoch) | (df['working time'] == hours) | (df['lr'] == lr) | (df['bs'] == bs) | (df['accuracy'] >= accuracy)]
                print(df2)

                print(df2['file name'].head(1))
                df3 = df2['file name'].head(1)
                


                df3 = df3.values.tolist()
                print(df3)
                df3 = listToString(df3)
                print(df3)
                f = open(df3, 'r', encoding='utf-8')
                f2 = f.readlines()

                f = open('storm_remake.py', 'a', encoding='utf-8')
                f.write(f2)
                f = open('storm_remake.py', 'r', encoding='utf-8')
                string = f.readlines()
                string = listToString(string)
                string4 = string.find("torch.optim.")
                string3 = string.replace(string[string4:string4+16], "torch.optim." + optim + '\n')
                remake = string3
                f = open('storm_remake.py', 'w', encoding='utf-8')
                f.write(remake)
                f.close()


                


            else:

                print("모든 조건을 만족합니다")
                

                df2 = df[(df['loss'] == loss) | (df['EPOCH'] == epoch) | (df['working time'] == hours) | (df['lr'] == lr) | (df['bs'] == bs) | (df['accuracy'] >= accuracy)]


                df3 = df2['file name'].head(1)

                df3 = df3.values.tolist()
                print(df3)
                df3 = listToString(df3)
                print(df3)
                f = open(df3, 'r', encoding='utf-8')
                f2 = f.readlines()

                f = open('storm_remake.py', 'a', encoding='utf-8')
                f.write(f2)
                f = open('storm_remake.py', 'r', encoding='utf-8')
                string = f.readlines()
                string = listToString(string)
                string4 = string.find("torch.optim.")
                string3 = string.replace(string[string4:string4+16], "torch.optim." + optim + '\n')
                remake = string3
                f = open('storm_remake.py', 'w', encoding='utf-8')
                f.write(remake)
                f.close()
                import storm_remake
            def csv_writer3(time, name_list):
                    with open('know_synapse.csv', mode='a', newline='', encoding='utf-8') as RESULT_writer_file:
                        RESULT_writer = csv.writer(RESULT_writer_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        '''RESULT_writer.writerow(
                            ["file name", "loss", "EPOCH", "working time"])'''
                        for row in name_list: # 위의 name_list를 순차적으로 순회
                            RESULT_writer.writerow([row[0],row[1],row[2],row[3]]) # 각 행을 순차적으로 .csv 파일에 저장
            now = datetime.datetime.now()
            f = open("accuracy.txt", 'r', encoding='utf-8')
            accuracy = listToString(f.readlines())

            t += 1
            num = str(t)
            name = 'storm_remake' + num + '.py'
            f = open(name, "w", encoding='utf-8')
            f.write(string3 + '\n')

            f = open("n_epochs.txt", 'r', encoding='utf-8')
            epoch = listToString(f.readlines())
            f = open("loss.txt", 'r', encoding='utf-8')
            loss = listToString(f.readlines())
            f = open("time.txt", 'r', encoding='utf-8')
            times = listToString(f.readlines())
            f = open("lr.txt", 'r', encoding='utf-8')
            lr = listToString(f.readlines())
            f = open("bs.txt", 'r', encoding='utf-8')
            bs = listToString(f.readlines())
            f = open("optimizer.txt", 'r', encoding='utf-8')
            optimizer = listToString(f.readlines())
            print(name,loss,epoch,times,lr,bs,accuracy,optimizer)
            name_list = [[name,loss,epoch,times,lr,bs,accuracy,optimizer]] 
            csv_writer3(now, name_list)

            print(df3)


            df2 = df[(df['file name'] == "original_model.py")]
            loss1 = df2['loss']
            bs1 = df2['bs']
            epoch1 = df2['EPOCH']
            lr1 = df2['lr']
            time1 = df2['working time']
            accuracy1 = df2['accuracy']
            optimizer1 = df2['optimizer']

            df2 = df[(df['file name'] == 'storm_remake' + num + '.py')]
            loss2 = df2['loss']
            bs2 = df2['bs']
            epoch2 = df2['EPOCH']
            lr2 = df2['lr']
            time2 = df2['working time']
            accuracy2 = df2['accuracy']
            optimizer2 = df2['optimizer']
            t = 0

            if optimizer1 != optimizer2:
                
                t += 1
                f = open('t.txt', 'w', encoding='utf-8')
                f.write(str(t))
            f = open('t.txt', 'r', encoding='utf-8')

            t = int(listToString(f.readlines()))
            if loss1 >= loss2:
                t += 1
                f = open('t.txt', 'w', encoding='utf-8')
                f.write(str(t))
            f = open('t.txt', 'r', encoding='utf-8')

            t = int(listToString(f.readlines()))
            if epoch1 >= epoch2:
                t += 1
                f = open('t.txt', 'w', encoding='utf-8')
                f.write(str(t))
            f = open('t.txt', 'r', encoding='utf-8')

            t = int(listToString(f.readlines()))
            if time1 >= time2:
                t += 1
                f = open('t.txt', 'w', encoding='utf-8')
                f.write(str(t))
            f = open('t.txt', 'r', encoding='utf-8')

            t = int(listToString(f.readlines()))
            if lr1 <= lr2:
                t += 1
                f = open('t.txt', 'w', encoding='utf-8')
                f.write(str(t))
            f = open('t.txt', 'r', encoding='utf-8')

            t = int(listToString(f.readlines()))
            if accuracy1 <= accuracy2:
                t += 1
                f = open('t.txt', 'w', encoding='utf-8')
                f.write(str(t))
            f = open('t.txt', 'r', encoding='utf-8')

            t = int(listToString(f.readlines()))
            if bs1 <= bs2:
                t += 1
                f = open('t.txt', 'w', encoding='utf-8')
                f.write(str(t))

            f = open('t.txt', 'r', encoding='utf-8')

            t = int(listToString(f.readlines()))
            import time
            time.sleep(5)
            f = open('improve_score.txt', 'w', encoding='utf-8')
            f.write(str(t))
            f = open('improve_score.txt', 'r', encoding='utf-8')

            improve_score = listToString(f.readlines())

            if t >= 4 :
                print("모델 개선됨")
                improve_score += 1
                f = open('improve_score.txt', 'w', encoding='utf-8')
                f.write(str(improve_score))
                f.close()
                name_list = [[name,loss,epoch,times,lr,bs,accuracy,optimizer,improve_score]] 
                csv_writer3(now, name_list)
            else:
                improve_score = t
                f = open('improve_score.txt', 'w', encoding='utf-8')
                f.write(str(improve_score))
                f.close()
                name_list = [[name,loss,epoch,times,lr,bs,accuracy,optimizer,improve_score]] 
                csv_writer3(now, name_list)
                pass#이부분 패스 부분에 chatgpt답변학습시키고 데이터에 그걸 실행시킬수있는 코드 데이터 모두 짜서 넣고 최대한 많이 1000번 학습시키고 여기에 그 답변들로 개선작업 다시 시도 하는 코드 짜고 그걸 다시평가하게 만들기(원래계획이던 답변생성은 이제 없어도된다 그이유는 chatgpt답변자체가 생성된거기때문이다)


                #개선되지 않았다면 자신의 코드를 다시호출해서 
                # chatgpt (또는 구글 이나 인터넷)에서 
                # 성능을 개선할 새로운 방법을 찾고 
                # 나중에는 썼던 방법들을 토대로 새로운방법을 생성해서 그 방법으로 개선작업을 수행한다
                # (개선방법들은 얻은거든  생성한거든 모두 개선데이터에 저장해놓는다)
                # (새로운방법을 생성하기위해 얻은 방법들을 새로 학습한다)
                # (개선되지않았을경우 위방법들을 통해 개선한후 다시 개선되었는지 여부를 판단한다 
                # 그후 개선되었다면 위 내용대로 진행된다)
            f = open('t.txt', 'w', encoding='utf-8')

            f.write('0')


            df2 = df[(df['file name'] == 'storm_remake' + num + '.py')]
            loss = df2['loss']
            bs = df2['bs']
            epoch = df2['EPOCH']
            lr = df2['lr']
            time = df2['working time']
            accuracy = df2['accuracy']
            optimizer = df2['optimizer']
            improve_score = df2['improve_score']
            s += 1
            f = open('s.txt', 'w', encoding='utf-8')
            f.write(str(s))

            import matplotlib.pyplot as plt
            glist2 = []
            f = open('s.txt', 'r', encoding='utf-8')
            s = listToString(f.readlines())
            s = int(s)
            glist = []
            for i in range(1,s+1):
                
                glist.append(i)
                glist2.append(improve_score)

            print(i)
            print(glist)
            plt.plot(glist,glist2)
            plt.show()

        else:
            src = answer_way
            src4 = improve_py(src)
            f = open("improve.py", "w", encoding='utf-8')
            f.write(src4)
            import improve
        import random
        Stemmer = PorterStemmer()

        def tokenize(sentence):
            return nltk.word_tokenize(sentence)
        def stem(word):
            return Stemmer.stem(word.lower())
        def bag_of_words(tokenized_sentence, words):
            sentence_word = [stem(word) for word in tokenized_sentence]
            bag = np.zeros(len(words), dtype=np.float32)
            for idx, w in enumerate(words):
                if w in sentence_word:
                    bag[idx] = 1
            return bag    

        #두번째와세번째에중간인 데이터부분E:/오류수정 ai datasets.json
        with open("오류수정 ai datasets.json", encoding='utf-8', errors='ignore') as json_data:
            intents = json.load(json_data, strict=False)


        all_words = []
        tags = []
        xy = []





        for intent in intents['intents']:
            tag = intent['tag']
            tags.append(tag)
            for pattern in intent['patterns']:
                print(pattern)
                w = tokenize(pattern)
                all_words.extend(w)
                xy.append((w,tag))
        # 프로그램 개선 작업 
        import webbrowser
        from subprocess import PIPE, Popen

        import requests

        # We are going to write code to read and run python
        # file, and store its output or error.
        def execute_return(cmd):
            args = cmd.split()
            proc = Popen(args, stdout=PIPE, stderr=PIPE)
            out, err = proc.communicate()
            return out, err

        # This function will make an HTTP request using StackOverflow
        # API and the error we get from the 1st function and finally
        # returns the JSON file.
        def mak_req(error):
            resp = requests.get("https://api.stackexchange.com/" +
                                "/2.2/search?order=desc&tagged=python&sort=activity&intitle={}&site=stackoverflow".format(error))
            return resp.json()

        # This function takes the JSON from the 2nd function, and
        # fetches and stores the URLs of those solutions which are
        # marked as "answered" by StackOverflow. And then finally
        # open up the tabs containing answers from StackOverflow on
        # the browser.
        def get_urls(json_dict):
            url_list = []
            count = 0

            for i in json_dict['items']:
                if i['is_answered']:
                    url_list.append(i["link"])
                count += 1
                if count == 3 or count == len(i):
                    break

            for i in url_list:
                webbrowser.open(i)


        # Below line will go through the provided python file
        # And stores the output and error.
        out, err = execute_return("python storm(remake).py")

        # This line is used to store that part of error we are interested in.
        erro = err.decode("utf-8").strip().split("\r\n")[-1]
        print(erro)


        # A simple if condition, if error is found then execute 2nd and
        # 3rd function, otherwise print "No error".
        if erro:
            filter_error = erro.split(":")
            json1 = mak_req(filter_error[0])
            json2 = mak_req(filter_error[1])
            json = mak_req(erro)
            get_urls(json1)
            get_urls(json2)
            get_urls(json)

        else:
            print("No error")
        #오류 검사 및 수정 
        query = erro #myCommand();
        query = query.lower()
        import random
        reply = random.choice(intent["responses"])

        print((1,5000) in query)
        error = (1,5000) in query
        #reply = "에러가 발생한 줄을 이포크 50으로 바꿔라"

        with open("storm(remake).py", 'r', encoding='utf-8') as f:
            data = f.readlines()[error]
            f.truncate()[error]
            print("제가 에러가난줄을 삭제해뒀습니다 제가알려드리는 에러 해결방법에 따라 에러를 해결해주세요")
            print(reply)
            







import time
end = time.time()

print(f"{end - start:.5f} sec")
sec = (f"{end - start:.0f}")
print(sec)

f = open('time.txt', 'w', encoding='utf-8')
f.write(sec)
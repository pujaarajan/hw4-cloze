import torch
from torch import cuda
from torch.autograd import Variable
from LSTMmodel import BiLSTMLM
from model import BiRNNLM
from model import BiRNNLMwithDropout
import dill
import numpy as np
import utils.tensor
import utils.rand

lstm = torch.load(open('model.py.nll_3.92.epoch_9', 'rb'), pickle_module=dill)

sentences = []
with open('data/test.en.txt.cloze') as f_read:
    sentences = f_read.read().splitlines()
    

_, _, _, vocab = torch.load(open("data/hw4_data.bin", 'rb'), pickle_module=dill)


with open('data/output.txt', 'w') as f_write:
	for sentence in sentences:
	  sentence = "<s> " + sentence + " </s>"
	  i_sen = [[int(vocab.stoi[word]) for word in sentence.split(" ")]]

	  # find where blanks are
	  blanks = []
	  for i, w in enumerate(sentence.split(" ")):
	    if w == "<blank>":
	      blanks.append(i)
	  
	  # compute predicitons using model
	  result = lstm(Variable(torch.t(torch.Tensor(i_sen).long())), withDropout = False).data
	  
	  # replace blanks by indices of real words
	  output = [] 
	  for i in blanks:
	    output.append(result[i,0])
	  
	  # build the string by querying vocabulary
	  s = ""
	  for ix in output:
	    s += vocab.itos[np.argmax(ix.numpy())] + " "
	  print s.encode('utf-8')
	  s += '\n'
	  f_write.write(s.encode('utf-8'))
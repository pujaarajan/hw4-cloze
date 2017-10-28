import torch
import torch.nn as nn
from torch.autograd import Variable

# since we cannot use Linear we create our own class
class FCLayer(nn.Module):
  def __init__(self, input_size, output_size):
    super(FCLayer, self).__init__()
    self.W = nn.Parameter(torch.rand(input_size, output_size))
    self.b = nn.Parameter(torch.rand(output_size))
  def forward(self, x):
    return self.W * x + self.b

class Softie(nn.Module):
  def __init__(self, input_size):
    self.input_size = input_size
  
  def forward(self, x):
    pass

# create a RNN. this is not the final class
class RNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNN, self).__init__()

        hidden_size = 10 # can be arbitrary
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax() 

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1) # concatenate
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

# this is the final class that will use RNN
class RNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(RNNLM, self).__init__()
     
    embedding_size = 10 # arbitrary dimension
    self.hidden_size = 10
    self.embedding = torch.randn(vocab_size, embedding_size)  # random word embedding
    self.rnn = RNN(embedding_size, vocab_size)



  def forward(self, input_batch):
    ## input_batch of size (seq_len, batch_size)
    seq_len, batch_size = input_batch.size()
    predictions = torch.zeros(input_batch.size())
    
    hidden = Variable(torch.zeros(1, self.hidden_size))
    for t in xrange(seq_len):
      word_ix = input_batch[t, :]
      w = self.embedding[word_ix.data, :]
      output, hidden = self.rnn(w, hidden)
      _, predictions[t,:] = output.data.topk(1) # get the index of the top item


    return predictions




# TODO: Your implementation goes here
class BiRNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(BiRNNLM, self).__init__()
    pass

  def forward(self, input_batch):
    pass

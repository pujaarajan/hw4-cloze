import torch
from torch import cuda
from torch.autograd import Variable
from LSTMmodel import BiLSTMLM
import dill

import utils.tensor
import utils.rand

def get_lm_input(data):
  input_data = []
  for sent in data:
    input_data.append(sent[:-1])
  return input_data


def get_lm_output(data):
  output_data = []
  for sent in data:
    output_data.append(sent[1:])
  return output_data

def main():

  _, _, test, vocab = torch.load(open("data/hw4_data.bin", 'rb'), pickle_module=dill)
  use_cuda = False

  test_in = get_lm_input(test)
  test_out = get_lm_output(test)

  batched_test_in, batched_test_in_mask, _ = utils.tensor.advanced_batchize(test_in, 1, vocab.stoi["<pad>"])
  batched_test_out, batched_test_out_mask, _ = utils.tensor.advanced_batchize(test_out, 1, vocab.stoi["<pad>"])

  vocab_size = len(vocab)
  bilstmlm = torch.load(open("LSTMmodel.py.nll_3.24.epoch_2", 'rb'), pickle_module=dill)


  for i, batch_i in enumerate(utils.rand.srange(len(batched_test_in))):
      test_in_batch = Variable(batched_test_in[batch_i])  # of size (seq_len, batch_size)
      test_out_batch = Variable(batched_test_out[batch_i])  # of size (seq_len, batch_size)
      test_in_mask = Variable(batched_test_in_mask[batch_i])
      test_out_mask = Variable(batched_test_out_mask[batch_i])

      sys_out_batch = bilstmlm(test_in_batch)  # (seq_len, batch_size, vocab_size) # TODO: substitute this with your module
      test_in_mask = test_in_mask.view(-1)
      test_in_mask = test_in_mask.unsqueeze(1).expand(len(test_in_mask), vocab_size)
      test_out_mask = test_out_mask.view(-1)
      sys_out_batch = sys_out_batch.view(-1, vocab_size)
      test_out_batch = test_out_batch.view(-1)
      sys_out_batch = sys_out_batch.masked_select(test_in_mask).view(-1, vocab_size)
      test_out_batch = test_out_batch.masked_select(test_out_mask)

      print(test_out_batch)
      


if __name__ == "__main__":
	main()
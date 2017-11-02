import torch
from torch import cuda
from torch.autograd import Variable
from LSTMmodel import BiLSTMLM
import dill
import numpy as np
import utils.tensor
import utils.rand

def embedding_to_word(embedding, vocab):
    return ' '.join(vocab.itos[i] for i in embedding.data)

_, _, test, vocab = torch.load(open("data/hw4_data.bin", 'rb'), pickle_module=dill)

batched_test, batched_test_mask = utils.tensor.advanced_batchize_no_sort(test, 1, vocab.stoi["<pad>"])
#batched_test, batched_test_mask, _ = utils.tensor.advanced_batchize(test, 1, vocab.stoi["<pad>"])

vocab_size = len(vocab)

bilstmlm = torch.load(open("LSTMmodel.py.nll_3.24.epoch_2", 'rb'), pickle_module=dill)

bilstmlm.cpu()

bilstmlm.eval()
m = 0
for line in test:
    #print(m, file=sys.stderr)
    m += 1
    blanks = []
    for i in range(len(line)):
        if vocab.itos[line[i]] == '<blank>':
            blanks.append(i)
    # print(blanks)
    # print(line)
    test_in = Variable(line).unsqueeze(1)
    test_out = bilstmlm(test_in)
    test_out = test_out.view(-1, vocab_size)
    cur = []
    newCur = []
    for i in blanks:
        # if vocab.itos[line[i]] == '<blank>':
            # print(test_out[i][1:])
        _, argmax = torch.max(test_out[i][1:], 0)
        newCur.append(argmax.data[0] + 1)
    count = 0
    while newCur != cur and count < 20:
        # print(7)
        count += 1
        # print(cur, newCur)
        cur = newCur
        newCur = []
        # print(cur)
        for j, i in enumerate(blanks):
            # print(j, i, blanks)
            line[i] = cur[j]
        test_in = Variable(line).unsqueeze(1)
        test_out = bilstmlm(test_in)
        test_out = test_out.view(-1, vocab_size)
        for i in blanks:
            # if vocab.itos[line[i]] == '<blank>':
                # print(test_out[i][1:])
            _, argmax = torch.max(test_out[i][1:], 0)
            newCur.append(argmax.data[0] + 1)
        # print(line)
    cur = []
    for val in newCur:
        cur.append(vocab.itos[val])
    print(' '.join(cur).encode('utf-8').strip())
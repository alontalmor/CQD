import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as weight_init
from config import *


class AttnDecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size , n_layers=1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = config.dropout_p
        self.max_length = config.MAX_LENGTH

        self.embedding = nn.Embedding(self.output_size + self.max_length, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size * 2, self.max_length)
        self.pgen_linear = nn.Linear(self.hidden_size * 3, 1)
        self.attn_combine = nn.Linear(self.hidden_size * 2 , self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.GRU = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, decoder_input, decoder_hidden, encoder_output, encoder_hiddens, encoder_hidden, output_mask=None):
        # input is actually the output of the decoder from t-1
        # hidden is the decoder hidden state

        # reembeding the output of the decoder (which is an index)
        decoder_input_embedded = self.embedding(decoder_input).view(1, 1, -1)
        # andomly zeroes some of the elements of the input tensor

        decoder_input_embedded = self.dropout(decoder_input_embedded)

        # concating the hidden and previous output and applying softmax to compute the attention weights
        # |embedded + hidden| =  2 * hidden_size
        # |attn_weights|  = input_size
        # attn() is a linear layer with weights W
        attn_weights = self.linear(torch.cat((encoder_hidden[0], decoder_hidden[0]), 1))

        # multiplying all attention weights with the encoder output respectively
        # |encoder_hidden| = input_size x hidden_size , |attn_weights| = input_size
        # |attn_applied| = hidden_size
        # NOTE: in "Data Recombination for Neural Semantic Parsing"
        # encoder_hidden is actually Bi (they use a bidirectional Encoder) Bi = concat(HiF,HiB)
        attn_applied = torch.bmm(F.softmax(attn_weights).unsqueeze(0),
                                 encoder_hiddens.unsqueeze(0))


        # concating the decoder output of t-1, with the attention.  |output| = 2 * hidden_size
        output = torch.cat((decoder_input_embedded[0], attn_applied[0]), 1)
        # linear layer of the decoder output from t-1 concatinated with attention. |output| = hidden_size
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, decoder_hidden = self.GRU(output, decoder_hidden)

        # computing generation probability
        # NOTE!! we should use decoder input not decoder_input_embedded
        Pgen = self.sigmoid(self.pgen_linear(torch.cat((attn_applied[0], decoder_hidden[0] , decoder_input_embedded[0]), 1)))


        #if output_mask is not None:
        #    input_dist = F.softmax(attn_weights[0] - output_mask[0:self.max_length])
        #    vocab_dist = F.softmax(self.out(output[0]) - output_mask[self.max_length:])
        #else:
        input_dist = F.softmax(attn_weights)
        vocab_dist = F.softmax(self.out(output[0]))

        #if output_mask is not None and output_mask[0:self.max_length].data.mean()==1000:
        #    softmax_output = torch.cat((Variable(torch.ones(self.max_length), requires_grad = False).view(1,-1) * -100, \
        #                                torch.log(vocab_dist)), 1)
        #elif output_mask is not None and output_mask[self.max_length:].data.mean()==1000:
        #    softmax_output = torch.cat((torch.log(input_dist) , \
        #                                Variable(torch.ones(self.output_size), requires_grad = False).view(1,-1) * -100), 1)
        #else:
        output_distribution = torch.cat(((1 - Pgen) * input_dist , Pgen * vocab_dist) , 1)

        #output_distribution.register_hook(print)
        #print(output_distribution.data[0].min())
        uniform = Variable(torch.FloatTensor(np.ones(self.output_size + self.max_length) / \
                                             (self.output_size + self.max_length)), requires_grad=False)
        output_distribution = output_distribution * (1 - config.epsilon_greedy) + uniform * config.epsilon_greedy
        softmax_output = torch.log(output_distribution)

        # WRONG - experimenting. ..
        #softmax_output = F.log_softmax(torch.cat(((1 - Pgen) * attn_weights, Pgen * self.out(output[0])), 1))

        return softmax_output, decoder_hidden, output_distribution

    def initHidden(self):
        # initialize hidden layer with zeros
        result = Variable(weight_init.xavier_normal(torch.Tensor(1, 1, self.hidden_size)))
        #result = Variable(torch.zeros(1, 1, self.hidden_size))
        if config.use_cuda:
            return result.cuda()
        else:
            return result


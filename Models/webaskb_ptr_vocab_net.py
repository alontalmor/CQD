
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from Models.Pytorch.encoder import EncoderRNN
from Models.Pytorch.abisee17_ptr_vocab_decoder import AttnDecoderRNN
from config import config

class WebAsKB_PtrVocabNet_Model():
    def __init__(self , input_lang, output_lang):

        self.input_lang = input_lang
        self.output_lang = output_lang

        if config.LOAD_SAVED_MODEL:
            self.encoder = torch.load(config.neural_model_dir  + 'encoder.pkl')
            self.decoder = torch.load(config.neural_model_dir  + 'decoder.pkl')
        else:
            self.encoder = EncoderRNN(input_lang.n_words, config.hidden_size)
            self.decoder = AttnDecoderRNN(output_lang.n_words, config.hidden_size)

        self.criterion = nn.NLLLoss()
        #self.criterion = nn.CrossEntropyLoss()

    def init_stats(self):
        self.avg_exact_token_match = 0
        self.exact_match = 0
        self.comp_accuracy = 0
        self.avg_one_tol_token_match = 0
        self.exact_match_one_tol = 0
        self.p1_accuracy = 0
        self.p2_accuracy = 0
        self.p1_1_right_accuracy = 0
        self.p1_1_left_accuracy = 0

    def init_optimizers(self):
        self.encoder_optimizer = optim.Adagrad(self.encoder.parameters(), lr=config.LR, lr_decay=config.ADA_GRAD_LR_DECAY,
                                          weight_decay=config.ADA_GRAD_L2)
        self.decoder_optimizer = optim.Adagrad(self.decoder.parameters(), lr=config.LR, lr_decay=config.ADA_GRAD_LR_DECAY,
                                          weight_decay=config.ADA_GRAD_L2)

    def optimizer_step(self):
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

    def evaluate_accuracy(self, target_variable, result, aux_data):
        accuracy = 0
        if config.use_cuda:
            delta = [abs(target_variable.cpu().view(-1).data.numpy()[i] - result[i]) for i in range(len(result))]
        else:
            delta = [abs(target_variable.view(-1).data.numpy()[i] - result[i]) for i in range(len(result))]

        if delta[0] == 0:
            accuracy += 0.4

        accuracy += ((pd.Series(delta[1:]) == 0) * 1.0).mean() * 0.6
        accuracy += ((pd.Series(delta[1:]) == 1) * 1.0).mean() * 0.3
        accuracy += ((pd.Series(delta[1:]) == 2) * 1.0).mean() * 0.1

        abs_delta_array = np.abs(np.array(delta))
        self.exact_match += (np.mean((abs_delta_array == 0) * 1.0) == 1.0) * 1.0

        if config.use_cuda:
            target = target_variable.cpu().view(-1).data.numpy()
        else:
            target = target_variable.view(-1).data.numpy()

        if target[0] == result[0]:
            self.comp_accuracy += 1



        return accuracy

    def calc_output_mask(self, input_variable, result):
        output_lang = self.output_lang
        output_mask = Variable(torch.ones(config.MAX_LENGTH + output_lang.n_words), requires_grad = False)

        # comp or cong
        if self.out_mask_state == 0:
            if len(result)>0:
                self.out_mask_state += 1
            else:
                output_mask[output_lang.word2index['Comp('] + config.MAX_LENGTH] = 0
                output_mask[output_lang.word2index['Conj('] + config.MAX_LENGTH] = 0


        # split1
        if self.out_mask_state == 1:
            if result[-1] == output_lang.word2index[','] + config.MAX_LENGTH and len(result)>2:
                self.out_mask_state += 1
            else:
                output_mask[0:len(input_variable)-1] = 0
                output_mask[output_lang.word2index[','] + config.MAX_LENGTH] = 0
        # split2
        if self.out_mask_state == 2:
            #if result[-1] == output_lang.word2index[')'] + config.MAX_LENGTH and \
            #        result[-2] != output_lang.word2index[','] + config.MAX_LENGTH:
            #    self.out_mask_state += 1
            #else:
            output_mask[0:len(input_variable)-1] = 0
            output_mask[output_lang.word2index['%composition'] + config.MAX_LENGTH] = 0
            output_mask[output_lang.word2index[')'] + config.MAX_LENGTH] = 0

        self.output_mask = output_mask

        return output_mask


    def calc_detailed_stats(self, sample_size):

        comp_accuracy_avg = self.comp_accuracy / sample_size
        p1_accuracy_avg = self.p1_accuracy / sample_size
        p2_accuracy_avg = self.p2_accuracy / sample_size
        p1_1_right_accuracy_avg = self.p1_1_right_accuracy / sample_size
        p1_1_left_accuracy_avg = self.p1_1_left_accuracy / sample_size

        return {'exact_match':self.exact_match / sample_size, \
                    'comp_accuracy':comp_accuracy_avg}

        #print('avg_exact_token_match %.4f' % (self.avg_exact_token_match / sample_size))
        #print('exact_match %.4f' % (self.exact_match / sample_size))
        #print('avg_one_tol_token_match %.4f' % (self.avg_one_tol_token_match / sample_size))
        #print('exact_match_one_tol %.4f' % (self.exact_match_one_tol / sample_size))

        #print('comp_accuracy %.4f' % (comp_accuracy_avg))
        #print('p1_accuracy %.4f' % (p1_accuracy_avg))
        #print('p2_accuracy %.4f' % (p2_accuracy_avg))
        #print('p1_1_right_accuracy %.4f' % (p1_1_right_accuracy_avg))
        #print('p1_1_left_accuracy %.4f' % (p1_1_left_accuracy_avg))

    def format_model_output(self,pairs_dev, model_out_seq):
        output_lang = self.output_lang
        input_tokens = [token['dependentGloss'] for token in pairs_dev['aux_data']['sorted_annotations']]

        if len(model_out_seq)==0:
            raise Exception('format_model_output_error', 'zero len output')

        if model_out_seq[0] == output_lang.word2index['Comp(']+config.MAX_LENGTH:
            comp = 'composition'
        elif model_out_seq[0] == output_lang.word2index['Conj(']+config.MAX_LENGTH:
            comp = 'conjunction'
        else:
            raise Exception('format_model_output_error', 'bad compositionality type')

        output_sup = pairs_dev['y'].view(-1).data.numpy()
        if len(output_sup) > 0:
            if output_lang.index2word[output_sup[0]-config.MAX_LENGTH] == 'Comp(':
                comp_sup = 'composition'
            else:
                comp_sup = 'conjunction'
        else:
            comp_sup = ''

        p1_sup = pairs_dev['aux_data']['p1']
        p2_sup = pairs_dev['aux_data']['p2']

        out_pos = 1
        split_part1_tokens = []
        while out_pos<len(model_out_seq) and model_out_seq[out_pos] != output_lang.word2index[','] + config.MAX_LENGTH:
            if model_out_seq[out_pos] >= len(input_tokens):
                raise Exception('format_model_output_error', 'illigal value - split1')
            split_part1_tokens.append(input_tokens[model_out_seq[out_pos]])
            out_pos+=1

        # skip the ','
        out_pos += 1

        split_part2_tokens = []
        while out_pos<len(model_out_seq) and model_out_seq[out_pos] != output_lang.word2index[')'] + config.MAX_LENGTH:
            if comp == 'composition' and model_out_seq[out_pos] == output_lang.word2index['%composition'] + config.MAX_LENGTH:
                split_part2_tokens.append('%composition')
            else:
                if model_out_seq[out_pos] >= len(input_tokens):
                    raise Exception('format_model_output_error', 'illigal value - split2')
                split_part2_tokens.append(input_tokens[model_out_seq[out_pos]])
            out_pos+=1

        if len(split_part1_tokens) == 0:
            raise Exception('format_model_output_error', 'split1 len 0')

        if len(split_part2_tokens) == 0:
            raise Exception('format_model_output_error', 'split2 len 0')

        # exactly one %composition if composition question
        if comp == 'composition' and ((pd.Series(split_part2_tokens) == '%composition') * 1.0).sum() != 1:
            raise Exception('format_model_output_error', 'no %composition in split2')

        return [{'ID': pairs_dev['aux_data']['ID'], 'comp': comp, 'comp_sup': comp_sup,
                           'same_comp': int(comp == comp_sup), 'p1_sup': p1_sup, \
                           'p2_sup': p2_sup, 'split_part1': ' '.join(split_part1_tokens), \
                           'split_part2': ' '.join(split_part2_tokens),
                           'question': pairs_dev['aux_data']['question'], \
                           'answers': pairs_dev['aux_data']['answers']}]

    def save_model(self):
        torch.save(self.encoder, config.neural_model_dir + 'encoder.pkl')
        torch.save(self.decoder, config.neural_model_dir + 'decoder.pkl')

    def forward(self,input_variable, target_variable, loss=0, DO_TECHER_FORCING=False):
        encoder_hidden = self.encoder.initHidden()

        input_length = len(input_variable)
        target_length = len(target_variable)

        encoder_outputs = Variable(torch.zeros(config.MAX_LENGTH, self.encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if config.use_cuda else encoder_outputs

        encoder_hiddens = Variable(torch.zeros(config.MAX_LENGTH, self.encoder.hidden_size))
        encoder_hiddens = encoder_hiddens.cuda() if config.use_cuda else encoder_hiddens

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_variable[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]
            encoder_hiddens[ei] = encoder_hidden[0][0]

        decoder_input = Variable(torch.LongTensor([[config.SOS_token]]))
        decoder_input = decoder_input.cuda() if config.use_cuda else decoder_input

        decoder_hidden = encoder_hidden
        result = []
        # Without teacher forcing: use its own predictions as the next input
        sub_optimal_chosen = False
        output_mask = None
        self.out_mask_state = 0

        for di in range(len(target_variable)):
            if config.use_output_masking:
                output_mask = self.calc_output_mask(input_variable,result)

            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_hidden, encoder_hiddens, encoder_hidden, output_mask)

            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            if DO_TECHER_FORCING:
                decoder_input = target_variable[di]
            else:
                decoder_input = Variable(torch.LongTensor([[int(np.argmax(decoder_attention.data[0].tolist()))]]))

            # we are computing logistical regression vs the hidden layer!!
            if len(target_variable)>0:
                loss += self.criterion(decoder_output, target_variable[di])

            result.append(np.argmax(decoder_output.data[0].tolist()))

        if type(loss)!=int:
            loss_value = loss.data[0] / target_length
        else:
            loss_value = 0
        return loss_value , result, loss
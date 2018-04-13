import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from Models.Pytorch.encoder import EncoderRNN
from Models.Pytorch.abisee17_ptr_vocab_decoder import AttnDecoderRNN
from config import *

class WebAsKB_PtrVocabNet_Model():
    def __init__(self , input_lang, output_lang, criterion=None):

        self.output_lang = output_lang
        self.input_lang = input_lang

        if config.LOAD_SAVED_MODEL:
            self.encoder = config.load_pytorch_model(config.neural_model_dir + config.input_model, 'encoder')
            self.decoder = config.load_pytorch_model(config.neural_model_dir + config.input_model, 'decoder')
        else:
            self.encoder = EncoderRNN(input_lang.n_words, config.hidden_size)
            self.decoder = AttnDecoderRNN(output_lang.n_words, config.hidden_size)

        #if criterion is None:
        self.criterion = nn.NLLLoss()
        #else:
        #    self.criterion = criterion
        #self.criterion = nn.CrossEntropyLoss()

        # model expressivness, used in Masking, and RL sampling.
        self.exp = {'Skip':0}

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
        #if config.use_cuda:
        #    delta = [abs(target_variable.cpu().view(-1).data.numpy()[i] - result[i]) for i in range(len(result))]
        #else:
        delta_seq = [abs(target_variable.view(-1).data.numpy()[i] - result[i]) for i in range(len(result))]
        abs_delta_seq_array = np.abs(np.array(delta_seq))
        exact_matchseq = (np.mean((abs_delta_seq_array == 0) * 1.0) == 1.0) * 1.0

        delta = []
        delta.append(abs(target_variable.view(-1).data.numpy()[0] - result[0]))
        delta.append(abs(self.mask_state['P1'] - aux_data['p1']))
        if self.mask_state['P2'] is not None:
            delta.append(abs(self.mask_state['P2'] - aux_data['p2']))
        else:
            if aux_data['p2'] == 0:
                delta.append(0.0)
            else:
                delta.append(100.0)

        if delta[0] == 0:
            accuracy += 0.4
        if len(delta) > 1:
            if delta[1] == 0:
                accuracy += 0.3
            if delta[1] == 1:
                accuracy += 0.15
            if delta[1] == 2:
                accuracy += 0.05
            if delta[2] == 0:
                accuracy += 0.3
            if delta[2] == 1:
                accuracy += 0.15
            if delta[2] == 2:
                accuracy += 0.05

        abs_delta_array = np.abs(np.array(delta))
        self.avg_exact_token_match += np.mean((abs_delta_array == 0) * 1.0)
        self.avg_one_tol_token_match += ((abs_delta_array[0] == 0) * 1.0 + np.sum((abs_delta_array[1:] <= 1) * 1.0)) / 3.0
        self.exact_match += (np.mean((abs_delta_array == 0) * 1.0) == 1.0) * 1.0
        self.exact_match_one_tol += ((abs_delta_array[0] == 0) & (np.mean((abs_delta_array <= 1) * 1.0) == 1.0)) * 1.0

        if config.use_cuda:
            target = target_variable.cpu().view(-1).data.numpy()
        else:
            target = target_variable.view(-1).data.numpy()

        if target[0] == result[0]:
            self.comp_accuracy += 1
        if len(delta) > 1:
            if target[1] == result[1]:
                self.p1_accuracy += 1
            if target[1] == result[1] - 1:
                self.p1_1_right_accuracy += 1
            if target[1] == result[1] + 1:
                self.p1_1_left_accuracy += 1
            if target[2] == result[2]:
                self.p2_accuracy += 1

        #accuracy += ((pd.Series(delta[1:]) == 0) * 1.0).mean() * 0.6
        #accuracy += ((pd.Series(delta[1:]) == 1) * 1.0).mean() * 0.3
        #accuracy += ((pd.Series(delta[1:]) == 2) * 1.0).mean() * 0.1

        #abs_delta_array = np.abs(np.array(delta))
        #self.exact_match += (np.mean((abs_delta_array == 0) * 1.0) == 1.0) * 1.0





        return accuracy

    def vocab_ind_to_word(self,ind):
        if ind - config.MAX_LENGTH >= 0:
            return self.output_lang.index2word[ind - config.MAX_LENGTH]
        else:
            return ind

    def vocab_word_to_ind(self,word):
        return self.output_lang.word2index[word] + config.MAX_LENGTH

    def calc_detailed_stats(self, sample_size):

        comp_accuracy_avg = self.comp_accuracy / sample_size
        p1_accuracy_avg = self.p1_accuracy / sample_size
        p2_accuracy_avg = self.p2_accuracy / sample_size
        p1_1_right_accuracy_avg = self.p1_1_right_accuracy / sample_size
        p1_1_left_accuracy_avg = self.p1_1_left_accuracy / sample_size

        return {'exact_match':self.exact_match / sample_size, \
                    'comp_accuracy':comp_accuracy_avg , \
                'avg_exact_token_match': self.avg_exact_token_match / sample_size, \
                'avg_one_tol_token_match': self.avg_one_tol_token_match / sample_size, \
                'exact_match_one_tol': self.exact_match_one_tol / sample_size, \
                'p1_accuracy': p1_accuracy_avg, \
                'p2_accuracy:': p2_accuracy_avg, \
                'p1_1_right_accuracy': p1_1_right_accuracy_avg, \
                'p1_1_left_accuracy': p1_1_left_accuracy_avg}

        #print('avg_exact_token_match %.4f' % (self.avg_exact_token_match / sample_size))
        #print('exact_match %.4f' % (self.exact_match / sample_size))
        #print('avg_one_tol_token_match %.4f' % (self.avg_one_tol_token_match / sample_size))
        #print('exact_match_one_tol %.4f' % (self.exact_match_one_tol / sample_size))

        #print('comp_accuracy %.4f' % (comp_accuracy_avg))
        #print('p1_accuracy %.4f' % (p1_accuracy_avg))
        #print('p2_accuracy %.4f' % (p2_accuracy_avg))
        #print('p1_1_right_accuracy %.4f' % (p1_1_right_accuracy_avg))
        #print('p1_1_left_accuracy %.4f' % (p1_1_left_accuracy_avg))

    def calc_output_mask(self, input_variable, result):
        output_lang = self.output_lang
        output_mask = torch.zeros(config.MAX_LENGTH + output_lang.n_words)
        input_len = len(input_variable) - 1

        try:
            # comp or cong
            if self.out_mask_state == 0:
                if len(result) > 0:
                    self.out_mask_state += 1
                else:
                    self.mask_state = {'P1':None,'P2':None}
                    output_mask[self.vocab_word_to_ind('Comp(')] = 1
                    output_mask[self.vocab_word_to_ind('Conj(')] = 1

            ##### Split1 #######
            if self.out_mask_state == 1:
                if result[-1] == output_lang.word2index[','] + config.MAX_LENGTH and len(result) > 2:
                    self.out_mask_state += 1
                    if self.mask_state['comp'] == 'Conjunction':
                        self.mask_state['P1'] = result[-2]
                    else:
                        self.mask_state['P2'] = result[-2]
                else:
                    #### Model chose Conjunction
                    if self.vocab_ind_to_word(result[-1]) == 'Conj(':
                        self.mask_state['comp'] = 'Conjunction'
                        output_mask[0:min(1 + config.skip_limit, input_len)] = 1
                    ### Model chose Composition
                    elif self.vocab_ind_to_word(result[-1]) == 'Comp(':
                        self.mask_state['comp'] = 'Composition'
                        output_mask[0:input_len] = 1
                    ### Split1, pos > 2
                    else:

                        if self.mask_state['comp'] == 'Composition':
                            # Storing P1 value
                            if self.vocab_ind_to_word(result[-2]) == 'Comp(':
                                self.mask_state['P1'] = result[-1]
                            # Only subsequent tokens allowed
                            if result[-1] < input_len - 1:
                                output_mask[result[-1] + 1: min(result[-1] + 2 + config.skip_limit, input_len)] = 1
                        else:
                            # we need at least one word in split2, so break before len-1
                            if result[-1] < input_len - 2:
                                output_mask[result[-1] + 1: min(result[-1] + 2 + config.skip_limit, input_len - 1)] = 1

                        output_mask[self.vocab_word_to_ind(',')] = 1

            ##### Split2 #######
            if self.out_mask_state == 2:
                ########### Composition ##############
                if self.mask_state['comp'] == 'Composition':
                    if self.vocab_ind_to_word(result[-1]) == ',':
                        if self.mask_state['P1'] > 0:
                            output_mask[0:min(1 + config.skip_limit, self.mask_state['P1'])] = 1
                        else:
                            output_mask[self.vocab_word_to_ind('%composition')] = 1
                    elif self.vocab_ind_to_word(result[-1]) == '%composition':
                        if self.mask_state['P2'] >= input_len - 1:
                            output_mask[self.vocab_word_to_ind(')')] = 1
                        else:
                            output_mask[self.mask_state['P2'] + 1] = 1
                    else:
                        if result[-1] == self.mask_state['P1'] - 1:
                            output_mask[self.vocab_word_to_ind('%composition')] = 1
                        else:
                            if result[-1] >= input_len - 1:
                                output_mask[self.vocab_word_to_ind(')')] = 1
                            else:
                                output_mask[result[-1] + 1 : min(result[-1] + 2 + config.skip_limit, input_len)] = 1
                ########### Conjunction ##############
                else:
                    # conjucntion "P2"
                    if self.vocab_ind_to_word(result[-1]) == ',':
                        # all previous split tokens OR first token unused
                        output_mask[1: self.mask_state['P1'] + 2] = 1
                    else:
                        # P2 used:
                        if result[-1] <= self.mask_state['P1']:
                            output_mask[self.mask_state['P1'] + 1] = 1
                            self.mask_state['P2'] = result[-1]
                        else:
                            if result[-1] >= input_len - 1:
                                output_mask[self.vocab_word_to_ind(')')] = 1
                            else:
                                output_mask[result[-1] + 1 : min(result[-1] + 2 + config.skip_limit, input_len)] = 1
        except:
            config.write_log('ERROR', "build mask exception", {'error_message': traceback.format_exc()})

        self.output_mask = output_mask

        return output_mask

    def format_model_output(self, pairs_dev, in_model_out_seq, output_dists, output_masks, skip_ind=None):
        model_out_seq = in_model_out_seq.copy()
        output_lang = self.output_lang
        input_tokens = [token['dependentGloss'] for token in pairs_dev['aux_data']['sorted_annotations']]
        pointer_ind = []
        seq2seq_output = []


        if len(model_out_seq)==0:
            raise Exception('format_model_output_error', 'zero len output')

        # RL trajectory generation - PATCH (we should generate trajectories by using P(W) not by skipping
        # words after sequence has been generated.
        skip_ind_list = []
        skip_token_list = []
        if config.gen_trajectories:
            if skip_ind is None:
                # randomaly chosing skips
                while len(skip_ind_list) < config.skip_limit:
                    skip_ind = random.randint(0, len(model_out_seq) - 1)
                    if model_out_seq[skip_ind] < len(input_tokens):
                        skip_ind_list.append(skip_ind)
                        skip_token_list.append(input_tokens[model_out_seq[skip_ind]])
                        del model_out_seq[skip_ind]
            else:
                # skips are given
                skip_ind_list.append(skip_ind)
                skip_token_list.append(input_tokens[model_out_seq[skip_ind]])
                del model_out_seq[skip_ind]

        if model_out_seq[0] == output_lang.word2index['Comp(']+config.MAX_LENGTH:
            comp = 'composition'
            pointer_ind.append(None)
            seq2seq_output.append('Comp(')
        elif model_out_seq[0] == output_lang.word2index['Conj(']+config.MAX_LENGTH:
            comp = 'conjunction'
            pointer_ind.append(None)
            seq2seq_output.append('Conj(')
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
            pointer_ind.append(model_out_seq[out_pos])
            seq2seq_output.append('Copy')
            out_pos+=1

        # skip the ','
        out_pos += 1
        pointer_ind.append(None)
        seq2seq_output.append(',')

        split_part2_tokens = []
        while out_pos<len(model_out_seq) and model_out_seq[out_pos] != output_lang.word2index[')'] + config.MAX_LENGTH:
            if comp == 'composition' and model_out_seq[out_pos] == output_lang.word2index['%composition'] + config.MAX_LENGTH:
                split_part2_tokens.append('%composition')
                pointer_ind.append(None)
                seq2seq_output.append('%composition')
            else:
                if model_out_seq[out_pos] >= len(input_tokens):
                    raise Exception('format_model_output_error', 'illigal value - split2')
                split_part2_tokens.append(input_tokens[model_out_seq[out_pos]])
                pointer_ind.append(model_out_seq[out_pos])
                seq2seq_output.append('Copy')

                ### PATCH !!!!
                #if comp == 'composition' and model_out_seq[out_pos - 1] == output_lang.word2index[
                #    '%composition'] + config.MAX_LENGTH:
                #    split_part1_tokens.append(split_part2_tokens[-1])
                #    split_part2_tokens = split_part2_tokens[0:-1]
            out_pos+=1

        pointer_ind.append(None)
        seq2seq_output.append(')')

        if len(split_part1_tokens) == 0:
            raise Exception('format_model_output_error', 'split1 len 0')

        if len(split_part2_tokens) == 0:
            raise Exception('format_model_output_error', 'split2 len 0')

        # exactly one %composition if composition question
        if comp == 'composition' and ((pd.Series(split_part2_tokens) == '%composition') * 1.0).sum() != 1:
            raise Exception('format_model_output_error', 'no %composition in split2')


        output = [{'ID': pairs_dev['aux_data']['ID'], 'comp': comp, 'comp_sup': comp_sup,
                    'same_comp': int(comp == comp_sup),\
                    'p1':self.mask_state['P1'], 'p1_sup': p1_sup, \
                    'p2':self.mask_state['P2'], 'p2_sup': p2_sup, \
                    'split_part1': ' '.join(split_part1_tokens), \
                    'split_part2': ' '.join(split_part2_tokens),
                    'question': pairs_dev['aux_data']['question'], \
                    'answers': pairs_dev['aux_data']['answers'], \
                    'sorted_annotations': pairs_dev['aux_data']['sorted_annotations'], \
                    'pointer_ind': pointer_ind, 'seq2seq_output': seq2seq_output, \
                    'output_seq': model_out_seq, 'skip_ind_list':skip_ind_list,\
                    'skip_token_list':skip_token_list}]

        if config.SAVE_DISTRIBUTIONS:
            output[0].update({'output_dists': output_dists,\
                    'output_masks': output_masks})

        return output

    def save_model(self, name):
        config.store_pytorch_model(self.encoder, config.neural_model_dir + config.out_subdir, 'encoder' + name)
        config.store_pytorch_model(self.decoder, config.neural_model_dir + config.out_subdir, 'decoder' + name)

    def forward(self,input_variable, target_variable, reward = 0, loss=0,  DO_TECHER_FORCING=False):
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
        output_masks = []
        output_dists = []
        # Without teacher forcing: use its own predictions as the next input
        sub_optimal_chosen = False
        output_mask = None
        output_dist = None
        self.out_mask_state = 0

        for di in range(len(target_variable)):
            if config.use_output_masking:
                output_mask = self.calc_output_mask(input_variable,result)

            decoder_output, decoder_hidden, output_dist = self.decoder(
                decoder_input, decoder_hidden, encoder_hidden, encoder_hiddens, encoder_hidden, output_mask)

            if config.RL_Training:
                loss += self.criterion(decoder_output, target_variable[di]) * reward
            else:
                loss += self.criterion(decoder_output, target_variable[di])

            if config.use_output_masking:
                if config.sample_output_dist:
                    masked_distribution = output_dist.data[0].numpy() * output_mask.numpy()
                    masked_distribution /= masked_distribution.sum()
                    curr_output = int(np.random.choice(len(masked_distribution), 1, p=masked_distribution)[0])
                else:
                    curr_output = np.argmax(decoder_output.data - ((output_mask == 0).float() * 1000))
            else:
                curr_output = np.argmax(decoder_output.data)

            if DO_TECHER_FORCING:
                decoder_input = target_variable[di]
            else:
                decoder_input = Variable(torch.LongTensor([curr_output]))

            result.append(curr_output)
            output_masks.append(output_mask.int().tolist())
            output_dists.append((output_dist.data[0] * 100).round().int().tolist())

            ## EOS
            if self.vocab_ind_to_word(curr_output) == ')':
                break

        if type(loss)!=int:
            loss_value = loss.data[0] / target_length
        else:
            loss_value = 0
        return loss_value , result, loss, output_dists, output_masks
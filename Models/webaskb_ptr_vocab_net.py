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
            self.encoder = config.load_pytorch_model(config.neural_model_dir + config.modeldir, 'encoder')
            self.decoder = config.load_pytorch_model(config.neural_model_dir + config.modeldir, 'decoder')
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

        if config.beam_search_gen:
            self.forward_func = self.beam_search_forward
        elif config.sample_output_dist:
            self.forward_func = self.traj_sampling_forward
        else:
            self.forward_func = self.forward

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

    def evaluate_accuracy_naacl18(self, target_variable, result, aux_data, mask_state):
        accuracy = 0
        #if config.use_cuda:
        #    delta = [abs(target_variable.cpu().view(-1).data.numpy()[i] - result[i]) for i in range(len(result))]
        #else:
        delta_seq = [abs(target_variable.view(-1).data.numpy()[i] - result[i]) for i in range(min(len(target_variable),len(result)))]
        abs_delta_seq_array = np.abs(np.array(delta_seq))
        exact_matchseq = (np.mean((abs_delta_seq_array == 0) * 1.0) == 1.0) * 1.0

        delta = []
        delta.append(abs(target_variable.view(-1).data.numpy()[0] - result[0]))
        delta.append(abs(mask_state['P1'] - aux_data['p1']))
        if mask_state['P2'] is not None:
            delta.append(abs(mask_state['P2'] - aux_data['p2']))
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

        return accuracy

    def evaluate_accuracy(self, target_variable, result, aux_data, mask_state):
        delta_seq = [abs(target_variable.view(-1).data.numpy()[i] - result[i]) for i in
                     range(min(len(target_variable), len(result)))]
        accuracy = 0
        accuracy += ((pd.Series(delta_seq[0]) == 0) * 1.0).mean() * 0.4
        accuracy += ((pd.Series(delta_seq[1:]) == 0) * 1.0).mean() * 0.6
        accuracy += ((pd.Series(delta_seq[1:]) == 1) * 1.0).mean() * 0.3
        accuracy += ((pd.Series(delta_seq[1:]) == 2) * 1.0).mean() * 0.1

        abs_delta_array = np.abs(np.array(delta_seq))
        self.exact_match += (np.mean((abs_delta_array == 0) * 1.0) == 1.0) * 1.0

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

    def calc_output_mask(self, input_variable, program, mask_state):
        output_lang = self.output_lang
        output_mask = torch.zeros(config.MAX_LENGTH + output_lang.n_words)
        input_len = len(input_variable) - 1

        # Init
        if mask_state is None:
            mask_state = {'state':'init','split1_inds':[],'split2_inds':[], 'skip_map':np.zeros(input_len), \
                          'skip_limit':config.skip_limit,'%composition_found':False, 'comp':None , \
                          'skip_ind_list':[], 'skip_to_%composition':0}

        try:
            # comp or cong
            if mask_state['state'] == 'init':
                if len(program) > 0:
                    if self.vocab_ind_to_word(program[-1]) == 'Conj(':
                        mask_state['comp'] = 'Conjunction'
                    else:
                        mask_state['comp'] = 'Composition'
                    mask_state['state'] = 'split1'
                else:
                    output_mask[self.vocab_word_to_ind('Comp(')] = 1
                    output_mask[self.vocab_word_to_ind('Conj(')] = 1


            ##### Split1 #######
            if mask_state['state'] == 'split1':
                if self.vocab_ind_to_word(program[-1]) == ',':
                    mask_state['state'] = 'split2'
                else:
                    if self.vocab_ind_to_word(program[-1]) == 'Comp(':
                        output_mask[0: input_len] = 1
                    else:
                        if self.vocab_ind_to_word(program[-1]) == 'Conj(':
                            first_ind = 0
                        else:
                            mask_state['split1_inds'].append(program[-1])
                            first_ind = program[-1] + 1

                            # Update skip Limit
                            mask_state['skip_limit'] -= int(mask_state['skip_map'][program[-1]])
                            output_mask[self.vocab_word_to_ind(',')] = 1

                        if (mask_state['comp'] == 'Composition' and first_ind + mask_state['skip_limit'] + 1 <= input_len) or \
                           (mask_state['comp'] == 'Conjunction' and first_ind + mask_state['skip_limit'] + 1 <= input_len - 1):
                            last_ind = first_ind + 1 + mask_state['skip_limit']
                        else:
                            if mask_state['comp'] == 'Conjunction':
                                last_ind = input_len - 1
                            else:
                                last_ind = input_len

                        if last_ind > first_ind:
                            output_mask[first_ind: last_ind] = 1

                        mask_state['skip_map'] = np.zeros(input_len)
                        for count, ind in enumerate(range(first_ind, last_ind)):
                            mask_state['skip_map'][ind] = count

            ##### Split2 #######
            if mask_state['state'] == 'split2' and self.vocab_ind_to_word(program[-1]) != ')':
                ########### Composition ##############
                if mask_state['comp'] == 'Composition':
                    if self.vocab_ind_to_word(program[-1]) == '%composition':
                        mask_state['%composition_found'] = True

                    if not mask_state['%composition_found']:
                        if self.vocab_ind_to_word(program[-1]) == ',':
                            first_ind = 0
                        else:
                            mask_state['split2_inds'].append(program[-1])
                            first_ind = program[-1] + 1
                            # Update skip Limit
                            mask_state['skip_limit'] -= int(mask_state['skip_map'][program[-1]])

                        if first_ind + mask_state['skip_limit'] >= mask_state['split1_inds'][0]:
                            output_mask[self.vocab_word_to_ind('%composition')] = 1
                            mask_state['skip_to_%composition'] = mask_state['split1_inds'][0] - first_ind
                            last_ind = mask_state['split1_inds'][0]
                        else:
                            last_ind = first_ind + 1 +  mask_state['skip_limit']

                        if last_ind>first_ind:
                            output_mask[first_ind: last_ind] = 1

                        mask_state['skip_map'] = np.zeros(input_len)
                        for count, ind in enumerate(range(first_ind, last_ind)):
                            mask_state['skip_map'][ind] = count

                    else: # mask_state['%composition_found']

                        if self.vocab_ind_to_word(program[-1]) == '%composition':
                            first_ind = mask_state['split1_inds'][-1] + 1
                            mask_state['skip_limit'] -= int(mask_state['skip_to_%composition'])
                        else:
                            mask_state['split2_inds'].append(program[-1])
                            first_ind = program[-1] + 1

                            # Update skip Limit
                            mask_state['skip_limit'] -= int(mask_state['skip_map'][program[-1]])

                        if first_ind + mask_state['skip_limit'] + 1 <= input_len:
                            last_ind = first_ind + 1 +  mask_state['skip_limit']
                        else:
                            last_ind = input_len
                            output_mask[self.vocab_word_to_ind(')')] = 1
                        if last_ind>first_ind:
                            output_mask[first_ind: last_ind] = 1

                        mask_state['skip_map'] = np.zeros(input_len)
                        for count, ind in enumerate(range(first_ind, last_ind)):
                            mask_state['skip_map'][ind] = count

                if mask_state['comp'] == 'Conjunction':
                    if self.vocab_ind_to_word(program[-1]) == ',' or program[-1] < mask_state['split1_inds'][-1]:
                        # 1 copy
                        if not program[-1] < mask_state['split1_inds'][-1]:
                            output_mask[0:mask_state['split1_inds'][-1] + 1] = 1
                        first_ind = mask_state['split1_inds'][-1] + 1
                    else:
                        mask_state['split2_inds'].append(program[-1])
                        first_ind = program[-1] + 1

                        # Update skip Limit
                        mask_state['skip_limit'] -= int(mask_state['skip_map'][program[-1]])

                    if first_ind + mask_state['skip_limit'] + 1 <= input_len:
                        last_ind = first_ind + 1 + mask_state['skip_limit']
                    else:
                        last_ind = input_len
                        output_mask[self.vocab_word_to_ind(')')] = 1
                    if last_ind > first_ind:
                        output_mask[first_ind: last_ind] = 1

                    mask_state['skip_map'] = np.zeros(input_len)
                    for count, ind in enumerate(range(first_ind, last_ind)):
                        mask_state['skip_map'][ind] = count
        except:
            config.write_log('ERROR', "build mask exception", {'error_message': traceback.format_exc()})

        prog_series = pd.Series(program)
        validity = pd.Series(index=range(input_len))
        validity[:] = 0
        validity.update(prog_series[prog_series<input_len].value_counts())

        #if mask_state['comp'] == 'Conjunction' and ((validity>1)*1.0).sum()>1:
        #    assert()

        #if mask_state['comp'] == 'Composition' and ((validity>1)*1.0).sum()>0:
        #    assert ()

        #if len(program)>1 and output_mask[self.vocab_word_to_ind(')')] == 1 and \
        #                ((validity==0)*1.0).sum() > config.skip_limit:
        #    assert ()

        #if len(program)>1 and output_mask[self.vocab_word_to_ind(')')] == 1 and \
        #                ((validity>0)*1.0).sum() > input_len-config.skip_limit:
        #    assert ()
        if len(program) > 1 and program[-1] == self.vocab_word_to_ind(')'):
            mask_state['skip_ind_list'] += list(validity[validity==0].index)

        return output_mask, mask_state

    def calc_output_mask_naacl18(self, input_variable, program, mask_state):
        output_lang = self.output_lang
        output_mask = torch.zeros(config.MAX_LENGTH + output_lang.n_words)
        input_len = len(input_variable) - 1

        if mask_state is None:
            mask_state = {'P1': None, 'P2': None, 'state': 0}

        try:
            # comp or cong
            if mask_state['state'] == 0:
                if len(program) > 0:
                    mask_state['state'] += 1
                else:
                    output_mask[self.vocab_word_to_ind('Comp(')] = 1
                    output_mask[self.vocab_word_to_ind('Conj(')] = 1

            ##### Split1 #######
            if mask_state['state'] == 1:
                if program[-1] == output_lang.word2index[','] + config.MAX_LENGTH and len(program) > 2:
                    mask_state['state'] += 1
                    if mask_state['comp'] == 'Conjunction':
                        mask_state['P1'] = program[-2]
                    else:
                        mask_state['P2'] = program[-2]
                else:
                    #### Model chose Conjunction
                    if self.vocab_ind_to_word(program[-1]) == 'Conj(':
                        mask_state['comp'] = 'Conjunction'
                        output_mask[0] = 1
                    ### Model chose Composition
                    elif self.vocab_ind_to_word(program[-1]) == 'Comp(':
                        mask_state['comp'] = 'Composition'
                        output_mask[0:input_len] = 1
                    ### Split1, pos > 2
                    else:

                        if mask_state['comp'] == 'Composition':
                            # Storing P1 value
                            if self.vocab_ind_to_word(program[-2]) == 'Comp(':
                                mask_state['P1'] = program[-1]
                            # Only subsequent tokens allowed
                            if program[-1] < input_len - 1:
                                output_mask[program[-1] + 1] = 1
                        else:
                            # we need at least one word in split2, so break before len-1
                            if program[-1] < input_len - 2:
                                output_mask[program[-1] + 1] = 1

                        output_mask[self.vocab_word_to_ind(',')] = 1

            ##### Split2 #######
            if mask_state['state'] == 2:
                ########### Composition ##############
                if mask_state['comp'] == 'Composition':
                    if self.vocab_ind_to_word(program[-1]) == ',':
                        if mask_state['P1'] > 0:
                            output_mask[0] = 1
                        else:
                            output_mask[self.vocab_word_to_ind('%composition')] = 1
                    elif self.vocab_ind_to_word(program[-1]) == '%composition':
                        if mask_state['P2'] >= input_len - 1:
                            output_mask[self.vocab_word_to_ind(')')] = 1
                        else:
                            output_mask[mask_state['P2'] + 1] = 1
                    else:
                        if program[-1] == mask_state['P1'] - 1:
                            output_mask[self.vocab_word_to_ind('%composition')] = 1
                        else:
                            if program[-1] >= input_len - 1:
                                output_mask[self.vocab_word_to_ind(')')] = 1
                            else:
                                output_mask[program[-1] + 1] = 1
                ########### Conjunction ##############
                else:
                    # conjucntion "P2"
                    if self.vocab_ind_to_word(program[-1]) == ',':
                        # all previous split tokens OR first token unused
                        output_mask[1: mask_state['P1'] + 2] = 1
                    else:
                        # P2 used:
                        if program[-1] <= mask_state['P1']:
                            output_mask[mask_state['P1'] + 1] = 1
                            mask_state['P2'] = program[-1]
                        else:
                            if program[-1] >= input_len - 1:
                                output_mask[self.vocab_word_to_ind(')')] = 1
                            else:
                                output_mask[program[-1] + 1] = 1
        except:
            config.write_log('ERROR', "build mask exception", {'error_message': traceback.format_exc()})

        return output_mask, mask_state

    def format_model_output(self, pairs_dev, in_program, output_dists, output_masks, mask_state, model_prob , skip_ind=None):
        program = in_program.copy()
        program_tokens = []
        output_lang = self.output_lang
        input_tokens = [token['dependentGloss'] for token in pairs_dev['aux_data']['sorted_annotations']]


        if len(program)==0:
            raise Exception('format_model_output_error', 'zero len output')

        # RL trajectory generation - PATCH (we should generate trajectories by using P(W) not by skipping
        # words after sequence has been generated.
        #skip_ind_list = []
        skip_token_list = []
        #if False and config.gen_trajectories:
        #    if skip_ind is None:
        #        # randomaly chosing skips
        #        while len(skip_ind_list) < config.skip_limit:
        #            skip_ind = random.randint(0, len(program) - 1)
        #            if program[skip_ind] < len(input_tokens):
        #                skip_ind_list.append(skip_ind)
        #                skip_token_list.append(input_tokens[program[skip_ind]])
        #                del program[skip_ind]
        #    else:
        #        # skips are given
        #        skip_ind_list.append(skip_ind)
        #        skip_token_list.append(input_tokens[program[skip_ind]])
        #        del program[skip_ind]

        for skip_ind in mask_state['skip_ind_list']:
            skip_token_list.append(input_tokens[skip_ind])

        if program[0] == output_lang.word2index['Comp(']+config.MAX_LENGTH:
            comp = 'composition'
            program_tokens.append('Comp(')
        elif program[0] == output_lang.word2index['Conj(']+config.MAX_LENGTH:
            comp = 'conjunction'
            program_tokens.append('Conj(')
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

        if 'p1' in pairs_dev['aux_data']:
            p1_sup = pairs_dev['aux_data']['p1']
            p2_sup = pairs_dev['aux_data']['p2']

        out_pos = 1
        split_part1_tokens = []
        while out_pos<len(program) and program[out_pos] != output_lang.word2index[','] + config.MAX_LENGTH:
            if program[out_pos] >= len(input_tokens):
                raise Exception('format_model_output_error', 'illigal value - split1')
            split_part1_tokens.append(input_tokens[program[out_pos]])
            program_tokens.append(input_tokens[program[out_pos]])
            out_pos+=1

        # skip the ','
        out_pos += 1
        program_tokens.append(',')

        split_part2_tokens = []
        while out_pos<len(program) and program[out_pos] != output_lang.word2index[')'] + config.MAX_LENGTH:
            if comp == 'composition' and program[out_pos] == output_lang.word2index['%composition'] + config.MAX_LENGTH:
                split_part2_tokens.append('%composition')
                program_tokens.append('%composition')
            else:
                if program[out_pos] >= len(input_tokens):
                    raise Exception('format_model_output_error', 'illigal value - split2')
                split_part2_tokens.append(input_tokens[program[out_pos]])
                program_tokens.append(input_tokens[program[out_pos]])

                ### PATCH !!!!
                #if comp == 'composition' and program[out_pos - 1] == output_lang.word2index[
                #    '%composition'] + config.MAX_LENGTH:
                #    split_part1_tokens.append(split_part2_tokens[-1])
                #    split_part2_tokens = split_part2_tokens[0:-1]
            out_pos+=1

        program_tokens.append(')')
        
        if len(split_part1_tokens) == 0:
            raise Exception('format_model_output_error', 'split1 len 0')

        if len(split_part2_tokens) == 0:
            raise Exception('format_model_output_error', 'split2 len 0')

        # exactly one %composition if composition question
        if comp == 'composition' and ((pd.Series(split_part2_tokens) == '%composition') * 1.0).sum() != 1:
            raise Exception('format_model_output_error', 'no %composition in split2')


        output = [{'ID': pairs_dev['aux_data']['ID'], 'comp': comp,
                    'split_part1': ' '.join(split_part1_tokens), \
                    'split_part2': ' '.join(split_part2_tokens),
                    'input_tokens': input_tokens, \
                    'input_len': len(input_tokens), \
                    'answers': pairs_dev['aux_data']['answers'], \
                    'question': pairs_dev['aux_data']['question'], \
                    'sorted_annotations': [{'dep':a['dep'],'dependentGloss':a['dependentGloss']} \
                                           for a in pairs_dev['aux_data']['sorted_annotations']],\
                    'program': program, \
                    'program_tokens': program_tokens, \
                    'output_len': len(program), \
                    'model_prob':model_prob}]

        if 'skip_limit' in mask_state:
            output[0].update({'skipped': config.skip_limit - mask_state['skip_limit'], \
                              'skip_ind_list': mask_state['skip_ind_list'], \
                              'skip_token_list': skip_token_list
                              })

        if 'P1' in mask_state:
            output[0].update({'p1':mask_state['P1'], 'p1_sup': p1_sup, \
                    'p2':mask_state['P2'], 'p2_sup': p2_sup, 'comp_sup': comp_sup})

        if config.rerun_program:
            output[0].update({'Reward_MRR':pairs_dev['aux_data']['Reward_MRR'], \
                              'Simp_Reward_MRR':pairs_dev['aux_data']['Simp_Reward_MRR'], \
                              'max_comp_wa_conf': pairs_dev['aux_data']['max_comp_wa_conf'], \
                              'max_simp_wa_conf': pairs_dev['aux_data']['max_simp_wa_conf']})

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
        output_prob = 0
        mask_state = None

        di = 0
        while di < config.MAX_LENGTH:
            if config.use_output_masking:
                output_mask, mask_state = self.calc_output_mask(input_variable, result, mask_state)

            decoder_output, decoder_hidden, output_dist = self.decoder(
                decoder_input, decoder_hidden, encoder_hidden, encoder_hiddens, encoder_hidden, output_mask)

            ## DEBUGING
            #decoder_output.register_hook(print)

            if di<len(target_variable):
                if config.RL_Training:
                    loss += self.criterion(decoder_output, target_variable[di]) * reward
                else:
                    loss += self.criterion(decoder_output, target_variable[di])

            if DO_TECHER_FORCING or config.rerun_program:
                curr_output = target_variable[di].data[0]
                decoder_input = target_variable[di]
                #if output_mask[target_variable[di].data[0]] != 1:
                #    assert()
                result.append(target_variable[di].data[0])
            else:
                if config.use_output_masking:
                    curr_output = np.argmax(decoder_output.data - ((output_mask == 0).float() * 1000))
                else:
                    curr_output = np.argmax(decoder_output.data)
                decoder_input = Variable(torch.LongTensor([curr_output]))
                result.append(curr_output)

            output_prob += decoder_output.data[0].numpy()[curr_output]
            output_masks.append(output_mask.int().tolist())
            output_dists.append((output_dist.data[0] * 100).round().int().tolist())

            ## EOS or teacher forcing an len> target var
            di += 1
            if self.vocab_ind_to_word(curr_output) == ')' or (DO_TECHER_FORCING and di >= len(target_variable)):
                # last masking (updating mask state with last program token)
                if config.use_output_masking:
                    output_mask, mask_state = self.calc_output_mask(input_variable, result, mask_state)
                break

        if type(loss)!=int:
            loss_value = loss.data[0] / target_length
        else:
            loss_value = 0
        return loss_value , result, loss, output_dists, output_masks , mask_state, output_prob

    def forward_rl_train(self,input_variable, target_variable, reward = 0, loss=0,  DO_TECHER_FORCING=True):
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
        output_prob = 0
        mask_state = None

        di = 0
        while di < config.MAX_LENGTH:
            # output masking is used in RL to update number of skips etc...
            if config.use_output_masking:
                output_mask, mask_state = self.calc_output_mask(input_variable, result, mask_state)

            decoder_output, decoder_hidden, output_dist = self.decoder(
                decoder_input, decoder_hidden, encoder_hidden, encoder_hiddens, encoder_hidden, output_mask)

            ## DEBUGING
            #decoder_output.register_hook(print)

            if di<len(target_variable):
                loss = self.criterion(decoder_output, target_variable[di])

            # RL training is always "Teacher forcing"
            curr_output = target_variable[di].data[0]
            decoder_input = target_variable[di]
            result.append(target_variable[di].data[0])

            output_prob += decoder_output.data[0].numpy()[curr_output]
            output_masks.append(output_mask.int().tolist())
            output_dists.append((output_dist.data[0] * 100).round().int().tolist())

            ## EOS or teacher forcing an len> target var
            di += 1
            if di >= len(target_variable):
                # last masking (updating mask state with last program token)
                if config.use_output_masking:
                    output_mask, mask_state = self.calc_output_mask(input_variable, result, mask_state)
                break

        loss_value = loss.data[0] / target_length

        return loss_value , result, loss, output_dists, output_masks , mask_state, output_prob


    def traj_sampling_forward(self, input_variable, target_variable, reward = 0, loss=0,  DO_TECHER_FORCING=False):
        loss_value = 0
        all_results = []
        all_output_dists = []
        all_output_masks = []
        all_mask_states = []
        all_output_probs = []

        for traj_num in range(config.traj_per_question):
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
            output_prob = 0
            mask_state = None

            di =0
            while di < config.MAX_LENGTH:
                if config.use_output_masking:
                    output_mask, mask_state = self.calc_output_mask(input_variable, result, mask_state)

                decoder_output, decoder_hidden, output_dist = self.decoder(
                    decoder_input, decoder_hidden, encoder_hidden, encoder_hiddens, encoder_hidden, output_mask)

                if config.use_output_masking:
                    uniform = np.ones(len(output_dist.data[0])) / len(output_dist.data[0])
                    masked_distribution = (output_dist.data[0].numpy() * config.alpha_uni + \
                                           uniform * (1-config.alpha_uni)) * output_mask.numpy()
                    masked_distribution /= masked_distribution.sum()
                    curr_output = int(np.random.choice(len(masked_distribution), 1, p=masked_distribution)[0])
                else:
                    curr_output = np.argmax(decoder_output.data)

                decoder_input = Variable(torch.LongTensor([curr_output]))

                result.append(curr_output)
                output_prob += decoder_output.data[0].numpy()[curr_output]
                output_masks.append(output_mask.int().tolist())
                output_dists.append((output_dist.data[0] * 100).round().int().tolist())

                ## EOS or teacher forcing an len> target var
                di += 1
                if self.vocab_ind_to_word(curr_output) == ')':
                    # last masking (updating mask state with last program token)
                    if config.use_output_masking:
                        output_mask, mask_state = self.calc_output_mask(input_variable, result, mask_state)
                    break

            all_results.append(result)
            all_output_dists.append(output_dists)
            all_output_masks.append(output_masks)
            all_mask_states.append(mask_state)
            all_output_probs.append(output_prob)


        return loss_value , all_results, loss, all_output_dists, all_output_masks , all_mask_states, all_output_probs

    def beam_search_forward(self, input_variable, target_variable, reward=0, loss=0, DO_TECHER_FORCING=False):
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

        result = []
        output_masks = []
        output_dists = []
        # Without teacher forcing: use its own predictions as the next input
        sub_optimal_chosen = False
        output_mask = None
        output_dist = None


        prev_beam = [{'prob':0, 'decoder_input':Variable(torch.LongTensor([[config.SOS_token]])), \
                      'output':[],'decoder_hidden':encoder_hidden, 'mask_state':{'P1': None, 'P2': None, 'state':0}}]
        final_beam = []
        di = 0
        while di < config.MAX_LENGTH and len(prev_beam)>0:

            new_beam = []
            for trajectory in prev_beam:
                if config.use_output_masking:
                    output_mask ,mask_state = self.calc_output_mask_naacl18(input_variable, trajectory['output'], trajectory['mask_state'])

                decoder_output, decoder_hidden, output_dist = self.decoder(
                    trajectory['decoder_input'], trajectory['decoder_hidden'], \
                    encoder_hidden, encoder_hiddens, encoder_hidden, output_mask)

                if config.use_output_masking:
                    masked_distribution = output_dist.data[0].numpy() * output_mask.numpy()
                    #masked_distribution /= masked_distribution.sum()
                    probs = pd.Series(masked_distribution.data.tolist()).round(4).sort_values(0, ascending=False)[0:config.K]

                    for ind, prob in probs.iteritems():
                        if prob > 0:
                            new_rec = {'prob': trajectory['prob'] + np.log(prob), 'output': trajectory['output'].copy() + [ind], \
                                                 'decoder_input': Variable(torch.LongTensor([ind])), \
                                                 'decoder_hidden': Variable(decoder_hidden.data), 'mask_state':mask_state.copy()}
                            if self.vocab_ind_to_word(ind) == ')':
                                final_beam.append(new_rec)
                            else:
                                new_beam.append(new_rec)
                else:
                    print('not supported yet')
                    assert()

            # using only the K best trjectories:
            new_beam = sorted(new_beam, key=itemgetter('prob'),  reverse=True)
            new_beam = new_beam[0:config.K]
            prev_beam = new_beam.copy()
            di  += 1

        final_beam += prev_beam
        for traj in final_beam:
            traj['prob'] /= len(traj['output'])
        final_beam = sorted(final_beam, key=itemgetter('prob'),  reverse=True)
        final_beam = final_beam[0:config.K]


        result = []
        mask_states = []
        output_probs = []
        for trajectory in final_beam:
            result.append(trajectory['output'])
            mask_states.append(trajectory['mask_state'])
            output_probs.append(np.exp(trajectory['prob']))

        # no loss when computing trajectories
        loss_value = 0
        return loss_value, result, loss, output_dists, output_masks, mask_states, output_probs


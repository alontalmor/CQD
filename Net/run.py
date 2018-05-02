from config import *
from io import open
import pandas as pd
import numpy as np
import random

# A general training and evaluation class for neural networks


class NNRun():
    def __init__(self, model, pairs_train, pairs_dev, pairs_trian_index):
        self.pairs_trian_index = pairs_trian_index
        self.pairs_train = pairs_train
        self.pairs_dev = pairs_dev
        self.model = model
        self.iteration = 0
        self.best_accuracy = 0
        model.init_optimizers()

    def train_rl(self):
        loss = 0
        self.max_testing_accuracy = 0
        self.start = datetime.datetime.now()

        self.train_loss = 0  # Reset every print_every
        self.stats = []
        # For early stopping
        self.best_accuracy = 0
        self.best_accuracy_iter = 0
        self.iteration = 0

        rl_update_data = []

        while self.iteration < self.best_accuracy_iter + config.NO_IMPROVEMENT_ITERS_TO_STOP \
                and self.iteration < config.MAX_ITER:
            self.iteration += 1

            # question number choice is not randomized (order of question was already randomized)
            chosen_question = self.iteration % len(self.pairs_trian_index)

            example_traj_inds = self.pairs_trian_index[list(self.pairs_trian_index.keys())[chosen_question]]

            # normalizing rewards
            rewards = pd.Series([self.pairs_train[ind]['aux_data']['Reward_MRR'] for ind in example_traj_inds])
            #model_probs = np.exp(pd.Series([self.pairs_train[ind]['aux_data']['model_prob'] for ind in example_traj_inds]))
            #norm_model_probs = model_probs / model_probs.sum()

            # assuming rewards are normalized per question, running all question trajectories sequentially
            example_rl_update_data = []
            for ind,chosen_ind in enumerate(example_traj_inds):
                #chosen_ind = random.choice(self.pairs_trian_index[list(self.pairs_trian_index.keys())[chosen_question]])

                training_pair = self.pairs_train[chosen_ind]
                input_variable = training_pair['x']
                target_variable = training_pair['y']

                reward = None
                reward = float(rewards.iloc[ind])
                if config.devide_by_traj_num and abs(reward) < config.min_reward_update:
                    continue

                train_loss, output_seq, loss , output_dists, output_masks, mask_state, model_prob = \
                        self.model.forward_rl_train(input_variable, target_variable, reward, loss,
                                                        DO_TECHER_FORCING=True)

                example_rl_update_data.append({'loss':loss,'log_model_prob':model_prob,'model_prob':np.exp(model_prob), \
                                               'reward':reward, 'ID':training_pair['aux_data']['ID']})

            # Updateing rewards
            model_probs = pd.Series([traj['model_prob'] for traj in example_rl_update_data])
            rewards = pd.Series([traj['reward'] for traj in example_rl_update_data])
            norm_model_probs = model_probs / model_probs.sum()

            if config.reward_sub_mean:
                if len(rewards) > 1:
                    # sub weighted average of traj model prob * reward
                    rewards -= (norm_model_probs * rewards).sum()
                    rewards *= norm_model_probs
                    # rewards += config.MIN_REWARD_TRESH / len(rewards)

            if config.devide_by_traj_num:
                rewards /= len(rewards)

            if config.max_margin_rl:
                if len(rewards) > 1:
                    traj_plus_minus_mat = pd.DataFrame()
                    for i in range(len(rewards)):
                        for j in range(i+1,len(rewards)):
                            traj_plus_minus_mat.iloc[i,j] = rewards[i]
                            print(i,j,rewards.iloc[i],rewards.iloc[j])
                    traj_plus = rewards.argmax()
                    traj_minus = rewards.argmin()
                    rewards = rewards[[traj_plus, traj_minus]]
                    example_traj_inds = [example_traj_inds[traj_plus], example_traj_inds[traj_minus]]
                    rewards += config.MIN_REWARD_TRESH - rewards.mean()

            # updating loss
            for traj, reward in zip(example_rl_update_data, rewards):
                traj['loss'] *= reward

                # Debug net:
                traj['model_scaled_reward'] = reward

            rl_update_data += example_rl_update_data


            ### PRINT TRAINING STATS ##
            if self.iteration % config.print_every == 0:
                print('--- iteration ' + str(self.iteration) +  ' run-time ' + str(datetime.datetime.now() - self.start) +  ' --------')
                config.write_log('INFO', 'Train stats', {'trainset loss': round(self.train_loss / config.print_every, 4) , \
                                                         'iteration':self.iteration})
                self.train_loss = 0

            ## EVALUATE MODEL ##
            if self.iteration % config.evaluate_every == 0:
                print('-- Evaluating on devset --- ')
                print('prev max adjusted accuracy %.4f' % (self.best_accuracy))
                model_output = self.evaluate()

                if self.best_accuracy + 0.001 < self.curr_accuracy or config.always_save_model:
                    print('saving model')
                    self.model.save_model('_' + str(self.iteration))

                    config.store_json(model_output, config.split_points_dir + config.out_subdir, config.eval_set + '_' + str(self.iteration))
                    config.store_csv(model_output, config.split_points_dir + config.out_subdir, config.eval_set + '_' + str(self.iteration))

                    self.best_accuracy = self.curr_accuracy
                    self.best_accuracy_iter = self.iteration

            # computing gradients (update is always per Example, so all RL trajectories are summed before step)
            if self.iteration % config.MINI_BATCH_SIZE == 0:
                total_loss = 0
                for traj in rl_update_data:
                    total_loss += traj['loss']

                self.train_loss += train_loss

                loss.backward()

                grad_max_vals = []
                grad_mean_vals = []
                for p in filter(lambda p: p.grad is not None, self.model.encoder.parameters()):
                    grad_max_vals.append(p.grad.data.abs().max())
                    grad_mean_vals.append(p.grad.data.abs().mean())
                for p in filter(lambda p: p.grad is not None, self.model.decoder.parameters()):
                    grad_max_vals.append(p.grad.data.abs().max())
                    grad_mean_vals.append(p.grad.data.abs().mean())

                if config.print_all_grad:
                    pass

                if config.grad_clip_value is not None:
                    for p in filter(lambda p: p.grad is not None, self.model.encoder.parameters()):
                        p.grad.data.clamp_(min=-config.grad_clip_value, max=config.grad_clip_value)
                    for p in filter(lambda p: p.grad is not None, self.model.decoder.parameters()):
                        p.grad.data.clamp_(min=-config.grad_clip_value, max=config.grad_clip_value)

                #torch.nn.utils.clip_grad_value_(self.model.encoder.parameters(), 1)
                #torch.nn.utils.clip_grad_value_(self.model.decoder.parameters(), 1)

                #print(self.model.decoder.out.weight.abs().max())
                #print(self.model.encoder.GRU.weight_hh_l0.abs().max())
                #print(self.model.encoder.GRU.weight_ih_l0.abs().max())



                #print(self.iteration)
                #print(self.model.decoder.out.weight.abs().max())
                #print(self.model.encoder.GRU.weight_hh_l0.abs().max())
                #print(self.model.encoder.GRU.weight_ih_l0.abs().max())

                debug_nn_df = pd.DataFrame(rl_update_data)
                if config.debug_nn:
                    config.write_log('DEBUG_NN', 'mini batch update',
                                 {'loss':total_loss.data[0], \
                                  'loss_reward_1':np.sum([row['loss'].data[0] for ind,row in debug_nn_df[debug_nn_df['reward']==1].iterrows()]), \
                                  'loss_not_reward_1':np.sum([row['loss'].data[0] for ind,row in debug_nn_df[debug_nn_df['reward']<1].iterrows()]), \
                                  'max grads':np.max(grad_max_vals), \
                                  'mean grads':np.mean(grad_mean_vals), \
                                  'mean_prob':debug_nn_df.groupby('ID')['model_prob'].mean().mean(), \
                                  'mean_prob_mass':debug_nn_df.groupby('ID')['model_prob'].sum().mean(), \
                                  'iteration': self.iteration})

                self.model.optimizer_step()
                rl_update_data = []

    def train_supervised(self):
        loss = 0
        self.max_testing_accuracy = 0
        self.start = datetime.datetime.now()

        self.train_loss = 0  # Reset every print_every
        self.stats = []
        # For early stopping
        self.best_accuracy = 0
        self.best_accuracy_iter = 0
        self.iteration = 0

        while self.iteration < self.best_accuracy_iter + config.NO_IMPROVEMENT_ITERS_TO_STOP \
                and self.iteration < config.MAX_ITER:
            self.iteration += 1

            # question number choice is not randomized (order of question was already randomized)
            chosen_question = self.iteration % len(self.pairs_trian_index)

            example_traj_inds = self.pairs_trian_index[list(self.pairs_trian_index.keys())[chosen_question]]

            # normalizing rewards
            if config.RL_Training:
                rewards = pd.Series([self.pairs_train[ind]['aux_data']['Reward_MRR'] for ind in example_traj_inds])
                model_probs = np.exp(pd.Series([self.pairs_train[ind]['aux_data']['model_prob'] for ind in example_traj_inds]))
                norm_model_probs = model_probs / model_probs.sum()

                if config.reward_sub_mean:
                    if len(rewards)>1:
                        # sub weighted average of traj model prob * reward
                        rewards -= (norm_model_probs * rewards).sum()
                        rewards *= norm_model_probs
                        #rewards += config.MIN_REWARD_TRESH / len(rewards)

                if config.devide_by_traj_num:
                    rewards /= len(rewards)

                if config.max_margin_rl:
                    if len(rewards) > 1:
                        traj_plus = rewards.argmax()
                        traj_minus = rewards.argmin()
                        rewards = rewards[[traj_plus,traj_minus]]
                        example_traj_inds = [example_traj_inds[traj_plus],example_traj_inds[traj_minus]]
                        rewards +=  config.MIN_REWARD_TRESH - rewards.mean()


            # assuming rewards are normalized per question, running all question trajectories sequentially
            for ind,chosen_ind in enumerate(example_traj_inds):
                #chosen_ind = random.choice(self.pairs_trian_index[list(self.pairs_trian_index.keys())[chosen_question]])

                training_pair = self.pairs_train[chosen_ind]
                input_variable = training_pair['x']
                target_variable = training_pair['y']

                reward = None
                if config.RL_Training:
                    reward = float(rewards.iloc[ind])
                    if abs(reward)<config.min_reward_update:
                        continue

                # Teacher forcing
                if config.use_teacher_forcing and self.iteration < config.teacher_forcing_full_until:
                    teacher_forcing = True
                elif config.use_teacher_forcing and self.iteration < config.teacher_forcing_partial_until:
                    teacher_forcing = True if random.random() < 0.5 else False
                else:
                    teacher_forcing = False

                train_loss, output_seq, loss , output_dists, output_masks, mask_state, model_prob = \
                        self.model.forward_func(input_variable, target_variable, reward, loss,
                                                        DO_TECHER_FORCING=teacher_forcing)

            ### PRINT TRAINING STATS ##
            if self.iteration % config.print_every == 0:
                print('--- iteration ' + str(self.iteration) +  ' run-time ' + str(datetime.datetime.now() - self.start) +  ' --------')
                config.write_log('INFO', 'Train stats', {'trainset loss': round(self.train_loss / config.print_every, 4) , \
                                                         'iteration':self.iteration})
                self.train_loss = 0

            ## EVALUATE MODEL ##
            if self.iteration % config.evaluate_every == 0:
                print('-- Evaluating on devset --- ')
                print('prev max adjusted accuracy %.4f' % (self.best_accuracy))
                model_output = self.evaluate()

                if self.best_accuracy + 0.001 < self.curr_accuracy or config.always_save_model:
                    print('saving model')
                    self.model.save_model('_' + str(self.iteration))

                    config.store_json(model_output, config.split_points_dir + config.out_subdir, config.eval_set + '_' + str(self.iteration))
                    config.store_csv(model_output, config.split_points_dir + config.out_subdir, config.eval_set + '_' + str(self.iteration))

                    self.best_accuracy = self.curr_accuracy
                    self.best_accuracy_iter = self.iteration

            # computing gradients (update is always per Example, so all RL trajectories are summed before step)
            if self.iteration % config.MINI_BATCH_SIZE == 0:
                self.train_loss += train_loss

                loss.backward()

                if config.print_all_grad:
                    for p in filter(lambda p: p.grad is not None, self.model.encoder.parameters()):
                        print(p.grad.data.abs().max())
                    for p in filter(lambda p: p.grad is not None, self.model.decoder.parameters()):
                        print(p.grad.data.abs().max())

                if config.grad_clip_value is not None:
                    for p in filter(lambda p: p.grad is not None, self.model.encoder.parameters()):
                        p.grad.data.clamp_(min=-config.grad_clip_value, max=config.grad_clip_value)
                    for p in filter(lambda p: p.grad is not None, self.model.decoder.parameters()):
                        p.grad.data.clamp_(min=-config.grad_clip_value, max=config.grad_clip_value)

                #torch.nn.utils.clip_grad_value_(self.model.encoder.parameters(), 1)
                #torch.nn.utils.clip_grad_value_(self.model.decoder.parameters(), 1)



                self.model.optimizer_step()

                print(self.iteration)
                #print(self.model.decoder.out.weight.abs().max())
                #print(self.model.encoder.GRU.weight_hh_l0.abs().max())
                #print(self.model.encoder.GRU.weight_ih_l0.abs().max())

                loss = 0

    def evaluate(self, gen_model_output=True):
        model_output = []
        model_format_errors = {}
        pairs_dev = [self.pairs_dev[i] for i in range(len(self.pairs_dev))]
        sample_size = min(config.max_evalset_size,len(self.pairs_dev))
        self.model.init_stats()

        self.test_loss = 0
        accuracy_avg = 0
        for test_iter in range(config.evalset_offset, config.evalset_offset + sample_size):
            if test_iter % 200 == 0:
                print(test_iter)
            testing_pair = pairs_dev[test_iter]

            test_loss , output_seq, loss, output_dists, output_masks, mask_state, model_prob  = \
                self.model.forward_func(testing_pair['x'], testing_pair['y'])
            self.test_loss += test_loss

            # generating model output
            if gen_model_output and config.gen_model_output:
                try:
                    if len(output_seq)>0 and config.generate_all_skips:
                        input_tokens = [token['dependentGloss'] for token in
                                        testing_pair['aux_data']['sorted_annotations']]
                        for seq, ms, op in zip(output_seq, mask_state, model_prob):
                            for skip_ind, token in enumerate(seq):
                                if seq[skip_ind] < len(input_tokens) and random.random() < 0.1:
                                    model_output += self.model.format_model_output(testing_pair, seq, output_dists,
                                                                       output_masks, ms,op, skip_ind)
                    elif config.beam_search_gen or config.gen_trajectories:
                        for seq, ms, op in zip(output_seq,mask_state,model_prob):
                            model_output += self.model.format_model_output(testing_pair, seq, output_dists,output_masks, ms, op)
                    else:
                        model_output +=  self.model.format_model_output(testing_pair, output_seq, output_dists, output_masks, mask_state, model_prob)
                except Exception as inst:
                    if inst.args[0] == 'format_model_output_error':
                        if inst.args[1] in model_format_errors:
                            model_format_errors[inst.args[1]] += 1
                        else:
                            model_format_errors[inst.args[1]] = 1
                    else:
                        print(traceback.format_exc())
                    # adding empty output
                    model_output += [{'ID': testing_pair['aux_data']['ID'], 'question': testing_pair['aux_data']['question'], \
                           'answers': testing_pair['aux_data']['answers']}]

            # cases in which no output_seq or target exist will be considered a mistake (equivalent to accuracy append 0)
            if output_seq == [] or len(testing_pair['y']) == 0:
                continue

            if not config.beam_search_gen and not config.gen_trajectories:
                accuracy_avg += self.model.evaluate_accuracy(testing_pair['y'], output_seq, testing_pair['aux_data'], mask_state)

        ##### LOG STATS #########
        self.curr_accuracy = accuracy_avg / sample_size
        detailed_stats = self.model.calc_detailed_stats(sample_size)
        detailed_stats.update({'evalset loss': round(self.test_loss / sample_size, 4), \
                                                      'adjusted accuracy': round(self.curr_accuracy, 4), \
                                                      'best_adjusted accuracy': round(self.best_accuracy, 4), \
                                                      'iteration': self.iteration})
        config.write_log('INFO', 'Evaluation stats', detailed_stats )
        for key in model_format_errors:
            model_format_errors[key] /= float(sample_size)
        config.write_log('INFO', 'Model Format Errors', model_format_errors)

        if gen_model_output and config.gen_model_output:
            return model_output
        else:
            return True






from config import *
from io import open
import pandas as pd
import numpy as np
import random

# A general training and evaluation class for neural networks


class NNRun():
    def __init__(self, model, pairs_train, pairs_dev):
        self.pairs_train = pairs_train
        self.pairs_dev = pairs_dev
        self.model = model
        self.iteration = 0
        self.best_accuracy = 0
        model.init_optimizers()

    def run_training(self):
        loss = 0
        self.max_testing_accuracy = 0
        self.start = datetime.datetime.now()

        self.train_loss = 0  # Reset every print_every
        self.stats = []
        # For early stopping
        self.best_accuracy = 0
        self.best_accuracy_iter = 0
        self.iteration = 0

        random.seed(7)
        inds = range(len(self.pairs_train))

        while self.iteration < self.best_accuracy_iter + config.NO_IMPROVEMENT_ITERS_TO_STOP \
                and self.iteration < config.MAX_ITER:
            self.iteration += 1
            chosen_ind = random.choice(inds)

            training_pair = self.pairs_train[chosen_ind]
            input_variable = training_pair['x']
            target_variable = training_pair['y'] \

            reward = None
            if config.RL_Training:
                reward = training_pair['aux_data']['Reward_MRR']

            # Teacher forcing
            if config.use_teacher_forcing and self.iteration < config.teacher_forcing_full_until:
                teacher_forcing = True
            elif config.use_teacher_forcing and self.iteration < config.teacher_forcing_partial_until:
                teacher_forcing = True if random.random() < 0.5 else False
            else:
                teacher_forcing = False

            train_loss, output_seq, loss , output_dists, output_masks = \
                    self.model.forward(input_variable, target_variable, reward, loss,
                                                    DO_TECHER_FORCING=teacher_forcing)

            # computing gradients
            if self.iteration % config.MINI_BATCH_SIZE == 0:
                self.train_loss += train_loss

                loss.backward()

                self.model.optimizer_step()

                loss = 0

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

                    config.store_json(model_output, config.split_points_dir + config.out_subdir, config.EVALUATION_SET + '_' + str(self.iteration))
                    config.store_csv(model_output, config.split_points_dir + config.out_subdir, config.EVALUATION_SET + '_' + str(self.iteration))

                    self.best_accuracy = self.curr_accuracy
                    self.best_accuracy_iter = self.iteration

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

            test_loss , output_seq, loss, output_dists, output_masks  = self.model.forward(testing_pair['x'], testing_pair['y'])
            self.test_loss += test_loss

            # generating model output
            if gen_model_output and config.gen_model_output:
                try:
                    if len(output_seq)>0 and config.generate_all_skips:
                        input_tokens = [token['dependentGloss'] for token in
                                        testing_pair['aux_data']['sorted_annotations']]
                        for skip_ind, token in enumerate(output_seq):
                            if output_seq[skip_ind] < len(input_tokens):
                                model_output += self.model.format_model_output(testing_pair, output_seq, output_dists,
                                                                       output_masks, skip_ind)
                    else:
                        model_output +=  self.model.format_model_output(testing_pair, output_seq, output_dists, output_masks)
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

            accuracy_avg += self.model.evaluate_accuracy(testing_pair['y'], output_seq, testing_pair['aux_data'])

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






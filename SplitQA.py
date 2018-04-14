from config import *
from Executors.conjunction import Conjunction
from Executors.simple_search import SimpleSearch
from Executors.composition import Composition


class SplitQA():
    def __init__(self):
        self.Executors = [Composition(),Conjunction(),SimpleSearch()]

        ##################################
        # calculate split points
        with open(config.split_points_dir + config.eval_set + '.json', 'r') as outfile:
            self.split_points = pd.DataFrame(json.load(outfile))
        
        ## Appending data from original complexwebquestions
        with open(config.complexwebquestions_dir + 'ComplexWebQuestions_' + config.eval_set + '.json', 'r') as outfile:
            complexwebquestions = pd.DataFrame(json.load(outfile))
        
        self.split_points = self.split_points.merge(pd.DataFrame(complexwebquestions)[['answers','ID']],on='ID',how='inner')
        self.split_points.rename(columns={'answers_y': 'answers'}, inplace=True)

    def run_executors(self):
        ##################################
        # generate webanswer first batch
        
        question_list = []
        for executor in self.Executors:
            question_list += executor.gen_webanswer_first_wave(self.split_points)
        
        # dropbing duplicates questions
        question_list = pd.DataFrame(question_list).drop_duplicates(subset=['question']).to_dict(orient='rows')
        
        # Generating RC answers
        batch_webanswer_question = []
        ind = 0
        for question in question_list:
            ind += 1
            if ind % 500 == 0:
                print
                ind
            cached_results = cache.get(ANSWER_SOURCE,
                                       {'question': question['question'], 'goog_query': question['goog_query'],
                                        'version': answer_batch.web_answer_version})
            if cached_results is not None and (not FULL_REPASS or cached_results != []):
                webanswer_dict[question['question']] = pd.DataFrame(cached_results)
            else:
                batch_webanswer_question.append(question)

        curr_batch_webanswer_question = batch_webanswer_question[offset:offset + BATCH_SIZE]
        if len(curr_batch_webanswer_question) == 0:
            break

        CacheResults = SearchCache.find(
            {'querystr': question['goog_query'], "page": 0, "type": 'SCREEN', "last_update": {"$gte": six_months_ago}})
        CacheResults_Count = CacheResults.count()
        if USE_GOOG_CACHE and CacheResults_Count > 0:
            cahched_item = CacheResults.next()
            question['google_results'] = cahched_item['results']

        offset += BATCH_SIZE
        if len(non_cached_questions)>0:
            for_goog_dict = {'target_dir': 'google','questions': [{'question': question['question'], 'goog_query': question['goog_query']} for
                                           question in non_cached_questions]}
            dbx.files_upload(json.dumps(for_goog_dict), '/google/' + batch_dir.split('batch_output/')[1][0:-1] + '_for_goog.json')

            #dbx.files_upload(json.dumps([{'question':question['question'],'goog_query':question['goog_query']} for question in non_cached_questions]),
            #                 '/google/' + batch_dir.split('batch_output/')[1][0:-1] + '_for_goog.json')

        if len(cached_questions)>0:
            dbx.files_upload(json.dumps(cached_questions, indent=4, sort_keys=True),
                             '/google/' + batch_dir.split('batch_output/')[1][0:-1] + '_done.json')



        with open(config.rc_answer_cache_dir + config.eval_set  + '.json', 'r') as outfile:
            webanswer_dict = json.load(outfile)
        
            # used as dataframes in the executors..
            for key in webanswer_dict.keys():
                webanswer_dict[key] = pd.DataFrame(webanswer_dict[key])
        
            for question in question_list:
                if question['question'] not in webanswer_dict:
                    print('Warning!!! RC cache is missing the following question: ' + question['question'])


        ##################################
        # run executors
        print('Running executors - first wave')
        question_list = []
        batch_webanswer_question = []
        for executor in self.Executors:
            print('Executor: ' + executor.name)
            exec_res = executor.proc_webanswer_wave_results(self.split_points,webanswer_dict)
            self.split_points = exec_res['split_points']
            question_list += exec_res['question_list']
        
        print('Running executors - second wave (second stage of composition questions)')
        ##################################
        # run executors - second wave
        for executor in self.Executors:
            print('Executor: ' + executor.name)
            exec_res = executor.proc_webanswer_second_wave_results(self.split_points,webanswer_dict)
            self.split_points = exec_res['split_points']
        
        
        results_path = config.data_dir +  'final_results' + '_' + config.eval_set
        print ('saving results in ' + results_path)
        
        self.split_points.to_csv(results_path + '.csv',encoding="utf-8",index=False)
        with open(results_path + '.json', 'w') as outfile:
            json.dump(self.split_points.to_dict(orient="rows"), outfile)

    def compute_final_results(self):
        comp = 0
        comp_count = 0
        conj = 0
        comj_count = 0
        SplitQA_P1_sum = 0
        SplitQA_oracle_P1_sum = 0
        SimpQA_P1_sum = 0
        only_comp_count = 0
        only_comp_count_right = 0
        only_search_count = 0
        only_search_count_right = 0
        we_chose_comp = 0
        we_chose_simp = 0
        both_count = 0
        for ind, q in self.split_points.iterrows():
            conjunction_rc_conf = -100
            composition_rc_conf = -100
            SimpQA_rc_conf = -100
            if q['SplitQA_conjunction_rc_conf'] is not None and len(q['SplitQA_conjunction_rc_conf']) > 0:
                conjunction_rc_conf = max(q['SplitQA_conjunction_rc_conf'])
            if q['SplitQA_composition_rc_conf'] is not None and len(q['SplitQA_composition_rc_conf']) > 0:
                composition_rc_conf = max(q['SplitQA_composition_rc_conf'])
            if q['SimpQA_rc_conf'] is not None and len(q['SimpQA_rc_conf']) > 0:
                SimpQA_rc_conf = max(q['SimpQA_rc_conf'])
        
            # constant normalization - (a bit ugly i know, will be learned in future work...)
            conjunction_rc_conf += 1.0
            composition_rc_conf += 2.0
        
            SimpQA_P1 = (q['SimpQA_MRR']==1)*100.0
            SplitQA_conj_P1 = (q['SplitQA_conjunction_MRR']==1)*100.0
            SplitQA_comp_P1 = (q['SplitQA_composition_MRR']==1)*100.0
        
            SimpQA_P1_sum += SimpQA_P1
        
            if q['comp'] == 'conjunction':
                if conjunction_rc_conf >= SimpQA_rc_conf:
                    we_chose_comp += 1
                    SplitQA_P1_sum += SplitQA_conj_P1
                else:
                    we_chose_simp += 1
                    SplitQA_P1_sum += SimpQA_P1
            else:
                if composition_rc_conf >= SimpQA_rc_conf:
                    we_chose_comp += 1
                    SplitQA_P1_sum += SplitQA_comp_P1
                else:
                    SplitQA_P1_sum += SimpQA_P1
                    we_chose_simp += 1
        
            SplitQA_oracle_P1_sum += max(SimpQA_P1, SplitQA_conj_P1, SplitQA_comp_P1)
        
            if (q['SimpQA_MRR'] == 1 and q['SplitQA_composition_MRR'] == 1) or \
                    (q['SimpQA_MRR'] == 1 and q['SplitQA_conjunction_MRR'] == 1):
                both_count += 1
        
            if q['comp'] == 'composition':
                if q['SimpQA_MRR'] < 1 and q['SplitQA_composition_MRR'] == 1:
                    if composition_rc_conf > SimpQA_rc_conf:
                        comp += 1
                        only_comp_count_right += 1
                    only_comp_count += 1
                    comp_count += 1
                if q['SimpQA_MRR'] == 1 and q['SplitQA_composition_MRR'] < 1:
                    if composition_rc_conf < SimpQA_rc_conf:
                        comp += 1
                        only_search_count_right += 1
                    only_search_count += 1
                    comp_count += 1
            else:
                if q['SimpQA_MRR'] < 1 and q['SplitQA_conjunction_MRR'] == 1:
                    if conjunction_rc_conf > SimpQA_rc_conf:
                        only_comp_count_right += 1
                        conj += 1
                    comj_count += 1
                    only_comp_count += 1
                if q['SimpQA_MRR'] == 1 and q['SplitQA_conjunction_MRR'] < 1:
                    if conjunction_rc_conf < SimpQA_rc_conf:
                        conj += 1
                        only_search_count_right += 1
                    only_search_count += 1
                    comj_count += 1
        
        print('SplitQA-composition RC conf accuracy: ' + str(float(comp) / comp_count))
        print('SplitQA-conjunction RC conf accuracy: ' + str(float(conj) / comj_count))
        print('number of only Compositionality : ' + str(only_comp_count))
        print('number of only Compositionality that we got right: ' + str(only_comp_count_right))
        print('number of only Search : ' + str(only_search_count))
        print('number of only Search that we got right: ' + str(only_search_count_right))
        print('number of both ( at same time): ' + str(both_count))
        print('compositionality portion: ' + str(float(we_chose_comp) / (we_chose_comp + we_chose_simp)))
        print('SimpQA: ' + str(float(SimpQA_P1_sum) / len(self.split_points)))
        print('SplitQA ' + str(float(SplitQA_P1_sum) / len(self.split_points)))
        print('SplitQAOracle ' + str(float(SplitQA_oracle_P1_sum) / len(self.split_points)))

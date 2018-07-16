__author__='efe arin'

import gensim
import numpy as np
import gensim, logging
import pandas as pd
from scipy.spatial import distance
from nltk.tokenize import word_tokenize
import gensim, logging
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
import time
import warnings
import os, sys, stat


import func_repo
import io_repo
import calculate_repo
import result_repo
import dominant_word_repo


t_start=time.time()

# TODO add naive bayes approach
# TODO add comparable lengths to mlp vectors
# TODO like tree search collect all possible vectors in memory then select the best
# TODO instead of sum and counts at the feature vectors their ratio could be used for some of them (mostly in variance case)
# TODO if all 0 normalization gets divide by 0 error write an exception
# TODO while founding expanded sum and count in feature vectors additionally sticked dominant words should be considered

# close polyfit poor conditioned warnings
warnings.simplefilter('ignore', np.RankWarning)

# in func repo convert numbers to strings
# data_divide_constant*fake_data_divide_constant>2 otherwise there will be a problem at io_repo.save()
data_divide_constant = 2
fake_data_divide_constant = 2
data_path = 'data/'
list_path = 'data/lists/'
main_turn_path = 'data/turns/'
# os.chmod(os.getcwd(),stat.S_IRWXU)


features = ['length_of_phrase',
            'direct_len'        , 'direct_mean'     ,'direct_var',
            'closest_len'       , 'closest_mean'    , 'closest_var',
            'expanded_len'      , 'expanded_mean'   , 'expanded_var',
            'expanded_len_mean' , 'expanded_len_var',
            'dominant_len'      , 'dominant_mean'   , 'dominant_var',
            'dominant_varience_mean']

# Load Google's pre-trained Word2Vec model.
#TODO: download gensim to conda
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=100)
# w2v_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
fake_w2v_model = w2v_model
# get data
df=pd.read_csv('data/data.csv',sep="\t", nrows=500).drop(['PhraseId', 'SentenceId'], axis=1)
# df=pd.read_csv('data/data.csv',sep="\t").drop(['PhraseId', 'SentenceId'], axis=1)
df['Sentiment']=(df['Sentiment']-2)/2
# df.Sentiment = df.Sentiment.astype(float, copy=False)
# df.Sentiment = df.Sentiment+1
# clean data
func_repo.clean_data(df)
print('celan_data done')
# divide data and get df list
clean_df_list = func_repo.divide_data(df, data_divide_constant*fake_data_divide_constant)
print('clean_df_list done')
io_repo.save(clean_df_list,list_path+'clean_df_list')
# get fully cleaned df list
fully_clean_df_list = func_repo.get_fully_cleaned_df_list(clean_df_list)
print('fully_clean_df_list done')
io_repo.save(fully_clean_df_list,list_path+'fully_clean_df_list')
# get sentence and word lists
word_df_list = func_repo.get_word_lists(fully_clean_df_list)
print('get sentence_word_lists done')
io_repo.save(word_df_list,list_path+'word_df_list')
# get dominant word df list
dominant_word_df = dominant_word_repo.get_dominant_word_df(fully_clean_df_list, 'data/dominant_words/' )
io_repo.save(dominant_word_df, 'data/dominant_words/dominant_words_df.csv')

calculated_test_scores_df_list=[]

for x in range(0, data_divide_constant):
    print('   main turn '+str(x+1)+' start')
    turn_path = main_turn_path+str(x+1)
    mlp_train_input = []
    mlp_train_output = []
    mlp_type = []
    # create required df_lists
    test_df_list, train_df_list, train_word_df_list = func_repo.get_required_df_lists(
        clean_df_list, fully_clean_df_list, word_df_list, fake_data_divide_constant, x)
    print('   get_required_df_lists done')

    # collect feature vectors for train set
    for y in range (0, len(train_df_list)):
        print('      fake_turn '+str(y+1)+' start')
        # prepare necessary df's
        fake_test_df, fake_train_df, fake_train_word_df = func_repo.get_required_fake_dfs (
            train_df_list, train_word_df_list, y)
        print('      get_required_fake_dfs done')
        # train w2v algorithm
        #TODO: always use google model
        #fake_w2v_model = func_repo.get_w2v_model(fake_train_sentence_df)
        #print('      get_w2v_model done')
        # get dominant word df with their linear phrase length dependent equations
        # get partial feature vectors
        # normalization for feature vector parameters needed output returns already normalized
        # add dimention of relevant length of sounts
        partial_mlp_train_input,partial_mlp_type, partial_mlp_train_output = calculate_repo.get_matrix(fake_test_df, fake_train_df,
                                                                                      fake_train_word_df,fake_w2v_model,
                                                                                      dominant_word_df)
        print('      get_matrix done')
        # add the partials to whole set
        mlp_train_input += partial_mlp_train_input
        mlp_train_output += partial_mlp_train_output
        mlp_type+=partial_mlp_type
        print('     fake_turn ' + str(y + 1) + ' done')
    # save feature vectors an normalized output score as df
    mlp_feed_df = io_repo.save([mlp_train_input, mlp_type, mlp_train_output], turn_path+'/mlp_feed_df.csv', 1)
    # normalize train set
    mlp_model_list, tree_model_list, scaler_list = calculate_repo.getModelsAndScalers(mlp_feed_df)
    print('   model and scaler done')
    # save model
    # to load model later
    # mlp_model = joblib.load('mlp_model.pkl')
    io_repo.save([[mlp_model_list,tree_model_list],scaler_list],turn_path,2)
    # create required dfs
    test_df, train_df, train_word_df = func_repo.dflists_to_dfs (test_df_list,train_df_list,
                                                                                  train_word_df_list)
    # get w2v model
    # w2v_model = func_repo.get_w2v_model(train_sentence_df)
    # print('   get_w2v_model done')
    # io_repo.save(w2v_model,turn_path+'/w2v_model')
    # get dominant word df

    # calculate for validation set
    calculated_test_scores_df = calculate_repo.calculate_test(test_df, train_df, train_word_df, w2v_model,
                                                              dominant_word_df, mlp_model_list, tree_model_list, scaler_list)
    print('   calculate_test done')
    # save result df
    io_repo.save(calculated_test_scores_df, turn_path+'/results/result_df.csv')
    # output result file
    result_repo.output_results(calculated_test_scores_df, 0.5, main_turn_path+str(x+1)+'/')
    print('   output_results done')
    calculated_test_scores_df_list.append(calculated_test_scores_df)
    print('   main turn ' + str(x + 1) + ' done')

total_calculated_test_scores_df = pd.concat(calculated_test_scores_df_list, ignore_index=True).reset_index(drop=True)
result_repo.output_results(total_calculated_test_scores_df,0.5,data_path)
io_repo.save(total_calculated_test_scores_df, data_path+'/results/result_df.csv')
print('time: '+ "%.2f"%(time.time()-t_start))

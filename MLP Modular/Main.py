__author__='efe arin'

import numpy as np
import gensim, logging
import pandas as pd
from scipy.spatial import distance
from nltk.tokenize import word_tokenize
import gensim, logging
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
import time

import func_repo
import io_repo
import calculate_repo
import result_repo
import dominant_word_repo

t_start=time.time()

# add naive bayes approach
# add comparable lengths to mlp vectors
# like tree search collect all possible vectors in memory then select
# add dominant words
# instead of sum and counts at the feature vectors their ratio could be used for some of them (mostly in variance case)

# in func repo convert numbers to strings
# data_divide_constant*fake_data_divide_constant>2 otherwise there will be a problem at io_repo.save()
data_divide_constant = 2
fake_data_divide_constant = 2
data_path = 'data/'
list_path = 'data/lists/'
main_turn_path = 'data/main_turns/'

# get data
df=pd.read_csv('data/data.csv',sep="\t")
df=df.head(n=1000)
# clean data
func_repo.clean_data(df)
# divide data and get df list
clean_df_list = func_repo.divide_data(df, data_divide_constant*fake_data_divide_constant)
io_repo.save(clean_df_list,list_path+'clean_df_list')
# get fully cleaned df list
fully_clean_df_list = func_repo.get_fully_cleaned_df_list(clean_df_list)
io_repo.save(fully_clean_df_list,list_path+'fully_clean_df_list')
# get sentence and word lists
sentence_df_list, word_df_list = func_repo.get_sentence_word_lists(fully_clean_df_list)
io_repo.save(sentence_df_list,list_path+'sentence_df_list')
io_repo.save(word_df_list,list_path+'word_df_list')

calculated_test_scores_df_list=[]

for x in range(0, data_divide_constant):
    mlp_train_input = []
    mlp_train_output = []
    # create required df_lists
    test_df_list, train_df_list, train_sentence_df_list, train_word_df_list = func_repo.get_required_df_lists(
        clean_df_list, fully_clean_df_list, sentence_df_list, word_df_list, fake_data_divide_constant, x)
    # collect feature vectors for train set
    for y in range (0, len(train_df_list)):
        # prepare necessary df's
        fake_test_df, fake_train_df, fake_train_sentence_df, fake_train_word_df = func_repo.get_required_fake_dfs (
            train_df_list, train_sentence_df_list, train_word_df_list, y)
        # train w2v algorithm
        fake_w2v_model = func_repo.get_w2v_model(fake_train_sentence_df)
        # get dominant word df with their linear phrase length dependent equations
        fake_dominant_word_df = dominant_word_repo.get_dominant_word_df(fake_train_df)
        # get partial feature vectors
        # normalization for feature vector parameters needed output returns already normalized
        # add dimention of relevant length of sounts
        partial_mlp_train_input, partial_mlp_train_output = calculate_repo.get_matrix(fake_test_df, fake_train_df,
                                                                                      fake_train_word_df,fake_w2v_model,
                                                                                      fake_dominant_word_df)
        # add the partials to whole set
        mlp_train_input += partial_mlp_train_input
        mlp_train_output += partial_mlp_train_output
    # save feature vectors an normalized output score as df
    io_repo.save([mlp_train_input, mlp_train_output], main_turn_path+str(x+1)+'/mlp_feed_df.csv')
    # normalize train set
    mlp_train_input, mlp_train_output, normalize_constants = func_repo.normalize_inp_out_lists(mlp_train_input,
                                                                                               mlp_train_output, [])
    # create mlp model
    mlp_model = MLPRegressor(hidden_layer_sizes=(20,), activation='identity',
                             alpha=0.001, learning_rate='constant', early_stopping=True)
    # train mlp model
    mlp_model.fit(mlp_train_input, mlp_train_output)
    # save model
    # to load model later
    # mlp_model = joblib.load('mlp_model.pkl')
    joblib.dump(mlp_model, main_turn_path+str(x+1)+'/mlp_model.pkl')
    # create required dfs
    test_df, train_df, train_sentence_df, train_word_df = func_repo.dflists_to_dfs (test_df_list,train_df_list,
                                                                                  train_sentence_df_list,
                                                                                  train_word_df_list)
    w2v_model = func_repo.get_w2v_model(train_sentence_df)
    dominant_word_df = dominant_word_repo.get_dominant_word_df(train_df)
    calculated_test_scores_df = calculate_repo.calculate_test(test_df, train_df, train_word_df, w2v_model,
                                                              dominant_word_df, mlp_model, normalize_constants)
    # save result df
    io_repo.save(calculated_test_scores_df, main_turn_path+str(x+1)+'/result_df.csv')
    # output result file
    result_repo.output_results(calculated_test_scores_df, 0.5, main_turn_path+str(x+1)+'/')
    calculated_test_scores_df_list.append(calculated_test_scores_df)

total_calculated_test_scores_df = pd.concat(calculated_test_scores_df_list, ignore_index=True).reset_index(drop=True)
io_repo.save(total_calculated_test_scores_df, data_path+'result_df.csv')
result_repo.output_results(total_calculated_test_scores_df,0.5,data_path)

print(time.time()-t_start)

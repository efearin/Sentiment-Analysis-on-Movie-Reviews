import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import os
from scipy.spatial import distance
from nltk.tokenize import word_tokenize
import gensim, logging
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import matplotlib
import math
import sklearn.neural_network as nn



# Check if folder exists if not create
# get path as string as 'directory' no return
def open_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# save 'df' as csv file to 'path' (ex: 'data/fofo.csv')
# if 'path' is not exist create new
# no return
def save_df (df,path):
    temp = path.split("/")
    file_name = temp[-1]
    del temp[-1]
    directory = '/'.join(temp)
    open_folder(directory)
    df.to_csv(directory+'/'+file_name, sep='\t')

# save dataframe list 'df_list' to folder 'directory' with their indexes in 'df_list'
# ex: directory = 'data/lists/clean_df_list' under that folder there will be csv files as 1.csv, 2.csv...
# represents all dataframe files in df_list in the order of their index
# df_list[0] => data/lists/clean_df_list/1.csv
def save_df_list (df_list,directory):
    for x in range(0,len(df_list)):
        save_df(df_list[x], directory + '/' + str(x + 1) + '.csv')

# save w2v model to the path given
# path example 'data/main_turn/1/fake_turns/1w2vmodel'
# file name is 1w2vmodel included to path
def save_w2v_model (model,path):
    model.save(path)

def save_fig (model,path):
    temp = path.split("/")
    del temp[-1]
    directory = '/'.join(temp)
    open_folder(directory)
    model.savefig(path)

# # save w2v model to the path given
# path example 'data/main_turn/1/fake_turns/mlp_feed_df.csv'
# file name is included to path
def get_mlp_feed_df (lst):
    mlp_train_input = lst[0]
    mlp_type = lst[1]
    mlp_train_output = lst[2]
    mlp_feed_df = pd.DataFrame({'length_of_phrase': pd.Series([item[0] for item in mlp_train_input]),

                                'direct_len': pd.Series([item[1] for item in mlp_train_input]),
                                'direct_mean': pd.Series([item[2] for item in mlp_train_input]),
                                'direct_var': pd.Series([item[3] for item in mlp_train_input]),

                                'closest_len': pd.Series([item[4] for item in mlp_train_input]),
                                'closest_mean': pd.Series([item[5] for item in mlp_train_input]),
                                'closest_var': pd.Series([item[6] for item in mlp_train_input]),

                                'expanded_len': pd.Series([item[7] for item in mlp_train_input]),
                                'expanded_mean': pd.Series([item[8] for item in mlp_train_input]),
                                'expanded_var': pd.Series([item[9] for item in mlp_train_input]),

                                'expanded_len_mean': pd.Series([item[10] for item in mlp_train_input]),
                                'expanded_len_var': pd.Series([item[11] for item in mlp_train_input]),

                                'dominant_len': pd.Series([item[12] for item in mlp_train_input]),
                                'dominant_mean': pd.Series([item[13] for item in mlp_train_input]),
                                'dominant_var': pd.Series([item[14] for item in mlp_train_input]),

                                'dominant_varience_mean': pd.Series([item[15] for item in mlp_train_input]),
                                'type': pd.Series(mlp_type),
                                'output': pd.Series(mlp_train_output)})
    return mlp_feed_df

# save whatever is given as an input
# mind that seperator of df_list and mlp_feed_df is list object (both are list) length
def save (object, path, mlp=0):
    if isinstance(object, pd.core.frame.DataFrame):
        save_df(object,path)
    elif isinstance(object, list):
        if len(object) > 2:
            if not mlp:
                save_df_list(object,path)
            if mlp:
                temp = path.split("/")
                del temp[-1]
                directory = '/'.join(temp)
                open_folder(directory)
                mlp_feed_df = get_mlp_feed_df(object)
                mlp_feed_df.to_csv(path, sep='\t')
                return mlp_feed_df
        if len(object)==2:
            for x in range(0,len(object[1])):
                if isinstance(object[0][0][x], nn.multilayer_perceptron.MLPRegressor):
                    joblib.dump(object[0][0][x], path+ '/mlp_model'+str(x)+'.pkl')
                    joblib.dump(object[0][1][x], path + '/tree_model' + str(x) + '.pkl')
                    joblib.dump(object[1][x], path + '/scaler' + str(x) + '.pkl')
    elif isinstance(object, gensim.models.word2vec.Word2Vec):
        save_w2v_model(object, path)
    # TODO couldn't recognize the object passed is a matplotlib plot
    elif isinstance(object, module):
        save_fig(object, path)
    else:
        print('file type error')
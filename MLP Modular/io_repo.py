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

# # save w2v model to the path given
# path example 'data/main_turn/1/fake_turns/mlp_feed_df.csv'
# file name is included to path
def save_mlp_feed_df (lst, path):
    temp = path.split("/")
    del temp[-1]
    directory = '/'.join(temp)
    open_folder(directory)
    mlp_train_input = lst[0]
    mlp_train_output = lst[1]
    mlp_feed_df = pd.DataFrame({'length_of_phrase': pd.Series([item[0] for item in mlp_train_input]),
                                'direct_sum': pd.Series([item[1] for item in mlp_train_input]),
                                'direct_count': pd.Series([item[2] for item in mlp_train_input]),
                                'closest_sum': pd.Series([item[3] for item in mlp_train_input]),
                                'closest_count': pd.Series([item[4] for item in mlp_train_input]),
                                'output': pd.Series(mlp_train_output)})
    mlp_feed_df.to_csv(path, sep='\t')

# save whatever is given as an input
# mind that seperator of df_list and mlp_feed_df is list object (both are list) length
def save (object, path):
    if isinstance(object, pd.core.frame.DataFrame):
        save_df(object,path)
    elif isinstance(object, list):
        if len(object) > 2:
            save_df_list(object,path)
        elif len(object) == 2:
            save_mlp_feed_df(object,path)
    elif isinstance(object, gensim.models.word2vec.Word2Vec):
        save_w2v_model(object, path)
    else:
        print('file type error')
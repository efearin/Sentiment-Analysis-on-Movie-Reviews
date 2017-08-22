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

def save (object, path):
    if isinstance(object, pd.core.frame.DataFrame):
        save_df(object,path)
    elif isinstance(object, list):
        save_df_list(object,path)
    elif isinstance(object, gensim.models.word2vec.Word2Vec):
        save_w2v_model(object, path)
    else:
        print('file type error')
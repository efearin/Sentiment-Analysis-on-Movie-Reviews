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






# Manipulate 'df', no return
# PURPOSE OF CODE
# Cleaning meaningless words and punctuations to prepare data
# to further train and test datasets seperation.
# Since test data should also be cleaned before scoring,
# this process shifted to the beginning
# and done over whole data before test and train devision.

# PROCESS NOTES
# WILL BE DONE
#  - ..\AppData\Roaming\nltk_data\corpora\stopwords\english_redesigned document
# could be checked for stopwords
# DONE
# - all phrases are lowered
# - stemmed using nltk PorterStemmer
# - punctuations are erased
# - stop words (nltk library pre created stop words list is redesigned to use) are erased
# - saved as data/data_clean.csv
def clean_data (df):
    # Replace capitals to non-capitals.
    df.Phrase = df.Phrase.str.lower()
    # There are no 'isn't', 'arent', 'havent' or such words in data,
    # they all have already changed to their not seperated forms like "is not" .
    # Stem the data using nltk PorterStemmer
    # which changes the words like "pythoner", "pythoning" etc. to the "python".
    ps = PorterStemmer()
    def stemming(x):
        proper_words = []
        words = x.split()
        for word in words:
            proper_words.append(ps.stem(word))
        return " ".join(proper_words)
    df.Phrase = df.Phrase.map(stemming)
    # Remove punctuations.
    df.Phrase = df.Phrase.str.replace('[^\w\s]', '')
    # nltk's english stopwords list at location
    # "..\AppData\Roaming\nltk_data\corpora\stopwords" includes words like "no", "not".
    # They changes the meaning in sentiment analysis since if "no" is erased from data
    # "not good" and "good" scored as same. So such stopwords are removed from pre-created list of nltk library.
    # The words erased from list are
    # "not, no, very, too, up, down, in, out, over, under, few"
    # Check again if miss.
    # import redesigned stopword list named as "english_redesigned" from nltk
    stopwords_set = set(stopwords.words('english_redesigned'))
    # define a map to delete stopwords under each phrase
    def stop_erase(x):
        proper_words = []
        words = x.split()
        for word in words:
            if not word in stopwords_set:
                proper_words.append(word)
        return " ".join(proper_words)
    df.Phrase = df.Phrase.map(stop_erase)

# 'df' devided a list of partial 'df's returned
# PURPOSE OF CODE
# Data is devided randomly into 100/'percent' equal group to use 'percent' of data for train and remaining for validation.

# PROCESS NOTES
# WILL BE DONE
# DONE
# - data is randomized as rows
# - devided into 5 group with equal number of rows
# - saved as 1df.csv, 2df.csv ..
def divide_data (df,divide_constant):
    partial_df_list=[]
    length_of_df = len(df)
    # Randomize rows in df.
    df = df.sample(frac=1).reset_index(drop=True)
    # Divide and add partials to list
    for x in range(0, divide_constant):
        partial_df_list.append(df[int(x*length_of_df/divide_constant):int((x+1)*length_of_df/divide_constant)].reset_index(drop=True))
    return partial_df_list

# Manipulate 'df', no return
# PURPOSE OF CODE
# Train data empty phrases are cleaned and same phrase rows are merged as their sentiment scores are averaged.

# PROCESS NOTES
# WILL BE DONE
# DONE
# - empty rows are assigned as NaN and erased
# - same phrases with different sentiment scores are avaraged and merged as single
def fully_clean_df (df):
    # Replace empty phrase rows to NaN then delete.
    #define a map to change empty phrases to NaN
    def change_nan(x):
        if isinstance(x,str):
            words=x.split()
            if len(words)>0:
                return x
        return np.nan
    df['Phrase'] = df.Phrase.map(change_nan)
    #remove NaN rows of df
    df = df.dropna()
    #reset indexing
    df = df.reset_index(drop=True)
    # Afer clean, the same same phrases with differen sentiment scores created.
    # Take a phrase as referance to the "phrase" list. +
    # Find the same ones in data and record the indexes to the "same" list.
    # In such same phrase groups assign the first one as referance by changing its sentiment score to group average
    # and change other phrases to NaN to remove later from the list.
    # Before doing that sentiment scores should be converted into float data type from intiger(as a default).
    # convert sentiment scores to float to write averaged ones
    df.Sentiment = df.Sentiment.astype(float, copy=False)
    # "phrase" holds the current phrase to compare through whole set and find the indexes of same other phrase rows
    # which will written in "same"
    # then they will changed to NaN for further clean process
    # summation holds the summation of same phrase sentiment scores that will avaraged and written to the first phrase among all same phrases
    for index, row in df.iterrows():
        if row.Phrase != np.nan:
            same = df.Phrase[df.Phrase == row.Phrase]
            if len(same.index) > 1:
                summation = 0
                for x in range(1, len(same.index)):
                    df.loc[same.index[x], 'Phrase'] = np.nan
                    summation += df.loc[same.index[x], 'Sentiment']
                summation += df.loc[same.index[0], 'Sentiment']
                df.loc[same.index[0], 'Sentiment'] = summation / len(same.index)
    # clean all NaN rows
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df

# take list of df
# return clean verison of them by calling fully_clean_df
# for each element of list
def get_fully_cleaned_df_list (main_df_list):
    df_list = list(main_df_list)
    for x in range(0,len(df_list)):
        df_list[x] = fully_clean_df(df_list[x])
    return df_list







# PURPOSE OF CODE
# From train dataset create a new one which contains only full sentences not their phrases to prevent over-fit on further word2vec train
# using dataset's sentenceId attributes. Get only full sentences to feed word2vec.
# Only full sentences are choosen since phrases of full sentences may overfits the word vectors
# in such a way that words in a sentence may strictly stick to each other if not only sentence but also phrases of that sentence fed.
# PROCESS NOTES
# WILL BE DONE
# DONE
# - longest phrases of same sentenceId collections of dataset is taken for new dataset "train-w2v.csv"
def get_sentence_df (main_df):
    # code finds the longest phrases of same sentenceid, assign other phrases as NaN then delete related rows
    # "same" holds the indexes of phrases with same sentence id
    # "ind_ex" holds the longest phrase index up to that point of code
    # "lenght" holds the length of longest phrase at index "ind_ex" up to that point of code
    # "new_lenght holds the coming phrase length that will be compared to "length" if longer "ind_ex" and "length" will be refreshed
    df = main_df.copy(deep=True)
    for indx, row in df.iterrows():
        if row.PhraseId != np.nan:
            same = df.SentenceId[df.SentenceId == row.SentenceId]
            if len(same.index) > 1:
                ind_ex = same.index[0]
                length = len(df.loc[ind_ex, 'Phrase'])
                for x in range(1, len(same.index)):
                    new_length = len(df.loc[same.index[x], 'Phrase'])
                    if new_length > length:
                        df.loc[ind_ex, "PhraseId"] = np.nan
                        ind_ex = same.index[x]
                        length = new_length
                    else:
                        df.loc[same.index[x], "PhraseId"] = np.nan
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df

# PURPOSE OF CODE
# Create dataset from train set that contains only phrases with single word.
# If no direct sentiment score of any word in test set found the word with closest to that word with sentiment score in this set will be used.
def get_word_df (main_df):
    df = main_df.copy(deep=True)
    mask = df.Phrase.str.match(r'\A[\w-]+\Z')
    df = df[mask]
    df = df.reset_index(drop=True)
    return df

# take fully cleaned df list
# return related word and sentence lists of each element
# by calling get_sentence_df and get_word_df
def get_sentence_word_lists (df_list):
    sentence_df_list = []
    word_df_list = []
    for x in df_list:
        sentence_df_list.append(get_sentence_df(x))
        word_df_list.append(get_word_df(x))
    return sentence_df_list, word_df_list

# get required df lists and return df lists
# that index of input lists are assigned as test remanings as train
def get_required_df_lists (clean_df_list, fully_clean_df_list, sentence_df_list, word_df_list, fake_data_divide_constant, index):
    test_df_list = clean_df_list[index * fake_data_divide_constant: index * fake_data_divide_constant +
                                                              fake_data_divide_constant]
    train_df_list = fully_clean_df_list[0: index * fake_data_divide_constant] + \
                    fully_clean_df_list[index * fake_data_divide_constant + fake_data_divide_constant:]
    train_sentence_df_list = sentence_df_list[0: index * fake_data_divide_constant] +\
                             sentence_df_list [index * fake_data_divide_constant + fake_data_divide_constant:]
    train_word_df_list = word_df_list[0: index * fake_data_divide_constant] +\
                             word_df_list [index * fake_data_divide_constant + fake_data_divide_constant:]
    return test_df_list, train_df_list, train_sentence_df_list, train_word_df_list

# get list of df and return df
# by assingning index element of inputs as fake test and remaninf fake train
# used to collect partial train feature vectors
def get_required_dfs (train_df_list, train_sentence_df_list, train_word_df_list, index):
    fake_test_df = train_df_list[index].reset_index(drop=True)
    fake_train_df = pd.concat([k for i, k in enumerate(train_df_list) if i != index], ignore_index=True).reset_index(drop=True)
    fake_train_sentence_df = pd.concat([k for i, k in enumerate(train_sentence_df_list) if i != index], ignore_index=True).reset_index(drop=True)
    fake_train_word_df = pd.concat([k for i, k in enumerate(train_word_df_list) if i != index], ignore_index=True).reset_index(drop=True)
    return fake_test_df, fake_train_df, fake_train_sentence_df, fake_train_word_df




# get dataframe and train w2v algorithm then return the related model
def get_w2v_model(df):
    tokenized_sentences = df.apply(lambda row: word_tokenize(row['Phrase']), axis=1)
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Word2Vec(tokenized_sentences, min_count=1)
    return model
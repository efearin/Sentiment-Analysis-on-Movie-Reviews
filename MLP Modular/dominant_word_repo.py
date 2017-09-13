# as version 2 dominant word scores has model (polynomial) of phrase lengths score=function(phrase_length)
# first get most frequent 500 words in corpus
# for each word collect the phrases that word exists in
# collect the length and scores of the phrases and
# fit it in polynomial such that score=function(phrase_length) using polyfit
# using model for each phrase calculate the error of score
# choose words as dominant that the variance of error below a value
# save the polynomial coefficients of the word to predict later scores by feeding length of phrases

import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# linear base function with parameter phrase length is used
# TODO mlp model structured might be used but how think about it
# train base mlp then extract dominant words with it
# add new spaces to feature vectors and train new mlp for the final use might be used
def get_dominant_word_df (df):
    # phrase list
    df_list=[]
    # word list
    df_word_list = []
    for index, row in df.iterrows():
        tmp = row.Phrase.split()
        df_list.append(tmp)
        df_word_list += tmp
    # get most 500 frequent word list
    # TODO instead if most frequent 'x' words, words with frequency more than 'y' approach might be better
    freq_list = nltk.FreqDist(df_word_list)
    common_words_list = freq_list.most_common(500)
    # min frequency bound could be added instead of getting top X frequent words
    # least frequent word of comman_words_list is around 150
    # might not be enough to decide if it is dominant
    # in common_words_list
    # variance will be used as reliable measure
    dominant_words_list =[]
    dominant_words_eqn_list=[]
    dominant_words_variance_list=[]
    for word in common_words_list:
        lengths=[]
        scores=[]
        for indx, phrase in enumerate(df_list):
            if word[0] in phrase:
                lengths.append(len(phrase))
                scores.append(df.loc[indx, 'Sentiment'])
        # plt.plot(lengths, scores,'ro')
        # plt.savefig(word[0]+".png")
        # plt.clf()
        # TODO polyfit order might be changed
        eqn = np.polyfit(lengths, scores, 5)
        err=[]
        for indx, length in enumerate(lengths):
            a=np.polyval(eqn, length)
            b=scores[indx]
            err.append(np.abs(a-b))
        err_mean = np.mean(err)
        err_variance = np.var(err)
        # TODO err_rate might be high decrease if necessary, no sense about varience
        if err_mean < 0.6 and err_variance < 0.8:
            # funcy=[]
            # for x in range (1,25):
            #     funcy.append(np.polyval(eqn,x))
            # plt.plot(lengths, scores,'ro')
            # plt.plot(range(1,25),funcy)
            # # plt.show()
            # plt.savefig(word[0]+".png")
            # plt.clf()
            dominant_words_list.append(word[0])
            dominant_words_eqn_list.append(eqn)
            dominant_words_variance_list.append(err_variance)

    dominant_words_df = pd.DataFrame({'equation': pd.Series(dominant_words_eqn_list),
                                      'word': pd.Series(dominant_words_list),
                                      'variance': pd.Series(dominant_words_variance_list)})
    return dominant_words_df


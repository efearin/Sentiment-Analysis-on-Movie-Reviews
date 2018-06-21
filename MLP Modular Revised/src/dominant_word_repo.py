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
import io_repo
import collections

# linear base function with parameter phrase length is used
# TODO mlp model structured might be used but how think about it
# train base mlp then extract dominant words with it
# add new spaces to feature vectors and train new mlp for the final use might be used
def get_dominant_word_df (df_list, path):
    # phrase list
    # word list
    df=pd.concat(df_list).reset_index(drop=True)
    common_words_list = collections.Counter(" ".join(df["Phrase"]).split()).most_common(1000)
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
        tmp = df[df['Phrase'].str.contains(word[0])]
        tmp['number_of_words'] = tmp['Phrase'].str.split().apply(len)
        # TODO polyfit order might be changed
        eqn = np.polyfit(tmp['number_of_words'], tmp['Sentiment'], 5)
        tmp['calculated_sentiment'] = np.polyval(eqn, tmp['number_of_words'])
        tmp['error'] = abs(tmp['Sentiment']-tmp['calculated_sentiment'])
        err_mean = tmp['error'].mean()
        err_variance = tmp['error'].var()
        # TODO err_rate might be high decrease if necessary, no sense about varience
        if err_mean < 0.6 and err_variance < 0.8:
            # if there is a path input given save the related graphs to path
            if len(path) > 0:
                funcy=[]
                upper = tmp['number_of_words'].max()+10
                for x in range (1,upper):
                    funcy.append(np.polyval(eqn,x))
                plt.plot(tmp['number_of_words'], tmp['calculated_sentiment'],'ro')
                plt.plot(range(1,upper),funcy)
                plt.ylim(-1,5)
                plt.xlabel('Phrase Length')
                plt.ylabel('Score')
                # plt.show()
                io_repo.open_folder(path)
                plt.savefig(path+word[0]+'.png')
                plt.clf()

                plt.hist(tmp['calculated_sentiment'])
                plt.xlabel('Score')
                plt.ylabel('Frequency')
                plt.savefig(path+word[0]+'_hist.png')
                plt.clf()


            dominant_words_list.append(word[0])
            dominant_words_eqn_list.append(eqn)
            dominant_words_variance_list.append(err_variance)

    dominant_words_df = pd.DataFrame({'equation': pd.Series(dominant_words_eqn_list),
                                      'word': pd.Series(dominant_words_list),
                                      'variance': pd.Series(dominant_words_variance_list)})
    return dominant_words_df


def get_common_word_df_list (dflist):
    common_words_df_list=[]
    for df in dflist:
        # phrase list
        # word list
        common_words_list = collections.Counter(" ".join(df["Phrase"]).split()).most_common(20)
        # min frequency bound could be added instead of getting top X frequent words
        # least frequent word of comman_words_list is around 150
        # might not be enough to decide if it is dominant
        # in common_words_list
        # variance will be used as reliable measure
        common_words_df =[]
        for word in common_words_list:
            tmp = df[df['Phrase'].str.contains(word[0])]
            tmp['number_of_words'] = tmp['Phrase'].str.split().apply(len)
            tmp['Phrase']=word[0]
            common_words_df.append(tmp)
            common_words_df_list.append(pd.concat(common_words_df).reset_index(drop=True))
    return common_words_df_list
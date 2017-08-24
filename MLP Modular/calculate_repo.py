import sys
import numpy as np
import gensim, logging
import pandas as pd
from scipy.spatial import distance
from nltk.tokenize import word_tokenize
import gensim, logging
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib


##### Train set construction functions

# get_ functions return value of sentiment score
# collect_ functions have no return, they do manipulation directly on attribution_count and attribution_sum
# means get_ functions returns' should be added to the attribution_sum but collect_ functions already do that internally
def get_direct_score(phrase):
    find = train_df.Phrase[train_df.Phrase == phrase]
    if len(find.index):
        return train_df.loc[find.index[0], "Sentiment"]
    else:
        return np.NaN

def get_similar_word_score(word):
    # check if input has a proper vector
    try:
        vec = w2v_model[word]
    except:
        # print("input kelime vektörü bulunanadı")
        return np.NaN
    # search for most similar word
    similar_index = -1
    dist = float('Inf')
    for index, row in train_word_df.iterrows():
        phrase = row.Phrase
        try:
            vec_to_compare = w2v_model[phrase]
            tmpdist = distance.euclidean(vec, vec_to_compare)
            if tmpdist < dist:
                similar_index = index
                dist = tmpdist
        except:
            continue
    if similar_index == -1:
        # non of word corpus words has vector
        return np.NaN
    return train_word_df.loc[similar_index, "Sentiment"]

def collect_independent_word_scores(phrase_words):
    global direct_attribution_count, direct_attribution_sum, closest_attribution_count, closest_attribution_sum
    for m in phrase_words:
        score = get_direct_score(m)
        if ~np.isnan(score):
            direct_attribution_sum += score
            direct_attribution_count += 1
        else:
            closest_score = get_similar_word_score(m)
            if ~np.isnan(closest_score):
                closest_attribution_sum += closest_score
                closest_attribution_count += 1

# optimisation needed there exists duplicate code blocks
# when code hits the longest window phrase it directly calculate it by stoping shift of window
# in that case for the remaning beginning and end phrase parts there meigth be a miss for longest window phrases
# at the end there is a need for weighted dominant word calculation
def collect_phrase_scores(phrase):
    global direct_attribution_count, direct_attribution_sum, closest_attribution_count, closest_attribution_sum
    score = get_direct_score(phrase)
    # leaf condition
    # if direct score found add to sum and increment count and return
    if ~np.isnan(score):
        direct_attribution_sum += score
        direct_attribution_count += 1
        return
    # if no direct score found split phrase and handle depending on number of words included
    else:
        phrase_words = phrase.split()
        length = len(phrase_words)
        # leaf condition
        # if phrase contains single word
        # and it doesn't have direct score since code doesn't return from the above part (already checked case)
        if length == 1:
            similar_word_score = get_similar_word_score(phrase_words)
            if ~np. isnan(similar_word_score):
                closest_attribution_sum += similar_word_score
                closest_attribution_count += 1
            return
        # leaf condition
        elif length == 2:
            collect_independent_word_scores(phrase_words)
            return
        # mid condition recursive part
        else:
            score = np.nan
            # algorithm seach for direct score of word collections between ''
            # let say total phrase is 'a b c d' and no direct score found
            # search for 'a b c' d then a 'b c d' if no found at middle of ''
            # searc for 'a b' c d then a 'b c' d then a b 'c d'
            # if direct score found let say at case a b 'c d' for the beginning 'a b' and end part '' recall algorithm
            # if no found, search independent words scores out of these for loops
            # x holds how many words shorten from the full phrase
            # shorten up to 2 words include group for single words case get scores seperately
            for x in range(1, length - 1):
                # y holds how many shift is done with the window size of ((phrase length)-x)) over full phrase
                for y in range(0, x + 1):
                    phrase_middle_words = phrase_words[y:length - x + y]
                    score = get_direct_score(" ".join(phrase_middle_words))
                    if ~np.isnan(score):
                        # if word group contains more then or equel to 2 words has direct score break
                        break
                if ~np.isnan(score):
                    # if word group contains more then or equel to 2 words has direct score
                    # add score to sum and increment count
                    # recall the function for beginning and end part then break
                    direct_attribution_sum += score
                    direct_attribution_count += 1
                    phrase_initial_words = phrase_words[0:y]
                    phrase_final_words = phrase_words[length - x + y:length]
                    collect_phrase_scores(" ".join(phrase_initial_words))
                    collect_phrase_scores(" ".join(phrase_final_words))
                    break
            if np.isnan(score):
                # no word group direct score found get independent word scores then break
                collect_independent_word_scores(phrase_words)

def get_matrix(test_df_inp, train_df_inp, train_word_df_inp, w2v_model_inp):
    # list will be returned which has list[0] as input, list[1] as output
    # output is bounded between 0 and 1 by division
    # and features in matrix[0] are as
    # length_of_phrase
    # direct_sum
    # direct_count
    # closest_sum
    # closest_count
    global direct_attribution_count, direct_attribution_sum, closest_attribution_count, closest_attribution_sum
    global train_df, train_word_df, w2v_model
    train_df = train_df_inp
    train_word_df = train_word_df_inp
    w2v_model = w2v_model_inp
    inp=[]
    out=[]
    for indx, row in test_df_inp.iterrows():
        if not isinstance(row.Phrase, str):
            continue
        else:
            phrase_length = len(row.Phrase.split())
            closest_attribution_count = 0
            closest_attribution_sum = 0
            direct_attribution_count = 0
            direct_attribution_sum = 0
            if len(row.Phrase):
                phrase = row.Phrase
                collect_phrase_scores(phrase)
            inp.append([float(phrase_length),float(direct_attribution_sum),float(direct_attribution_count),
                        float(closest_attribution_sum),float(closest_attribution_count)])
            out.append(float(row.Sentiment)/4)
    return inp, out

def calculate_test(test_df, train_df, train_word_df, w2v_model, mlp_model):
    test_feature_vectors, test_real_scores = get_matrix(test_df, train_df, train_word_df, w2v_model)
    calculated_scores = mlp_model.predict(test_feature_vectors)
    error = []
    for x in range(0,len(test_real_scores)):
        error.append(abs(test_real_scores[x]-(4*calculated_scores[x]))/4)
    result_df = pd.DataFrame({'sentiment': pd.Series(test_real_scores),
                              'calculation': pd.Series(calculated_scores),
                              'error': pd.Series(error)})
    return result_df
# if __name__ == "__main__":
#     closest_attribution_count = 0
#     closest_attribution_sum = 0
#     direct_attribution_count = 0
#     direct_attribution_sum = 0
#     test = sys.argv[1]
#     train = sys.argv[2]
#     words_df = sys.argv[3]
#     w2v_model = sys.argv[4]
#     get_matrix(test)





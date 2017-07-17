import numpy as np
import gensim, logging
import pandas as pd
from scipy.spatial import distance
from nltk.tokenize import word_tokenize
import gensim, logging
from sklearn.neural_network import MLPRegressor

df_complete_train = pd.read_csv('data/train-clean.csv', sep="\t")
df_complete_train = df_complete_train.drop('Unnamed: 0', 1).drop('Unnamed: 0.1', 1)
df_complete_test = pd.read_csv('data/test.csv', sep="\t")
df_complete_test = df_complete_test.drop('Unnamed: 0', 1).drop('Unnamed: 0.1', 1)
print(df_complete_train.head())
print(df_complete_test.head())


##### Preprocess functions

def get_dfset(df):
    tmp=len(df)
    a=int(tmp/5)
    b=int(2*tmp/5)
    c=int(3*tmp/5)
    d=int(4*tmp/5)
    dfset = []
    dfset.append(df[0:a])
    dfset.append(df[a:b].reset_index(drop=True))
    dfset.append(df[b:c].reset_index(drop=True))
    dfset.append(df[c:d].reset_index(drop=True))
    dfset.append(df[d:].reset_index(drop=True))
    return dfset

def get_sentences(df):
    same = []
    ind_ex_ = 0
    length = 0
    new_length = 0
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

def get_w2v_model(df):
    tokenized_sentences = df.apply(lambda row: word_tokenize(row['Phrase']), axis=1)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Word2Vec(tokenized_sentences, min_count=1)
    return model

def get_words(df):
    mask = df.Phrase.str.match(r'\A[\w-]+\Z')
    df = df[mask]
    return df

##### Train set construction functions

# get_ functions return value of sentiment score
# collect_ functions have no return, they do manipulation directly on attribution_count and attribution_sum
# means get_ functions returns' should be added to the attribution_sum but collect_ functions already do that internally
def get_direct_score(phrase):
    find = train.Phrase[train.Phrase == phrase]
    if len(find.index):
        return train.loc[find.index[0], "Sentiment"]
    else:
        return np.NaN

def get_similar_word_score(word):
    try:
        vec = w2v_model[word]
        tmpvec = w2v_model[words_df.loc[0, "Phrase"]]
        tmpindx = 0
        dist = distance.euclidean(vec, tmpvec)
        for indx1, row in words_df.iterrows():
            tmpdist = distance.euclidean(vec, w2v_model[row.Phrase])
            if tmpdist < dist:
                dist = tmpdist
                tmpvec = w2v_model[row.Phrase]
                tmpindx = indx1
        return words_df.loc[tmpindx, "Sentiment"]
    except:
        return np.NaN

def collect_independent_word_scores(phrase_words):
    global direct_attribution_count, direct_attribution_sum, closest_attribution_count, closest_attribution_sum
    for m in phrase_words:
        score = get_direct_score(m)
        if ~np.isnan(score):
            direct_attribution_sum += score
            direct_attribution_count += 1
        else:
            closest_attribution_sum += get_similar_word_score(m)
            closest_attribution_count += 1

# optimisation needed there exists duplicate code blocks
# when code hits the longest window phrase it directly calculate it by stoping shift of window
# in that case for the remaning beginning and end phrase parts there meigth be a miss for longest window phrases
# at the end there is a need for weighted dominant word calculation
def collect_phrase_scores(phrase):
    global direct_attribution_count, direct_attribution_sum, closest_attribution_count, closest_attribution_sum
    score = get_direct_score(phrase)
    if ~np.isnan(score):
        # if direct score found add to sum and increment count
        direct_attribution_sum += score
        direct_attribution_count += 1
        return
    else:
        # if no direct score found split phrase
        phrase_words = phrase.split()
        length = len(phrase_words)
        if length == 1:
            # if phrase contains single word
            # and it doesn't have direct score since code doesn't return from the above part (already checked case)
            closest_attribution_sum += get_similar_word_score(phrase_words)
            closest_attribution_count += 1
            return
        elif length == 2:
            collect_independent_word_scores(phrase_words)
        else:
            score = np.nan
            # algorithm seach for direct score of word collections between ''
            # let say total phrase is 'a b c d' and no direct score found
            # search for 'a b c' d then a 'b c d' if no found at middle of ''
            # searc for 'a b' c d then a 'b c' d then a b 'c d'
            # if direct score found let say at case a b 'c d' for the beginning 'a b' and end part '' recall algorithm
            # if no found search independent words scores our of these for loops
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

def get_matrix(df):
    # list will be returned which has list[0] as input, list[1] as output
    # output is bounded between 0 and 1 by division
    # output is regration scores betwwen 0-5 and features in matrix[0] are as
    # length_of_phrase
    # direct_sum
    # direct_count
    # closest_sum
    # closest_count
    inp=[]
    out=[]
    for indx, row in df.iterrows():
        length = len(row.Phrase.split())
        closest_attribution_count = 0
        closest_attribution_sum = 0
        direct_attribution_count = 0
        direct_attribution_sum = 0
        if isinstance(row.Phrase, str):
            if len(row.Phrase):
                collect_phrase_scores(row.Phrase)
                inp.append([float(length),float(direct_attribution_sum),float(direct_attribution_count),
                            float(closest_attribution_sum),float(closest_attribution_count)])
                out.append(float(row.Sentiment)/5)
    return [inp,out]

##### Test score calculation functions
# Returns result df which contains rusults of calculatable rows
# so if size does not macth (input df and output df) there are some phrases that can not be calculated
def calculate(df):
    matrix = get_matrix(df.Phrase)
    real_scores = matrix[1]
    calculated_scores = mlp_model.predict(matrix[0])
    error = []
    for x in range(0,len(real_scores)):
        error.append(abs(real_scores[x]-(5*calculated_scores[x])))
    result_df = pd.DataFrame({'Sentiment': pd.Series(real_scores),
                              'calculation': pd.Series(calculated_scores),
                              'error': pd.Series(error)})
    return result_df

# MAIN

# construct train matrix be carreful test sets create train sets for the MLP
# since features depends on corpus used !!
dfset = get_dfset(df_complete_train)
train_inp=[]
train_out=[]
for x in range(0, len(dfset)):
    test = dfset[0]
    del dfset[0]
    train = pd.concat(dfset)
    sentence_df = get_sentences(train)
    w2v_model = get_w2v_model(sentence_df)
    words_df = get_words(train)
    partial_mlp_train_matrix = get_matrix(test)
    train_inp += partial_mlp_train_matrix[0]
    train_out += partial_mlp_train_matrix[1]
    dfset.append(test)

# construct MLP model
mlp_model = MLPRegressor(hidden_layer_sizes=(20,), activation='logistic', solver='adam',
                   alpha=0.001, learning_rate='constant', max_iter=100, random_state=5,
                   shuffle=True, early_stopping=True)
mlp_model.fit(train_inp,train_out)

# calculate

df_calculation = calculate(df_complete_test)
df_calculation.to_csv("data/test-MLPcalculated.csv", sep='\t')
import numpy as np
import gensim, logging
import pandas as pd
from scipy.spatial import distance
from nltk.tokenize import word_tokenize
import gensim, logging
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib

df_complete_train = pd.read_csv('data/train_clean.csv', sep="\t")
df_complete_train = df_complete_train.drop('Unnamed: 0', 1).drop('Unnamed: 0.1', 1)
df_complete_test = pd.read_csv('data/test.csv', sep="\t")
df_complete_test = df_complete_test.drop('Unnamed: 0', 1).drop('Unnamed: 0.1', 1)
# print(df_complete_train.head())
# print(df_complete_test.head())


##### Preprocess functions

def get_dfset(df):
    # takes full dataframe corpus devide it 5 equel piese add each piece to a list
    # return the list
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
    df.to_csv("data/MLPsentence.csv", sep='\t')
    return df

def get_w2v_model(df):
    tokenized_sentences = df.apply(lambda row: word_tokenize(row['Phrase']), axis=1)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Word2Vec(tokenized_sentences, min_count=1)
    model.save('MLPw2vmodel')
    return model

def get_words(df):
    mask = df.Phrase.str.match(r'\A[\w-]+\Z')
    df = df[mask]
    df = df.reset_index(drop=True)
    df.to_csv("data/MLPwords.csv", sep='\t')
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
    # check if input has a proper vector
    try:
        vec = w2v_model[word]
    except:
        # print("input kelime vektörü bulunanadı")
        return np.NaN
    # check if there is an initial vector closest to input at the word corpus
    # checking is limited, might be changed !
    similar_index = 0
    found = False
    for x in range(0, int(len(words_df)/2)):
        try:
            similar_vec = w2v_model[words_df.loc[x, "Phrase"]]
            similar_index = x
            found = True
        except:
            pass
        if found:
            break
    if not found:
        # print("inputa en yakın başlangıç vektörü bulunamadı")
        return np.NaN
    # if there is a vector of input and there is a initial vector to start
    dist = distance.euclidean(vec, similar_vec)
    for indx1, row in words_df.iterrows():
        phrase = row.Phrase
        try:
            vec_to_compare = w2v_model[phrase]
        except:
            continue
        # if a vector to compare is found
        tmpdist = distance.euclidean(vec, vec_to_compare)
        if tmpdist < dist:
            dist = tmpdist
            similar_vec = vec_to_compare
            similar_index = indx1
    return words_df.loc[similar_index, "Sentiment"]

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
            similar_word_score =  get_similar_word_score(phrase_words)
            if ~np. isnan(similar_word_score):
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
    global direct_attribution_count, direct_attribution_sum, closest_attribution_count, closest_attribution_sum
    # list will be returned which has list[0] as input, list[1] as output
    # output is bounded between 0 and 1 by division
    # and features in matrix[0] are as
    # length_of_phrase
    # direct_sum
    # direct_count
    # closest_sum
    # closest_count
    inp=[]
    out=[]
    for indx, row in df.iterrows():
        if not isinstance(row.Phrase, str):
            continue
        length = len(row.Phrase.split())
        closest_attribution_count = 0
        closest_attribution_sum = 0
        direct_attribution_count = 0
        direct_attribution_sum = 0
        if isinstance(row.Phrase, str):
            if len(row.Phrase):
                phrase = row.Phrase
                collect_phrase_scores(phrase)
                inp.append([float(length),float(direct_attribution_sum),float(direct_attribution_count),
                            float(closest_attribution_sum),float(closest_attribution_count)])
                out.append(float(row.Sentiment)/4)
    return [inp,out]

##### Test score calculation functions
# Returns result df which contains rusults of calculatable rows
# so if size does not macth (input df and output df) there are some phrases that can not be calculated
def calculate(df):
    matrix = get_matrix(df)
    real_scores = matrix[1]
    calculated_scores = mlp_model.predict(matrix[0])
    error = []
    for x in range(0,len(real_scores)):
        error.append(abs(real_scores[x]-(4*calculated_scores[x]))/4)
    result_df = pd.DataFrame({'Sentiment': pd.Series(real_scores),
                              'calculation': pd.Series(calculated_scores),
                              'error': pd.Series(error)})
    return result_df


##########
## MAIN ##
##########,

# devide train into 5 piece and create a list with them
# fake train and fake test sets will be created using this list in loop
dfset = get_dfset(df_complete_train)

# create empty lists for mlp model train
# train_inp will be the feature vectors of mlp train
# train_out will be the output values of the mlp train
train_inp=[]
train_out=[]

# create feature vector parameters and initialize it to zero
# for each phrase these values will first initialized to zero then manipulated
# which will be assigned as feature vector of the phrase at the end of the process
closest_attribution_count = 0
closest_attribution_sum = 0
direct_attribution_count = 0
direct_attribution_sum = 0

for x in range(0, len(dfset)):

    test = dfset[0]
    del dfset[0]
    train = pd.concat(dfset, ignore_index=True)

    train.reset_index(drop=True)
    test.reset_index(drop=True)

    train.to_csv("data/Partial Feature Vectors Data of Train Data/"+str(x+1)+"/train"+str(x+1)+".csv", sep='\t')
    test.to_csv("data/Partial Feature Vectors Data of Train Data/"+str(x+1)+"/test"+str(x+1)+".csv", sep='\t')
    print('fake train and test data sets of main train set '
          'to get feature vectors of fake train part is done: '+str(x+1))

    sentence_df = get_sentences(train)
    sentence_df.to_csv("data/Partial Feature Vectors Data of Train Data/"+str(x+1)+"/sentence"+str(x+1)+".csv", sep='\t')
    print('fake train set sentence set extracted: '+str(x+1))

    w2v_model = get_w2v_model(sentence_df)
    w2v_model.save("data/Partial Feature Vectors Data of Train Data/"+str(x+1)+"/w2vmodel"+str(x+1))
    print('fake train set w2v model created: '+str(x+1))

    words_df = get_words(train)
    words_df.to_csv("data/Partial Feature Vectors Data of Train Data/"+str(x+1)+"/word"+str(x+1)+".csv", sep='\t')
    print('fake train set word set extracted: '+str(x+1))

    partial_mlp_train_matrix = get_matrix(test)
    partial_train_inp = partial_mlp_train_matrix[0]
    partial_train_out = partial_mlp_train_matrix[1]
    partial_mlp_feed_df = pd.DataFrame({'length_of_phrase': pd.Series([item[0] for item in partial_train_inp]),
                                'direct_sum': pd.Series([item[1] for item in partial_train_inp]),
                                'direct_count': pd.Series([item[2] for item in partial_train_inp]),
                                'closest_sum': pd.Series([item[3] for item in partial_train_inp]),
                                'closest_count': pd.Series([item[4] for item in partial_train_inp]),
                                'output': pd.Series(partial_train_out)})
    partial_mlp_feed_df.to_csv("data/Partial Feature Vectors Data of Train Data/"+str(x+1)+"/partial_mlp_feed"+str(x+1)+".csv", sep='\t')
    print('fake test set feature vectors and related real scores as output extracted: '+str(x+1))

    train_inp += partial_mlp_train_matrix[0]
    train_out += partial_mlp_train_matrix[1]
    dfset.append(test)
    print('created feature vectors and outputs added to MLP train sets and partial  is done: '+str(x+1))
    print('')

# get final feature and output df and save
mlp_feed_df = pd.DataFrame({'length_of_phrase': pd.Series([item[0] for item in train_inp]),
                            'direct_sum': pd.Series([item[1] for item in train_inp]),
                            'direct_count': pd.Series([item[2] for item in train_inp]),
                            'closest_sum': pd.Series([item[3] for item in train_inp]),
                            'closest_count': pd.Series([item[4] for item in train_inp]),
                            'output': pd.Series(train_out)})
mlp_feed_df.to_csv("data/mlp_train.csv", sep='\t')
print('MLP train dataframe is created')
print('')

mlp_model = MLPRegressor(hidden_layer_sizes=(20,), activation='identity',
                   alpha=0.001, learning_rate='constant', early_stopping=True)
mlp_model.fit(train_inp, train_out)
joblib.dump(mlp_model, 'mlp_model.pkl')
# to load model later
# mlp_model = joblib.load('mlp_model.pkl')
print('mlp model is created')
print('')


# initialized for calculation reassigning for all sets
# by deviding all sets initially construction can be made for whole sets needed for test calculation could be optimized
# now dublicate calculations exist !
test = df_complete_test
train = df_complete_train
train.reset_index(drop=True)
test.reset_index(drop=True)

# # already under data
# sentence_df = get_sentences(train)
# sentence_df.to_csv("data/train_sentences.csv", sep='\t')
sentence_df = pd.read_csv('data/train_sentences.csv', sep="\t").drop('Unnamed: 0', 1)
sentence_df.reset_index(drop=True)
print('main train set sentence set extracted')

# # already under data
# w2v_model = get_w2v_model(sentence_df)
# w2v_model.save("data/w2vmodel" + str(x + 1))
w2v_model = gensim.models.Word2Vec.load("data/w2v_model")
print('main train set w2v model created')


# # already under data
# words_df = get_words(train)
# words_df.to_csv("data/train_words" + str(x + 1) + ".csv", sep='\t')
words_df = pd.read_csv('data/train_words.csv', sep="\t").drop('Unnamed: 0', 1)
words_df.reset_index(drop=True)
print('main train set word set extracted: ' + str(x + 1))


# calculate
df_calculation = calculate(test)
df_calculation.to_csv("data/test_calculated.csv", sep='\t')
print('calculation is done')

#--------not used
# test = dfset[3]
# del dfset[3]
# train = pd.concat(dfset, ignore_index=True)
#
# train.reset_index(drop=True)
# test.reset_index(drop=True)
#
# train.to_csv("traintmp.csv", sep='\t')
# test.to_csv("testtmp.csv", sep='\t')
#
# sentence_df = get_sentences(train)
# print('sentence done: ' + str(3 + 1))
# w2v_model = get_w2v_model(sentence_df)
# print('w2v model done: ' + str(3 + 1))
# words_df = get_words(train)
#--------

#------not used
# x=0
# test = dfset[0]
# del dfset[0]
# dfset.append(test)
#
# x=1
# test = dfset[0]
# del dfset[0]
# dfset.append(test)
#
# x=2
# test = dfset[0]
# del dfset[0]
# dfset.append(test)
#
# X=3
# test = dfset[0]
# del dfset[0]
# train = pd.concat(dfset, ignore_index=True)
# train.reset_index(drop=True)
# test.reset_index(drop=True)
# sentence_df = pd.read_csv('data/MLPdata/sentence4.csv', sep="\t")
# w2v_model = get_w2v_model(sentence_df)
# w2v_model.save("data/MLPdata/w2vmodel" + str(x + 1))
# print('w2v model done: ' + str(x + 1))
# words_df = get_words(train)
# words_df.to_csv("data/MLPdata/word" + str(x + 1) + ".csv", sep='\t')
# print('word done: ' + str(x + 1))
# partial_mlp_train_matrix = get_matrix(test)
# print('matrix done: ' + str(x + 1))
# partial_train_inp = partial_mlp_train_matrix[0]
# partial_train_out = partial_mlp_train_matrix[1]
# partial_mlp_feed_df = pd.DataFrame({'length_of_phrase': pd.Series([item[0] for item in partial_train_inp]),
#                                     'direct_sum': pd.Series([item[1] for item in partial_train_inp]),
#                                     'direct_count': pd.Series([item[2] for item in partial_train_inp]),
#                                     'closest_sum': pd.Series([item[3] for item in partial_train_inp]),
#                                     'closest_count': pd.Series([item[4] for item in partial_train_inp]),
#                                     'output': pd.Series(partial_train_out)})
# partial_mlp_feed_df.to_csv("data/MLPdata/partial_mlp_feed" + str(x + 1) + ".csv", sep='\t')
# train_inp += partial_mlp_train_matrix[0]
# train_out += partial_mlp_train_matrix[1]
# dfset.append(test)
# print('for loop done: ' + str(x + 1))
# print('')
# w2v_model = gensim.models.Word2Vec.load("data/MLPdata/w2vmodel2")
# words_df = pd.read_csv('data/MLPdata/word2.csv', sep="\t")
# partial_mlp_train_matrix = get_matrix(test)
# partial_train_inp = partial_mlp_train_matrix[0]
# partial_train_out = partial_mlp_train_matrix[1]
# partial_mlp_feed_df = pd.DataFrame({'length_of_phrase': pd.Series([item[0] for item in partial_train_inp]),
#                             'direct_sum': pd.Series([item[1] for item in partial_train_inp]),
#                             'direct_count': pd.Series([item[2] for item in partial_train_inp]),
#                             'closest_sum': pd.Series([item[3] for item in partial_train_inp]),
#                             'closest_count': pd.Series([item[4] for item in partial_train_inp]),
#                             'output': pd.Series(partial_train_out)})
# partial_mlp_feed_df.to_csv("data/MLPdata/partial_mlp_feed"+str(x+1)+".csv", sep='\t')
# dfset.append(test)
#------------------------------
#====not used
# mlpraw = []
# mlpraw.append(pd.read_csv('data/MLPdata/1/partial_mlp_feed1.csv', sep="\t"))
# mlpraw.append(pd.read_csv('data/MLPdata/2/partial_mlp_feed2.csv', sep="\t"))
# mlpraw.append(pd.read_csv('data/MLPdata/3/partial_mlp_feed3.csv', sep="\t"))
# mlpraw.append(pd.read_csv('data/MLPdata/4/partial_mlp_feed4.csv', sep="\t"))
# mlpraw.append(pd.read_csv('data/MLPdata/5/partial_mlp_feed5.csv', sep="\t"))
# mlpraw = pd.concat(mlpraw, ignore_index=True).drop('Unnamed: 0', 1)
# mlpraw.to_csv("data/MLPdata/merged_mlp.csv", sep='\t')

# mlpraw = pd.read_csv('data/MLPdata/merged_mlp.csv', sep="\t").drop('Unnamed: 0', 1)
# mlpraw.reset_index(drop=True)
# mlpraw.output = mlpraw.output.multiply(1.25)
# print(mlpraw.output.describe())
# #====
# train_out = mlpraw.output.tolist()
# mlpraw = mlpraw.drop('output', 1)
# for row in mlpraw.iterrows():
#     index, data = row
#     train_inp.append(data.tolist()[::-1])
#=====
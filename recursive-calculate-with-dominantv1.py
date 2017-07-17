import numpy as np
import gensim, logging
import pandas as pd
from scipy.spatial import distance

model = gensim.models.Word2Vec.load("w2vmodel")
words = pd.read_csv('data/words.csv', sep="\t")
dominant_words = pd.read_csv('data/words_dominant.csv', sep="\t")
train = pd.read_csv('data/train-clean.csv', sep="\t")
test = pd.read_csv('data/test.csv', sep="\t").drop('Unnamed: 0', 1).drop('Unnamed: 0.1', 1)

test['calculation'] = pd.Series(np.NaN, index=test.index)
test['error'] = pd.Series(np.NaN, index=test.index)


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
        vec = model[word]
        tmpvec = model[words.loc[0, "Phrase"]]
        tmpindx = 0
        dist = distance.euclidean(vec, tmpvec)
        for indx1, row in words.iterrows():
            tmpdist = distance.euclidean(vec, model[row.Phrase])
            if tmpdist < dist:
                dist = tmpdist
                tmpvec = model[row.Phrase]
                tmpindx = indx1
        return words.loc[tmpindx, "Sentiment"]
    except:
        return np.NaN


def collect_independent_word_scores(phrase_words):
    global attribution_count, attribution_sum
    for m in phrase_words:
        score = get_direct_score(m)
        if ~np.isnan(score):
            attribution_sum += score
            attribution_count += 1
        else:
            attribution_sum += get_similar_word_score(m)
            attribution_count += 1

# if there is at least one dominant word return true by calculating sum and count else false
# if more then one dominant word sum their scores and increment count so they will be averaged
def if_dominant_word_score(phrase_words):
    global attribution_count, attribution_sum
    found = False
    for n in phrase_words:
        dominant_found = dominant_words.word[dominant_words.word == n]
        if len(dominant_found.index):
            found = True
            attribution_count += float(dominant_words.loc[dominant_found.index[0], 'score'])
            attribution_count += 1
    return found



# optimisation needed there exists duplicate code blocks
# when code hits the longest window phrase it directly calculate it by stoping shift of window
# in that case for the remaning beginning and end phrase parts there meigth be a miss for longest window phrases
# at the end there is a need for weighted dominant word calculation
def collect_phrase_scores(phrase):
    global attribution_count, attribution_sum
    score = get_direct_score(phrase)
    if ~np.isnan(score):
        # if direct score found add to sum and increment count
        attribution_sum += score
        attribution_count += 1
        return
    else:
        # if no direct score found split phrase
        phrase_words = phrase.split()
        length = len(phrase_words)
        # if there exist any dominant words return (scores are calculated inside function)
        if if_dominant_word_score(phrase_words):
            return
        if length == 1:
            # if phrase contains single word
            # and it doesn't have direct score since code doesn't return from the above part (already checked case)
            attribution_sum += get_similar_word_score(phrase_words)
            attribution_count += 1
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
                    attribution_sum += score
                    attribution_count += 1
                    phrase_initial_words = phrase_words[0:y]
                    phrase_final_words = phrase_words[length - x + y:length]
                    collect_phrase_scores(" ".join(phrase_initial_words))
                    collect_phrase_scores(" ".join(phrase_final_words))
                    break
            if np.isnan(score):
                # no word group direct score found get independent word scores then break
                collect_independent_word_scores(phrase_words)


# Calculate sentiment score of each row and write them on calculate row in dataset
for indx, row in test.iterrows():
    attribution_count = 0
    attribution_sum = 0
    if isinstance(row.Phrase, str):
        if len(row.Phrase):
            collect_phrase_scores(row.Phrase)
            test.loc[indx, 'calculation'] = attribution_sum / attribution_count

# write magnitude of error at error column for each row
for indx, row in test.iterrows():
    test.loc[indx, "error"] = abs(row.Sentiment - row.calculation)

test.to_csv("data/test-calculated-recursive-with-dominantv1.csv", sep='\t')

import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# get flattened list form of Phrases' of data
df = pd.read_csv('data/train-clean.csv',sep="\t", encoding='latin1')
df_tokenized = df.apply(lambda row: nltk.word_tokenize(row['Phrase']), axis=1)
df_list=df_tokenized.tolist()
df_word_list = []
for sublist in df_list:
    for item in sublist:
        df_word_list.append(item)

# get most 500 frequent word list
freq_list = nltk.FreqDist(df_word_list)
common_words_list = freq_list.most_common(500)
# min frequency bound could be added instead of getting top X frequent words
# least frequent word of comman_words_list is around 150
# might not be enough to decide if it is dominant

# in common_words_list
dominant_words_list =[]
dominant_words_eqn_list=[]
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

    eqn = np.polyfit(lengths, scores, 5)
    err_sum=0
    for indx, length in enumerate(lengths):
        a=np.polyval(eqn, length)
        b=scores[indx]
        err_sum += np.abs(a-b)
    err_rate = err_sum / len(lengths)
    if err_rate < 0.6:

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

dominant_words_df = pd.DataFrame({'eqn': pd.Series(dominant_words_eqn_list),
                                'word': pd.Series(dominant_words_list)})

dominant_words_df.to_csv("data/words_dominant.csv", sep='\t')

print("done successfully")
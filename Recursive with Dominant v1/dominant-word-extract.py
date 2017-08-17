# as version 1 to find what are dominant words
# first extract most 500 frequently used words
# for each frequent word calculate the varience of the sores of phrases that includes the word
# choose the words that varience of the related phrases scores below a value
# attribute the score of the dominant words as mean of the scores of phrases that includes the word

import nltk
import pandas as pd

# get flattened list form of Phrases' of data
df = pd.read_csv('data/train-clean.csv',sep="\t", encoding='latin1')
df_tokenized = df.apply(lambda row: nltk.word_tokenize(row['Phrase']), axis=1)
df_list=df_tokenized.tolist()
df_word_list=[]
for sublist in df_list:
    for item in sublist:
        df_word_list.append(item)

# get most 500 frequent word list
freq_list = nltk.FreqDist(df_word_list)
common_words_list = freq_list.most_common(500)
# min frequency bound could be added least frequent word of comman_words_list is around 150
# might not be enoughn to decide if it is dominant

# get dominant words by using the varience of sentiment scores that common words are used in
# Phrase length could be added as parameter to decision
dominant_words_list =[]
means_of_dominant_words_list = []
for word in common_words_list:
    tmpdf = df[df['Phrase'].str.contains(word[0])]
    a = tmpdf.Sentiment.var()
    if a < 0.6:
        dominant_words_list.append(word[0])
        means_of_dominant_words_list.append(tmpdf.Sentiment.mean())

# create a dataframe of the words and related scores
# (means of scores of the phrases that the dominant words used in)
# then save as words_dominant.csv
dominant_words_df = pd.DataFrame({'word': pd.Series(dominant_words_list),
                                'score': pd.Series(means_of_dominant_words_list)})

dominant_words_df.to_csv("data/words_dominant.csv", sep='\t')

print("done successfully")
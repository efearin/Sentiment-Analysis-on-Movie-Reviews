import pandas as pd
import nltk

df = pd.read_csv('data/train-clean.csv', sep="\t")
df_tokenized = df.apply(lambda row: nltk.word_tokenize(row['Phrase']), axis=1)
freq = nltk.FreqDist(df_tokenized)

for word, frequency in freq.most_common(50):
    print(u'{};{}'.format(word, frequency))

a=4
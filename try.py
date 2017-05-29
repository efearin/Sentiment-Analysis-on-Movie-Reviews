import pandas as pd
df=pd.read_csv('data\cleaneddf.csv',sep="\t")

df.Sentiment=df.Sentiment.astype(float)
phrase=[]
sum_score=0
count_same_phrase=0
avarage_score=0
for index,row in df.head().iterrows():
    if len(phrase)==0:
        phrase.append(row.Phrase)
        sum_score+=row.Sentiment
        count_same_phrase+=1
    else:
        if row.Phrase==phrase[0]:
            sum_score+=row.Sentiment
            count_same_phrase+=1
        else:
            avarage_score=sum_score/count_same_phrase
            df.ix[index-1-count_same_phrase].Sentiment=avarage_score
            while (count_same_phrase > 1):
                df.ix[index-count_same_phrase].Phrase=np.nan
                count_same_phrase-=1
        sum_score=row.Sentiment
        count_same_phrase=1
        avarage_score=row.Sentiment
        phrase[0]=row.Phrase      
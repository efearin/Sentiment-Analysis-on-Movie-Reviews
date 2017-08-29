import pandas as pd
import result_repo as rr

# df=pd.read_csv('data/main_turns/1/result_df.csv',sep="\t")
# rr.output_results(df,0.5,'data/dummy')

df=pd.read_csv('data/data.csv',sep="\t").head()
print(type(df.loc[0,'Phrase']))
b=df.loc[1,'Phrase']
print(df.dtypes)
a=df["Phrase"].astype('str').str.contains(b)
print(a)
print(type(a))
c=df[a]

k=len(df)

import pandas as pd

df = pd.read_csv('final.csv')

#print(df.info())

new = df['Hotel Name'].str.split('-',n=1,expand=True)

df['Hotel Name'] = new[0]
print(df.head())

df.to_csv('really_final.csv')
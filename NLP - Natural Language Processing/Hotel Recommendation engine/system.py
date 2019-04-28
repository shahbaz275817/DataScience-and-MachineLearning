import pandas as pd
#from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import requests
import json
from flask import Flask, request, jsonify

# df = pd.read_csv('cleaned_hotel_data.csv')
# df2 = pd.DataFrame(data=None, columns=df.columns)
# df.drop(columns = ['id'], inplace = True)
# #df.set_index('Hotel Name',inplace=True)
# # print(df.head())
# #TODO: TRANSFORM THE DATAFRAME ACCORDING TO HOTEL NAMES
# df['Reviews_Key_words'] = df['Reviews_Key_words'].map(lambda x: x.lstrip('[]"“”!'))
# df['Brief_Key_words'] = df['Brief_Key_words'].map(lambda x: x.lstrip('[]"”“!'))
# df['Brief_Key_words'] = df['Brief_Key_words'].map(lambda x: x.lstrip(']"“”!'))
#
# print(df.head())

# grouping same hotels together
# df = df.groupby('Hotel Name').agg({'Category':'first',
#                                    'City':'first',
#                                    'Facilities':'first',
#                                    'Overall Rating':'first',
#                                    'Type':'first',
#                              'Reviews_Key_words': ', '.join,
#                              'Brief_Key_words': ', '.join,
#                              'Info':'first' }).reset_index()
#
#
#
# df.to_csv('final.csv')
#
# #TODO: CREATEA A BAG OF WORDS COLUMN


data = pd.read_csv('final.csv')
data.drop(columns = ['id'], inplace = True)

data.set_index('Hotel Name',inplace=True)

#
# instantiating and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(data['Reviews_Key_words'])

# generating the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# creating a Series for the movie titles so they are associated to an ordered numerical
# list I will use in the function to match the indexes
# data.set_index('Hotel Name',inplace=True)
indices = pd.Series(data.index)

#  defining the function that takes in hotel name
# as input and returns the top 10 recommended hotels
def recommendations(title, cosine_sim=cosine_sim):
    # initializing the empty list of recommended movies
    recommended_hotels = []

    # gettin the index of the movie that matches the title
    idx = indices[indices == title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)

    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_hotels.append(list(data.index)[i])

    return recommended_hotels



print(recommendations('Abbazia_di_Novacella-Varna_South_Tyrol_Province_Trentino_Alto_Adige'))


###############



app = Flask(__name__)
@app.route('/api/', methods=['POST'])
def api():
     hotel = request.json['hotel']
     prediction = recommendations(hotel)
     return jsonify({'prediction': list(prediction)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
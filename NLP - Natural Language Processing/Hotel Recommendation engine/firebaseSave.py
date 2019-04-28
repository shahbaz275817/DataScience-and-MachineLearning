import pandas as pd
import requests

df = pd.read_csv('really_final.csv')

URL = "https://us-central1-hotel-finder-e69a6.cloudfunctions.net/saveHotelData"

for index, row in df.iterrows():
   data = {"id":int(index),
    "hotel_name":row['Hotel Name'],
    "category":row['Category'],
    "city":row['City'],
    "facilities":row['Facilities'],
    "rating": row['Overall Rating'],
    "type": row['Type'],
    "inf": row['Info']}

   r = requests.post(url = URL, data = data)
   # extracting response text
   pastebin_url = r.text
   print(pastebin_url+" "+str(index))


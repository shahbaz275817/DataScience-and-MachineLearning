import requests
url = 'http://localhost:5000/api/'
r = requests.post(url,json={'hotel':'Abbazia_di_Novacella-Varna_South_Tyrol_Province_Trentino_Alto_Adige'})
print(r.json())
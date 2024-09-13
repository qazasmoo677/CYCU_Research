from pymongo import MongoClient
from numpy import *
import random

client = MongoClient('localhost', 27017)
db_final = client.Yelp_Final
review_final = db_final.review

def load_rating_data():
    prefer = []
    tempIds = review_final.find({}, no_cursor_timeout=True, batch_size=10)
    for item in tempIds:
        uid = int(item['newUserId'])
        mid = int(item['newBusinessId'])
        rat = float(item['stars'])
        prefer.append([uid, mid, rat])
    tempIds.close()
    data = array(prefer)
    return data
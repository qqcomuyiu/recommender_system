import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import optuna
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import operator


def addtocart(grouped):
    addtocart = grouped['addtocart']
    # creating dictionary for key value pair 
    count_addtocart ={}
    #since addtocart is a list, we will convert it into numpy array for further manipulations
    addtocart = np.array(addtocart[:])
    #counting uniques values of addtocart items of this numpy addtocart array
    unique, counts = np.unique(addtocart, return_counts=True)
    # converting unique and counts as a dictionay with key as unique and value as counts
    count_addtocart = dict(zip(unique, counts))
    #sorting the dictionary
    sort_count_addtocart = sorted(count_addtocart.items(), key = operator.itemgetter(1), reverse = True)
    # keeping number of unique views on X-axis
    x = [i[0] for i in sort_count_addtocart[:7]]
    # keeping count number of views on Y-axis
    y = [i[1] for i in sort_count_addtocart[:7]]
    sns.barplot(x=x, y=y,order=x, palette="pastel")
    
    
def view(grouped):
    views = grouped['view']
    # creating dictionary for key value pair 
    count_view ={}
    #since views is a list, we will convert it into numpy array for further manipulations
    views = np.array(views[:])
    #counting uniques values of views of this numpy views array
    unique, counts = np.unique(views, return_counts=True)
    # converting unique and counts as a dictionay with key as unique and value as counts
    count_view = dict(zip(unique, counts))
    #sorting the dictionary
    sort_count_view = sorted(count_view.items(), key = operator.itemgetter(1), reverse = True)
    # keeping number of unique views on X-axis
    x = [i[0] for i in sort_count_view[:7]]
    # keeping count number of views on Y-axis
    y = [i[1] for i in sort_count_view[:7]]
    sns.barplot(x=x, y=y, order=x, palette="rocket")
    
    
def transaction(grouped):
    transaction = grouped['transaction']

    count_transaction ={}
    #since addtocart is a list, we will convert it into numpy array for further manipulations
    transaction = np.array(transaction[:])
    #counting uniques values of addtocart items of this numpy addtocart array
    unique, counts = np.unique(transaction, return_counts=True)
    # converting unique and counts as a dictionay with key as unique and value as counts
    count_transaction = dict(zip(unique, counts))
    #sorting the dictionary
    sort_count_transaction = sorted(count_transaction.items(), key = operator.itemgetter(1), reverse = True)
    # keeping number of unique views on X-axis
    x = [i[0] for i in sort_count_transaction[:7]]
    # keeping count number of views on Y-axis
    y = [i[1] for i in sort_count_transaction[:7]]
    sns.barplot(x=x, y=y, order=x, palette="vlag")
    
    
    
    
def recommend_items(item_id, purchased_items):
    recommendation_list =[]
    for x in purchased_items:
        if item_id in x:
            recommendation_list +=x
    
    # remove the pass item from the list and merge the above created list
    recommendation_list = list(set(recommendation_list) - set([item_id]))
    return recommendation_list



def data_process(grouped):
    grouped_df = grouped.reset_index()
    grouped_expanded = grouped_df.explode("itemid").reset_index(drop=True)
    le = LabelEncoder()
    grouped_expanded["event_encoded"] = le.fit_transform(grouped_expanded["event"])
    grouped_expanded["event_encoded"] = grouped_expanded["event_encoded"].astype(int)
    grouped_expanded["itemid"] = grouped_expanded["itemid"].astype(int)
    sequences = grouped_expanded[["event_encoded", "itemid"]].to_numpy()

    return torch.tensor(sequences, dtype=torch.float32)
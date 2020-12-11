import pandas as pd
import numpy as np

from sklearn import preprocessing

missing_values = ["n/a", "na", "--", " ?","?"]

data = pd.read_csv('data/house_dataset.csv', na_values=missing_values)

print(data.shape)
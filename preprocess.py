import pandas as pd
import numpy as np

from sklearn import preprocessing

missing_values = ["n/a", "na", "--", " ?","?", " "]

data = pd.read_csv('data/house_dataset.csv', na_values=missing_values)

#print(data.isna().sum()) #no na value

#checking unique value of id, understanding if there is two input from same house

#print(data.duplicated().any()) #False so there is no exact data duplicates
#print(data["id"].duplicated().any()) #True so there is a change between prices between houses
#since this is false we now know that some of the houses has multiple values in datasets


dups = pd.concat(x for idx, x in data.groupby("id") if len(x) > 1)
dups = dups[["id","date","house_price","yr_renovated"]]#wanna see that if the renovation is the reason to the price change
renovated = dups[dups["yr_renovated"]!=0]
#print(renovated, len(renovated)) #we only haave 8 house and their renovation year is much earlier than the price year so change of the price is not related to them it is prob caused by naturally with time

#lets check if any of the dups has same price tags

prices = pd.concat(x for idx, x in dups.groupby(["id","house_price"]) if len(x) > 1)
#print(prices) #we only have 6 house that have the same price.

#=========================================================================0
#we can visualize the change of prices with the time for same houses.

import seaborn as sns
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

#group by month
dups['date'] = pd.to_datetime(dups['date']) #btw we should also change the data itself for later use
data['date'] = pd.to_datetime(data["date"])

#print(dups["date"])
#print(dups.groupby(dups['date'].dt.strftime('%B'))[['id','house_price']].mean().sort_values('house_price'))
# plot
plt.plot(dups["date"],dups["house_price"])
plt.gcf().autofmt_xdate()

#plt.show()

#============================================================0
# lets see the variables impact to the price

corr = data.corr()
hh = sns.heatmap(corr, cmap="YlGnBu")
#plt.show()

#lets get the higher correlations with house_price column
st  = pd.DataFrame(corr['house_price'], index=corr.index)
print(st.sort_values('house_price'))
#now we can see that house price more relatable with;

# living15_sqft     0.585417
# above_sqft        0.605467
# grade             0.667358
# living_sqft       0.701993


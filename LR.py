from sklearn.linear_model import LinearRegression #I'm gonna use LinearRegression for predicting house prices
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import preprocess as pre

data = pre.data #getting the cleaned data
y = data["house_price"]
X = data.drop(["id","date","house_price"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
############################## Plot of train and test

plt.hist(y_train, bins=100, alpha=0.5, label='train')
plt.hist(y_test, bins=100, alpha=0.5, label='test')
plt.legend(loc='upper right')
plt.gca().set(title='Train/Test Price Histogram')
plt.show()

#let's change the values to the percentages
from matplotlib.ticker import PercentFormatter

plt.hist(y_train, weights=np.ones(len(y_train)) / len(y_train),bins=50, alpha=0.5, label='train')
plt.hist(y_test, weights=np.ones(len(y_test)) / len(y_test),bins=50, alpha=0.5, label='test')
plt.legend(loc='upper right')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.gca().set(title='Train/Test Price Histogram')
plt.show()



################################# print result #######
from sklearn.metrics import mean_squared_error as mean_sq_err
from sklearn.metrics import mean_absolute_error as mean_abs_err
import math
def print_result_metrics(num,X, y,y_pred):

    mse = mean_sq_err(y, y_pred)
    rmse = math.sqrt(mse / num)
    rse = math.sqrt(mse / (num - 2))
    rsquare = LR.score(X, y)
    mae = mean_abs_err(y, y_pred)

    return(print('RSE=',rse),
            print('R-Square=',rsquare),
            print('rmse=',rmse),
            print('mae=',mae))

def print_coef_int(coef,int,cols):


    coef = coef.squeeze().tolist()

    labels_coef = list(zip(cols, coef))

    return (  print('Intercept is:', int),
              print(labels_coef))

from mlxtend.feature_selection import SequentialFeatureSelector as sfs

################## with step-wise
sw_LR = LinearRegression()

step_LR1 = sfs(sw_LR,k_features = 10,forward=True,floating=False, verbose=2, scoring='r2',cv=5) # I did this with 10 and later on 4 as k_features
step_LR = step_LR1.fit(X, y)
#these are the selected features
feat_cols = list(step_LR.k_feature_idx_)

#now we can work with our selected features
LR = LinearRegression()
LR.fit(X_train.iloc[:, feat_cols],y_train)

X_p = X_train.iloc[:, feat_cols] #for getting the column names from sw

cols = list(X_p)

print_coef_int(LR.coef_,LR.intercept_,cols)


from sklearn.metrics import r2_score as rr


y_train_pred = LR.predict(X_train.iloc[:, feat_cols])
#print('Training R2 on selected features: %.3f' % rr(y_train, y_train_pred))

y_test_pred = LR.predict(X_test.iloc[:, feat_cols])
#print('Testing R2 on selected features: %.3f' % rr(y_test, y_test_pred))
num_data_train = X_train.iloc[:,feat_cols].shape[0]
num_data_test = X_test.iloc[:,feat_cols].shape[0]
print("STEPWISE TRAIN RESULT")
print_result_metrics(num_data_train,X_train.iloc[:, feat_cols],y_train, y_train_pred)
print("STEPWISE TEST RESULT")
print_result_metrics(num_data_test,X_test.iloc[:, feat_cols],y_test, y_test_pred)


####################################################################

#lets do our LR without step-wise optimization

LR = LinearRegression()
LR.fit(X_train,y_train)

y_train_pred = LR.predict(X_train)
y_test_pred = LR.predict(X_test)

cols = list(X_train)

print_coef_int(LR.coef_,LR.intercept_,cols)

num_data_train = X_train.shape[0]
print("NO/STEPWISE TRAIN RESULT")
print_result_metrics(num_data_train,X_train,y_train, y_train_pred)
print("NO/STEPWISE TEST RESULT")
num_data_test = X_test.shape[0]
print_result_metrics(num_data_test,X_test,y_test, y_test_pred)


########################## doing with higher correlation features
#features from correlation

print("####################################")
print("SELECTED FEATURES FROM CORRELATION")

corr_cols = ["living_sqft","grade","above_sqft","living15_sqft"]
X = data[corr_cols]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)


#with stepwise

sw_LR = LinearRegression()

step_LR1 = sfs(sw_LR,k_features = 2,forward=True,floating=False, verbose=2, scoring='r2',cv=5)
step_LR = step_LR1.fit(X, y)

#these are the selected features
feat_cols = list(step_LR.k_feature_idx_)

#now we can work with our selected features
LR = LinearRegression()
LR.fit(X_train.iloc[:, feat_cols],y_train)

cols = list(X_train.iloc[:, feat_cols])

print_coef_int(LR.coef_,LR.intercept_,cols)

y_train_pred = LR.predict(X_train.iloc[:, feat_cols])

y_test_pred = LR.predict(X_test.iloc[:, feat_cols])

num_data_train = X_train.iloc[:,feat_cols].shape[0]
num_data_test = X_test.iloc[:,feat_cols].shape[0]
print("STEPWISE TRAIN RESULT")
print_result_metrics(num_data_train,X_train.iloc[:, feat_cols],y_train, y_train_pred)
print("STEPWISE TEST RESULT")
print_result_metrics(num_data_test,X_test.iloc[:, feat_cols],y_test, y_test_pred)

#without step-wise optimization

LR = LinearRegression()
LR.fit(X_train,y_train)

y_train_pred = LR.predict(X_train)
y_test_pred = LR.predict(X_test)

cols = list(X_train)

print_coef_int(LR.coef_,LR.intercept_,cols)

num_data_train = X_train.shape[0]
print("NO/STEPWISE TRAIN RESULT")
print_result_metrics(num_data_train,X_train,y_train, y_train_pred)
print("NO/STEPWISE TEST RESULT")
num_data_test = X_test.shape[0]
print_result_metrics(num_data_test,X_test,y_test, y_test_pred)

###########################################################################

cols = list(X_p) #getting the column names from sw approach
print(cols)
#  for 4 k_features => ['living_sqft', 'view', 'grade', 'lat']

###########################################################################
#Table

from prettytable import PrettyTable

columns = ("Regression Type","Stepwise","Feature numbers","Train/test", 'RSE','R-Square','rmse','mae')
table_vals = [["Linear","Yes","10/18","Train",1644.4985703200318,0.6875607738543572,1644.3898035901718,
127078.17976722633,],
            ["Linear","Yes","10/18","Test",2535.4796911387966,
0.7118906211428095,
2535.0883832128397,
125779.95714334877
],
              ["Linear","No","18","Train",1632.5040473364163,0.6921018451684073,
 1632.3960739213485,
125983.67612357177
],
              ["Linear","No","18","Test",2528.3786569938957,
0.7135021581133348,
2527.98844499111,
124921.93596731553
],
              ["Linear","Yes","2/4","Train",2014.3169838998056
,0.5312360749400357
,2014.1837574712793
,163432.88381560676
],
              ["Linear","Yes","2/4","Test",3201.9893297042095
,0.5405091322534787
,3201.49515741487
,164638.17739081755
],
              ["Linear","No","4","Train",2002.3264905348226
,0.5368002257254657
,2002.1940571545704
,161730.88278946574
],
              ["Linear","No","4","Test",3164.1505428120095
,0.5513048247228332
,3163.6622102922697
,161891.90315839328
],["Linear","Yes","4/18","Train",1789.8039899087335
,0.629908170973726
,1789.6856127142585
,137062.58055520704
],
    ["Linear","Yes","4/18","Test", 2804.571080872571
,0.6474911484288584
,2804.1382432928854
,136220.25043225786
]]

t = PrettyTable(columns)

for row in table_vals:
    t.add_row(row)

print(t)


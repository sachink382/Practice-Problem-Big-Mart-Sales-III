#Big Mart Practice Problem by Analytics Vidhya 

#Importing Libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import statistics  

#Importing dataset
dataset = pd.read_csv('Train_UWu5bXk.txt')
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 11]


#Missing Values

x.apply(lambda x: sum(x.isnull()))
#We found two variables with missing values – Item_Weight and Outlet_Size
#Lets impute the former by the average weight of the particular item
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#imputer = imputer.fit(x[:, 0])
#x[:, 0] = imputer.transform(x[:, 0])

mean1 = x["Item_Weight"].mean()
x["Item_Weight"] = x["Item_Weight"].replace(np.nan, mean1)

# Lets impute Outlet_Size with the mode of the Outlet_Size for the particular type of outlet
mode1 = statistics.mode(x["Outlet_Size"])
x["Outlet_Size"] = x["Outlet_Size"].replace(np.nan, mode1)


#Feature Engineering 
#x.pivot_table(index='Outlet_Type') - for tables

#Item_Visibility - We noticed that the minimum value here is 0, which makes no practical sense. 
#Lets consider it like missing information and impute it with mean visibility of that product.

mean2 = x["Item_Visibility"].mean()
x['Item_Visibility'] = x['Item_Visibility'].replace({0:0.0661})

#we should look at the visibility of the product in that particular store as compared to the mean visibility of that product across all stores. 
#This will give some idea about how much importance was given to that product in a store as compared to other stores.

Item_Visibility_MeanRatio = x['Item_Visibility']/mean2    
x['Item_Visibility_MeanRatio'] = Item_Visibility_MeanRatio

#Create a broad category of Type of Item
#Item_Type variable has 16 categories - its a good idea to combine them

x['Item_Type_Combined'] = x['Item_Identifier'].apply(lambda z: z[0:2])
#Rename them to more intuitive categories:
x['Item_Type_Combined'] = x['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})

#a new column depicting the years of operation of a store.
x['Outlet_Years'] = 2013 - x['Outlet_Establishment_Year']

#Modify categories of Item_Fat_Content
#We found typos and difference in representation in categories of Item_Fat_Content variable.
x['Item_Fat_Content'] = x['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})

x.loc[x['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"

#Deleting Item identifier as it has no use 
del x['Item_Identifier']
del x['Item_Type']
del x['Outlet_Establishment_Year']


# Categorical Variable

#from sklearn.preprocessing import LabelEncoder
#labelencoder_x = LabelEncoder()
#x['Item_Fat_Content'] = labelencoder_x.fit_transform(x['Item_Fat_Content'])

# But this line of code will give different countries different numbers and by that our ML algo will 
#understand that one country is greater than another. This method is good for categories like Small, medium, large etc



#To solve this issue we will use Dummy Variables 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x['Item_Fat_Content'] = labelencoder_x.fit_transform(x['Item_Fat_Content'])
x['Outlet_Identifier'] = labelencoder_x.fit_transform(x['Outlet_Identifier'])
x['Outlet_Size'] = labelencoder_x.fit_transform(x['Outlet_Size'])
x['Outlet_Location_Type'] = labelencoder_x.fit_transform(x['Outlet_Location_Type'])
x['Outlet_Type'] = labelencoder_x.fit_transform(x['Outlet_Type'])
x['Item_Type_Combined'] = labelencoder_x.fit_transform(x['Item_Type_Combined'])


#One Hot Coding:
x = pd.get_dummies(x, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet_Identifier'])

#If y is also categorical then 
#But for dependent variable we don't have to use OneHotEncoder as ML knows this
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)

#---------------------------------------------------------------------------------------------------------------

#All the above steps will be done for test set also 

 
#Importing dataset
testset = pd.read_csv('Test_u94Q5KV.txt')

#Missing Values

testset.apply(lambda testset: sum(testset.isnull()))
#We found two variables with missing values – Item_Weight and Outlet_Size

mean3 = testset["Item_Weight"].mean()
testset["Item_Weight"] = testset["Item_Weight"].replace(np.nan, mean3)

# Lets impute Outlet_Size with the mode of the Outlet_Size for the particular type of outlet
mode2 = statistics.mode(testset["Outlet_Size"])
testset["Outlet_Size"] = testset["Outlet_Size"].replace(np.nan, mode2)


#Feature Engineering 
#x.pivot_table(index='Outlet_Type') - for tables

#Item_Visibility - We noticed that the minimum value here is 0, which makes no practical sense. 
#Lets consider it like missing information and impute it with mean visibility of that product.

mean4 = testset["Item_Visibility"].mean()
testset['Item_Visibility'] = testset['Item_Visibility'].replace({0:0.0657})

#we should look at the visibility of the product in that particular store as compared to the mean visibility of that product across all stores. 
#This will give some idea about how much importance was given to that product in a store as compared to other stores.

Item_Visibility_MeanRatiotestset = testset['Item_Visibility']/mean4   
testset['Item_Visibility_MeanRatiotestset'] = Item_Visibility_MeanRatiotestset

#Create a broad category of Type of Item
#Item_Type variable has 16 categories - its a good idea to combine them

testset['Item_Type_Combined'] = testset['Item_Identifier'].apply(lambda z1: z1[0:2])
#Rename them to more intuitive categories:
testset['Item_Type_Combined'] = testset['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})

#a new column depicting the years of operation of a store.
testset['Outlet_Years'] = 2013 - testset['Outlet_Establishment_Year']

#Modify categories of Item_Fat_Content
#We found typos and difference in representation in categories of Item_Fat_Content variable.
testset['Item_Fat_Content'] = testset['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})

testset.loc[testset['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"

#for submission purpose 
submittion_item_identifier = testset['Item_Identifier']

#Deleting Item identifier as it has no use 
del testset['Item_Identifier']
del testset['Item_Type']
del testset['Outlet_Establishment_Year']


# Categorical Variable

#from sklearn.preprocessing import LabelEncoder
#labelencoder_x = LabelEncoder()
#x['Item_Fat_Content'] = labelencoder_x.fit_transform(x['Item_Fat_Content'])

# But this line of code will give different countries different numbers and by that our ML algo will 
#understand that one country is greater than another. This method is good for categories like Small, medium, large etc



#To solve this issue we will use Dummy Variables 
from sklearn.preprocessing import LabelEncoder
labelencoder_testset = LabelEncoder()
testset['Item_Fat_Content'] = labelencoder_testset.fit_transform(testset['Item_Fat_Content'])
testset['Outlet_Identifier'] = labelencoder_testset.fit_transform(testset['Outlet_Identifier'])
testset['Outlet_Size'] = labelencoder_testset.fit_transform(testset['Outlet_Size'])
testset['Outlet_Location_Type'] = labelencoder_testset.fit_transform(testset['Outlet_Location_Type'])
testset['Outlet_Type'] = labelencoder_testset.fit_transform(testset['Outlet_Type'])
testset['Item_Type_Combined'] = labelencoder_testset.fit_transform(testset['Item_Type_Combined'])


#One Hot Coding:
testset = pd.get_dummies(testset, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet_Identifier'])

#Feature Scaling ## No feature scaling for y in categorical 
#from sklearn.preprocessing import StandardScaler
#sc_x = StandardScaler()
#x = sc_x.fit_transform(x)
#testset = sc_x.transform(testset)

#----------------------------------------------------------------------------------------------------------------


#Model Building
#1-baseline model
mean_sales = y.mean()

#Define a dataframe with IDs for submission:
base1 = testset[['Item_Identifier','Outlet_Identifier']]
base1['Item_Outlet_Sales'] = mean_sales

#Export submission file
base1.to_csv("alg0.csv",index=False)

##splitting the dataset
#from sklearn.model_selection import train_test_split 
#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = .2, random_state = 0)


#Linear Regression Model 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)

#predicting the testset result 
y_pred = regressor.predict(testset)

#Define a dataframe with IDs for submission:
base2 = testset[['Item_Identifier','Outlet_Identifier']]
base2['Item_Outlet_Sales'] = y_pred

#Export submission file
base2.to_csv("linearregression.csv",index=False)


#SVR model 
from sklearn.svm import SVR
regressor = SVR(kernel = 'poly')
regressor.fit(x,y)
y_pred = regressor.predict(testset)

base3 = testset[['Item_Identifier','Outlet_Identifier']]
base3['Item_Outlet_Sales'] = y_pred

#Export submission file
base3.to_csv("svr_poly.csv",index=False)

#Decision Tree
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(x,y)

y_pred = regressor.predict(testset)

base3 = testset[['Item_Identifier','Outlet_Identifier']]
base3['Item_Outlet_Sales'] = y_pred

#Export submission file
base3.to_csv("tree.csv",index=False)

#Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 8)

regressor.fit(x,y)

y_pred = regressor.predict(testset)

base3 = testset[['Item_Identifier','Outlet_Identifier']]
base3['Item_Outlet_Sales'] = y_pred

#Export submission file
base3.to_csv("Forest.csv",index=False)





















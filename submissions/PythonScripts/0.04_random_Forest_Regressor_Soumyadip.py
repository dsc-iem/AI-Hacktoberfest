#!/usr/bin/env python
# coding: utf-8

# ![dsc-logo](https://raw.githubusercontent.com/divyake/Cysec-Hacktoberfest/dcc84465cfcff73981f8fcb5c8fe3b1710c007e1/assets/logo.svg)
# 
# <img src='https://upload.wikimedia.org/wikipedia/commons/d/d8/Deerfire_high_res_edit.jpg' width='1200px' style="vertical-align:middle"/>

# **Hello** reader, In this notebook, I have covered the model-building part for Forest Fire Prediction and the Uni-variate analysis of the target variable(area). I have plotted different charts and graphs in order to describe how the target variable is changing and what are the properties of that variable.
# <br>
# 
# ### Table of Content:
# 1. Analysis on the data
#       * Basic info about data
#       * what are the Numerical features?
#       * what are the Continuous Numerical feafeatures?
#       * what are the Discrete Numerical feafeatures?
#       * what are the Categorical Features?And quantitative analysis on those features.
# <br>
# 2. Statistical insights of all the features
# <br>
# 3. Uni-variate analysis of Area/Target variable
#       * Scatter Plot
#       * Line Plot
#       * PDF (Probability density function) Plot
#       * CDF (cumulative distribution function) Plot
#       * Histogram Plot
#       * Violin Plot
# <br>
# 4. Data Preprocessing
#       * Encoding of data
#       * Feature selection
#       * Train test split
# <br>     
# 5. Model building
#       * Model Building using Selected features
#       * Cross Validation
#       * Metric Reports
# 
# ### Note: The plots are in cufflinks and plotly, so it won't be visible inside github. To see the plots all you need to do is to clone the repository and open it in your local system or you can open the file inside [kaggle](https://www.kaggle.com/) too.

# # Utils:

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from plotly.offline import iplot
import plotly as py
import plotly.tools as tls
import cufflinks as cf

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# In[2]:


pd.pandas.set_option('display.max_columns',None)
pd.pandas.set_option('display.max_rows',None)


# In[3]:


cm = sns.light_palette("green", as_cmap=True)


# In[4]:


py.offline.init_notebook_mode(connected = True)
cf.go_offline()
cf.set_config_file(theme='solar')
plt.style.use('ggplot')


# # Analysis on the data

# In[5]:


df = pd.read_csv('../input/forest-fire-prediction/forestfires.csv')
df.head().style.set_properties(**{'background-color': 'black',
                           'color': 'lawngreen',
                           'border-color': 'white'})
# .style.background_gradient(cmap='Reds')


# In[6]:


df.info()


# ## what are the Numerical features?

# In[7]:


numerical_features = [features for features in df.columns if df[features].dtypes != 'O']
print('Number of Numerical variables are: ', len(numerical_features))
print('Numerical features are: ', numerical_features)
df[numerical_features].head().style.background_gradient(cmap=cm)


# In[8]:


discrete_feature = [features for features in numerical_features if len(df[features].unique())< 20]
print(f"length of discrete numerical variables are: {len(discrete_feature)}")
print(f"And the discreate features are: {discrete_feature}")
# lets see the head of the data frame consists of discrete numerical values
df[discrete_feature].head().style.background_gradient(cm).highlight_null('green')


# In[9]:


df['X'].value_counts()


# In[10]:


df['Y'].value_counts()


# In[11]:


df['rain'].value_counts()


# In[12]:


# lets see the different values in each discreate variables
print(df['X'].value_counts())
print('\n')
print(df['Y'].value_counts())
print('\n')
print(df['rain'].value_counts())


# In[13]:


#  lets search for year feature
year_feature = [features for features in numerical_features if 'Yr' in features or 'Year' in features or 'yr' in features or 'year' in features]
print(f"year features are : {year_feature}")


# In[14]:


continuous_feature=[features for features in numerical_features if features not in discrete_feature]
print(f"Continuous feature Count {len(continuous_feature)}")
print(f"Continuous feature are: {continuous_feature}")

# lets see the head
df[continuous_feature].head().style.background_gradient(cmap=cm)


# In[15]:


categorical_features = [features for features in df.columns if df[features].dtypes =='O']
print(f"Now categorical variables are: {categorical_features}")
print(f"number of categorical variables are: {categorical_features}")

# see the head
# CANT COLOR A CATEGORICAL VARIABLE
df[categorical_features].head()


# In[16]:


df['month'].describe()


# In[17]:


df['day'].describe()


# In[18]:


# lets see the different values in each categorical variables
print(df['month'].value_counts())
print('\n')
print(df['day'].value_counts())
print('\n')


# In[19]:


df.describe().style.background_gradient(cmap='Reds')


# ## Lets bring some color and draw some graphs
# 
# |Type of variable|Column name|
# |--|--|
# |Numerical Variables|'X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area'|
# |Year variable/features|No year variables|
# |Discrete Variables|'X', 'Y', 'rain'|
# |Continuous Variables|'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'area'|
# |Categorical Variables|'month', 'day'|
# 

# ## Uni-variate analysis of Area:
# 
# 
# 
# 
# 
# 

# ## Scatter Plot of Area:

# In[20]:


df['area'].iplot(kind = 'scatter' , mode = 'markers',title="Scatter plot of area",
                            yTitle='area',xTitle = 'id')


# ### **Few observations:**
# - This shows the different values of the target variable in the form of dots/bubbles so hoover over to get a closer look at the data points.
# - It shows that there are some extreme values in the area column and most of the values are near to zero.And this really makes sense, because we can see forests are getting burned-out but most of the cases the fire can't spread large enough. And from the statistical table we can also verify that, the Huge fire spread outs are really few in number as the total average of the area column is 12.84.
# - And the interesting thing is the areas where fire spreading was large,they don't really follow any pattern. Like-> we can't say if FFDM is around this much then there will be a high probability of having a huge burn of area. But this kind of pattern makes no sense in this data for any value of an area.

# ## Line plot of Area:

# In[21]:


df['area'].iplot(title="Line plot of area",
                            yTitle='area',xTitle = 'id')


# ### Few observations:
# - In this plot how the area is changing row-wise.
# - Its a normal line plot to make you understand how the area is changing(Scatter plot could be enough but sometimes line plot makes more sense to people and sometimes scatter plot, both shows kind of same things but in different ways).
# - This plot also shows that the target variable is continuous in nature.

# ## Density(PDF) Plot of Area:

# In[22]:


import plotly.figure_factory as ff
import numpy as np
np.random.seed(1)


x = np.array(df['area'])
hist_data = [x]
group_labels = ['area'] 

fig = ff.create_distplot(hist_data, group_labels)
fig.show()


# ### Few observations:
# - Now, this is the density aka PDF plot, for some places you might need to zoom in a bit to see the details.
# - This plot shows that the mean value for the area is near to zero(but not zero) and there are fewer huge burned areas.

# ## Histogram of Area:

# In[23]:


pd.DataFrame(df["area"]).iplot(kind="histogram", 
                bins=40, 
                theme="solar",
                title="Histogram of area",
                xTitle='area', 
                yTitle='Count',
                asFigure=True)


# ### Few observations:
# - Histograms are often called a Distribution plot, as this shows the distribution of a particular feature. By distribution, I mean how many points of a feature lies in a particular range.
# - If you observe the plot you will understand that 465 points which are having values from -25(actually 0, but mentioned in the plot) to 24.9, then there are 37 points which range from 25 to 74.9, and so on.
# - This thing again shows that there are few burnouts that cover less area. 
# - Just think, the area is in hector so can you imagine how huge 1090.84 hectors of area is.

# ## Violin Plot of Area:

# In[24]:


import plotly.express as px

# df = px.data.tips()
fig = px.violin(df, y="area", box=True, # draw box plot inside the violin
                points='all', # can be 'outliers', or False
               )
fig.show()


# ### Few observations:
# -  Now, this is an interesting plot, why? cause in this one plot you can see that the commonly used statistical terms are plotted in the same chart.
# - You can see the quantiles, max value, min value, median, KDE, etc. Hoover over the plot and you will get to know what is the max value what is the median and how the KDE is changing.

# ### Lable Encoding:
# Using label encoding to encode the categorical variables.

# In[25]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

for i in categorical_features:
    df[i]=label_encoder.fit_transform(df[i])


# ### Model using selected fetures:

# In[26]:


all_fe = ['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area']


# ### Lets check Corelation of different features with target:

# In[27]:


corr_new_train=df.corr()
plt.figure(figsize=(10,20))
sns.heatmap(corr_new_train[['area']].sort_values(by=['area'],ascending=False).head(60),vmin=-1, cmap='seismic', annot=True)
plt.ylabel('features')
plt.xlabel('Target')
plt.title("Corelation of different fitures with target")
plt.show()


# ### Taking top 6 features for model building.

# In[28]:


fs1 = ['X','Y','month','FFMC','DMC','DC','temp','area'] 


# In[29]:


df_fs1 = df[fs1]
df_fs1.head()


# ## Tran Test(val) Split:

# In[30]:


SEED = 42

data = df_fs1.copy()
y = data['area']
x = data.drop(['area'],axis=1)


from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val = train_test_split(x,y,test_size = 0.2,random_state = SEED)


# #### I have tried normal Random forest, Random forest with no Randomized Search CV,xgboost and stacking of multiple ml models and the results are something like this.
# 
# |reg      |	rmse	|	mse	| r2	    |
# |-|-|-|-|  
# |simple_rf =|109.5595|12003.2882|-0.0182|
# |rscv_rf = |108.3183|11732.8603|0.00465|
# |rscv_xgboost = |108.7711|11831.1615|-0.00368|
# |stacking = |109.0897|11900.57185|-0.0095|
# |rf with rscv and feature selection =|106.35|11311.77|0.04|
# 
# <br>
# PS: rscv is Randomized Search CV and rf is random forest.
# 
# So, the improvements in **rf with rscv and feature selection** are better, that's why I am keeping it.
# 
# 
# 

# ## Model Building using Selected features:

# In[31]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor


# In[32]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]



random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[33]:


reg_rf_rscv = RandomForestRegressor()


# In[34]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
random_search_rf = RandomizedSearchCV(reg_rf_rscv, random_grid,n_iter=5, n_jobs=1, cv=5,verbose=2)


# In[35]:


random_search_rf.fit(x_train,y_train)


# In[36]:


random_search_rf.best_params_


# In[37]:


base_model = RandomForestRegressor(n_estimators= 1200,
                                     min_samples_split= 10,
                                     min_samples_leaf= 2,
                                     max_features= 'auto',
                                     max_depth= 20,
                                     bootstrap= True,
                                    random_state = SEED)
base_model.fit(x_train, y_train)
# base_accuracy = evaluate(base_model, x_val,y_val)


# In[38]:


y_pred_rf_rscv = base_model.predict(x_val)


# In[39]:


def MSE(model_preds, ground_truths):
  return mean_squared_error(model_preds, ground_truths)

def MAE(model_preds, ground_truths):
  return mean_absolute_error(model_preds, ground_truths)

def Other_Err(model_preds, ground_truths):
  return r2_score( ground_truths,model_preds)

def RMSE(model_preds, ground_truths):
  return np.sqrt(mean_squared_error(model_preds, ground_truths))


# In[40]:


print(f"mean squared error: {MSE(y_pred_rf_rscv,y_val)}")
print(f"mean absolute error: {MAE(y_pred_rf_rscv,y_val)}")
print(f"r2 error: {Other_Err(y_pred_rf_rscv,y_val)}")
print(f"root mean squared error: {RMSE(y_pred_rf_rscv,y_val)}")


# ### Cross Validation CV:

# In[41]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# In[42]:


score = cross_val_score(reg_rf_rscv, x_train, y_train, cv=k_fold, n_jobs=1, scoring='r2')
print(score)


# In[43]:


print('train r2 %2f' %(1 * score.mean()))


# In[44]:


score_val = cross_val_score(reg_rf_rscv, x_val, y_val, cv=k_fold, n_jobs=1, scoring='r2')
print(score)


# In[45]:


print('train r2 %2f' %(1 * score_val.mean()))


# In[52]:


pd.DataFrame(score).iplot(title="R2 score of diferent CV for training data",xTitle = "count",yTitle="R2 Score")


# In[51]:


pd.DataFrame(score_val).iplot(title="R2 score of diferent CV for validation data",xTitle = "count",yTitle="R2 Score")


# # Metric Reports
#  
# 
# | Metrics 	| Values 	| 
# |-	|-	|
# | MSE 	| 11311.77	|
# | RMSE 	| 106.35 	|
# | MAE 	| 25.40 	|
# | R2 Score 	| 0.04 	|
# 
# 

# In[53]:


import pickle
filename = 'finalized_model.pkl'
pickle.dump(reg_rf_rscv,open(filename,'wb'))


# # Saved Model:
# 
# model link - https://drive.google.com/file/d/1isIoiZRKjQLzTdb2YiGa-nAnMUGvMmh6/view?usp=sharing

# In[ ]:





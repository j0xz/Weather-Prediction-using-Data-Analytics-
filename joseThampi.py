#!/usr/bin/env python
# coding: utf-8

# In[4]:


conda install pandas


# In[3]:


conda install -c conda-forge pandas-profiling


# In[1]:


#importing the packages
import pandas as pd
import pandas_profiling


# In[2]:


# read the file
df = pd.read_csv("Weather Data.csv",encoding='ISO-8859-1')


# In[3]:


df.head()


# In[4]:


profile = df.profile_report(title='Pandas Profiling Report')
   
# save the report as html file
profile.to_file(output_file="pandas_profiling1.html")
   
# save the report as json file
profile.to_file(output_file="pandas_profiling2.json")


# In[5]:


# Rename columns
df.columns = ['Datetime', 'Temp [C]', 'Dew Point Temp [C]', 'Rel Hum [%]', 'Wind Speed [km/h]', 'Visibility [km]', 'Press [kPa]', 'Weather']


# In[6]:


# Convert time stamp column to time format
df.index = pd.to_datetime(df['Datetime']).dt.floor('T')
df = df.iloc[:, 1:]


# In[7]:


# Remove empty rows (if any are present in the data).
df.drop(df[df.isnull().any(axis = 1)].index, inplace = True)


# In[8]:


# Remove duplicates (if any are present in the dataset).
df.drop_duplicates(inplace = True)


# In[9]:


# Separate 'Weather' column into three separate parts (each description of weather conditions in a separate column)
weather_split = ['Weather - p. 1', 'Weather - p. 2', 'Weather - p. 3']
df[weather_split] = df['Weather'].str.split(',', expand = True)


# In[10]:


# Remove 'Weather' column (redundant).
df.drop(['Weather'], axis = 1, inplace = True)


# In[11]:


# Storage of data on weather conditions according to zero-one coding.
import numpy as np
weather_category_list = np.array([])

for column in df[weather_split]:
    weather_category_list = np.append(weather_category_list, df[weather_split][column].unique())

weather_category_list = weather_category_list[weather_category_list != None]
weather_category_list = np.unique(weather_category_list)

df[weather_category_list] = 0

for column in df[weather_split]:
    for index in df[weather_split].index:
        if df.loc[index, column] != None:
            df.at[index, df.loc[index, column]] = df.loc[index, df.loc[index, column]] + 1


# In[12]:


# Delete 'Weather - p. 1', 'Weather - p. 2' and 'Weather - p. 3' columns (redundant)
df.drop(weather_split, axis = 1, inplace = True)


# In[13]:


# Create an auxiliary set of column names in measured values
weather_measurement_data = ['Temp [C]', 'Dew Point Temp [C]', 'Rel Hum [%]', 'Wind Speed [km/h]', 'Visibility [km]', 'Press [kPa]']


# In[14]:


# Display the dataset in tabular form (data after correction and formatting) - first records
df.head(5)


# In[15]:


# Display of data frame information (data after correction and formatting).
df.info()


# In[21]:


# Display a statistical summary for selected columns
df[weather_measurement_data].describe()


# In[22]:


#Regression Algorithm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[23]:


df['Datetime (month)'] = pd.to_datetime(df.index).month
df['Datetime (day)'] = pd.to_datetime(df.index).day
df['Datetime (hour)'] = pd.to_datetime(df.index).hour

param_ml_input = ['Datetime (month)', 'Datetime (day)', 'Datetime (hour)', 'Temp [C]', 'Rel Hum [%]', 'Press [kPa]', 'Fog', 'Rain', 'Snow']
param_ml_output = 'Visibility [km]'


# In[24]:


X_data = df[param_ml_input]
y_data = df[param_ml_output]

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 1/3)


# In[25]:


scaler =  StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[26]:


from sklearn.neural_network import MLPRegressor


# In[27]:


mlp = MLPRegressor(max_iter = 5000).fit(X_train_scaled, y_train)
y_pred = mlp.predict(X_test_scaled)


# In[28]:


result = pd.DataFrame({'Actual value (y_test)': y_test,
                       'Value predicted by the model (y_pred)': y_pred,
                       'Difference': abs(y_pred - y_test)})
result.head()


# In[29]:


result.describe()[result.describe().index != 'count']


# In[31]:


# Random Forest
from sklearn.ensemble import RandomForestRegressor


# In[32]:


rfr = RandomForestRegressor().fit(X_train_scaled, y_train)
y_pred = rfr.predict(X_test_scaled)


# In[33]:


result = pd.DataFrame({'Actual value (y_test)': y_test,
                       'Value predicted by the model (y_pred)': y_pred,
                       'Difference': abs(y_pred - y_test)})
result.head()


# In[34]:


result.describe()[result.describe().index != 'count']


# In[41]:


# Set color scheme
color_1 = '#BFAF9D'
color_2 = '#C1C1D5'
color_3 = '#A16F86'
color_4 = sns.diverging_palette(h_neg = 32, h_pos = 32, s = 21, l = 68, as_cmap = True)


# In[55]:


# Data Visualization
# Display of box charts
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import numpy as np

plt.figure(figsize = (18, 10))

plt.subplot(231)
sns.boxenplot(data = df, y = 'Temp [C]', color = color_1)

plt.subplot(232)
sns.boxenplot(data = df, y = 'Dew Point Temp [C]', color = color_1)

plt.subplot(233)
sns.boxenplot(data = df, y = 'Rel Hum [%]', color = color_1)

plt.subplot(234)
sns.boxenplot(data = df, y = 'Wind Speed [km/h]', color = color_1)

plt.subplot(235)
sns.boxenplot(data = df, y = 'Visibility [km]', color = color_1)

plt.subplot(236)
sns.boxenplot(data = df, y = 'Press [kPa]', color = color_1)
plt.show()


# In[56]:


# Calculation of moving averages of individual measurement parameters
for column in df[weather_measurement_data]:
    df['SMA168 ' + column] = df[column].rolling('168h', center = True).mean()


# In[57]:


# Display line graphs
plt.figure(figsize = (20, 30))

plt.subplot(611)
sns.lineplot(data = df, x = 'Datetime', y = 'Temp [C]', color = color_2)
sns.lineplot(data = df, x = 'Datetime', y = 'SMA168 Temp [C]', color = color_3, lw = 3)

plt.subplot(612)
sns.lineplot(data = df, x = 'Datetime', y = 'Dew Point Temp [C]', color = color_2)
sns.lineplot(data = df, x = 'Datetime', y = 'SMA168 Dew Point Temp [C]', color = color_3, lw = 3)

plt.subplot(613)
sns.lineplot(data = df, x = 'Datetime', y = 'Rel Hum [%]', color = color_2)
sns.lineplot(data = df, x = 'Datetime', y = 'SMA168 Rel Hum [%]', color = color_3, lw = 3)

plt.subplot(614)
sns.lineplot(data = df, x = 'Datetime', y = 'Wind Speed [km/h]', color = color_2)
sns.lineplot(data = df, x = 'Datetime', y = 'SMA168 Wind Speed [km/h]', color = color_3, lw = 3)

plt.subplot(615)
sns.lineplot(data = df, x = 'Datetime', y = 'Visibility [km]', color = color_2)
sns.lineplot(data = df, x = 'Datetime', y = 'SMA168 Visibility [km]', color = color_3, lw = 3)

plt.subplot(616)
sns.lineplot(data = df, x = 'Datetime', y = 'Press [kPa]', color = color_2)
sns.lineplot(data = df, x = 'Datetime', y = 'SMA168 Press [kPa]', color = color_3, lw = 3)

plt.subplots_adjust(hspace = 0.3)
plt.show()


# In[1]:


pip install tslearn


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from statsmodels.tsa.seasonal import seasonal_decompose
from tslearn.clustering import TimeSeriesKMeans


# In[2]:


import os
for dirname, _, filenames in os.walk('Weather Data.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[9]:


data = pd.read_csv('Weather Data.csv', parse_dates=['Date/Time'], index_col=[0])
data.sample()


# In[10]:


data.index.is_monotonic


# In[11]:


data.info()


# In[12]:


data.duplicated().sum()


# In[13]:


mapper = {
    'Temp_C':'temp_C',
    'Dew Point Temp_C':'dew_point_temp_C',
    'Rel Hum_%':'rel_hum_%',
    'Wind Speed_km/h':'wind_speed_km/h',
    'Visibility_km':'visibility_km',
    'Press_kPa':'press_kPa',
    'Weather':'weather'
}
data.rename(columns=mapper, inplace=True)
data.sample(2)


# In[14]:


#Explorartory Data Analysis

data.describe(datetime_is_numeric=True).T


# In[15]:


data.hist(bins = 20, figsize = (20, 18), color = 'violet');


# In[ ]:


#Rolling Mean


# In[16]:


data_rolling = data.drop('weather', axis=1).rolling(240).mean()


plt.figure(figsize=(20,18))
col = 1
for i in data.drop('weather', axis=1).columns:
    plt.subplot(3, 2, col)
    sns.lineplot(data=data, x='Date/Time', y=i, color='cyan')
    sns.lineplot(data=data_rolling, x='Date/Time', y=i, color='orange')
    col += 1


# In[ ]:


#Trend in data for temperature, dew point, visability, and a bit for humidity


# In[17]:


plt.figure(figsize=(10,8))
sns.heatmap(data[data.columns[:6]].corr());


# In[ ]:


#Vectorizing the Categorical feature


# In[18]:


data['weather'].nunique()


# In[19]:


corpus = data['weather']
vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(corpus)

weather_tfidf = pd.DataFrame(data=vectors.toarray(), index=corpus.index, columns=vectorizer.get_feature_names_out())
weather_tfidf.shape


# In[20]:


weather_tfidf.sample(3)


# In[21]:


for i in weather_tfidf.columns:
    print(i)
    decomposed = seasonal_decompose(weather_tfidf[i])
    
    plt.figure(figsize=(8, 8))

    plt.subplot(311)
    decomposed.trend.plot(ax=plt.gca())
    plt.title('Trend')

    plt.tight_layout()
    plt.show()


# In[ ]:





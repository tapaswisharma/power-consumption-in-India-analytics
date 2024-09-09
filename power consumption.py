#!/usr/bin/env python
# coding: utf-8

# In[7]:



import pandas as pd
import numpy as np
df = pd.read_csv("dataset_tk.csv")
df.head()


# In[9]:


df.head()


# In[10]:


df.drop("Punjab",axis=1,inplace=True)
df.drop("Arunachal Pradesh",axis=1,inplace=True)
df.drop("UP",axis=1,inplace=True)
df.drop("HP",axis=1,inplace=True)


# In[11]:


df.head()


# In[12]:


df.drop("Delhi",axis=1,inplace=True)


# In[13]:


df.head()


# In[14]:


df.info()


# In[15]:


print(df.isnull().sum())


# In[ ]:


##Deleting the column with missing data (if there are many null values)


# In[16]:


updated_df = df.dropna(axis=1)


# In[17]:


updated_df.info()


# In[ ]:


##Deleting the row with missing data (if there are many null values)


# In[18]:


updated_df = df.dropna(axis=0)


# In[19]:


updated_df.info()


# In[ ]:


#Central Tendency-mean


# In[20]:


df['Goa'] = df['Goa'].fillna(df['Goa'].mean())


# In[21]:


df.info()


# In[22]:


print(df.isnull().sum())


# In[ ]:


##Central Tendency-median


# In[23]:


df['Goa'] = df['Goa'].fillna(df['Goa'].median())


# In[24]:


df.info()


# In[25]:


print(df.isnull().sum())


# In[ ]:


##Central Tendency-mode


# In[26]:


df['Goa'] = df['Goa'].fillna(df['Goa'].mode()[0])
df.info()


# In[27]:


print(df.isnull().sum())


# In[ ]:


# 5.   Interpolation.


# In[28]:


import pandas as pd
import numpy as np
df2 = pd.read_csv("dataset_tk.csv")


# In[29]:


df2.head()


# In[30]:


df2.drop("Delhi",axis=1,inplace=True)
df2.drop("Punjab",axis=1,inplace=True)
df2.drop("Goa",axis=1,inplace=True)
df2.drop("UP",axis=1,inplace=True)
df2.drop("HP",axis=1,inplace=True)
df2.drop("Assam",axis=1,inplace=True)
df2.head()


# In[31]:


df2.info()


# In[32]:


print(df2.isnull().sum())


# In[34]:


df2['MP'].interpolate(method='linear', direction = 'forward', inplace=True)
df2.info()


# In[35]:


print(df2.isnull().sum())


# In[ ]:


## KNN-Finite difference


# In[40]:


pip install fancyimpute


# In[39]:


import pandas as pd
import numpy as np
df = pd.read_csv("dataset_tk.csv")
df


# In[38]:





# In[52]:


pip install fancyimpute


# In[53]:


from fancyimpute import KNN as KNN


# In[54]:


knn_imputer = KNN()
data = knn_imputer.fit_transform(df)
data


# In[ ]:


#Outlier estimation and data cleaning using quartile and visualization methods.


# In[3]:


pip install functools 


# In[5]:


import pandas as pd
import numpy as np


# In[7]:


df = pd.read_csv('dataset_tk.csv')
df.head()


# In[8]:


to_drop = ['Punjab',
           'Goa',
           'Assam',
           'UP',
           'MP',
           'HP',
           'Delhi',
           'Manipur']
df.drop(to_drop, inplace = True, axis = 1)
df.head()


# In[ ]:


## 2.   Set the index of the dataset


# In[11]:


df.set_index('DNH', inplace = True)
df.head()


# In[12]:


df['J&K'].head(25)


# In[ ]:


# 3. Cleaning columns using the .apply function


# In[18]:


unwanted_characters = ['[', ',', '-']
defclean_dates(item):
    dop= str(item.loc['J&K'])
                      if dop == 'nan'or dop[0] == '[':
        return np.NaN
    for character in unwanted_characters:
        if character in dop:          
            character_index = dop.find(character)            
            dop = dop[:character_index]
            return dop
        df['J&K'] = df.apply(clean_dates, axis = 1)


# In[19]:


df.head()


# In[ ]:


.   Cleaning entire dataset


# In[24]:


get_ipython().system('more dataset_tk.csv')


# In[27]:


dataset_tk = []
withopen('dataset_tk.csv', 'r') as file: 
    items = file.readlines() 
    states = list(filter(lambda x: '[edit]'in x, items))
    for index, state inenumerate(states):  
        start = items.index(state) + 1
        if index == 49: 
            end = len(items)else:          
                end = items.index(states[index + 1])       
                Tamil= map(lambda x: [state, x], items[start:end])      
                dataset_tk.extend(Tamil)


# In[ ]:


Renaming columns and skipping rows


# In[28]:


dataset_tk_df = pd.read_csv('dataset_tk.csv')
dataset_tk_df.head()


# In[29]:


dataset_tk_df = pd.read_csv('dataset_tk.csv', skiprows = 3, header = 0)
dataset_tk_df.head()


# In[30]:


print(df.shape)
print(df.info())


# In[31]:


df.describe()


# In[ ]:


Box -whisker plot and quartile


# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("dataset_tk.csv")
print(df.shape)
print(df.info())


# In[33]:


df.describe()


# In[ ]:


#IQR


# In[35]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[ ]:


#MILD


# In[36]:


(print(df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR)))


# In[ ]:


#Extreme


# In[37]:


(print(df < (Q1 - 3 * IQR)) |(df > (Q3 + 3 * IQR)))


# In[38]:


print(df['Andhra Pradesh'].skew())
df['Andhra Pradesh'].describe()


# In[41]:


plt.boxplot(df["DNH"])
plt.show()


# In[43]:


df.DNH.hist()


# In[ ]:


#Quantile method 


# In[44]:


print(df['DNH'].quantile(0.10))
print(df['DNH'].quantile(0.90))


# In[45]:


df["DNH"] = np.where(df["DNH"] <7.81, 7.81,df['DNH'])
df["DNH"] = np.where(df["DNH"] >18.9,18.9,df['DNH'])


# In[46]:


print(df['DNH'].skew())


# In[ ]:


#MILD


# In[49]:


df_out = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
   print(df_out.shape)


# In[50]:


df_out


# In[ ]:


#Extreme 


# In[51]:


df_out1 = df[~((df < (Q1 - 3 * IQR)) |(df > (Q3 + 3 * IQR))).any(axis=1)]
print(df_out1.shape)


# In[52]:


df_out1.count


# In[53]:


print(df_out['DNH'].skew())


# In[54]:


print(df_out1['DNH'].skew())


# In[ ]:


#No of extreme outliers


# In[55]:


df_out2= df[((df < (Q1 - 3 * IQR)) |(df > (Q3 + 3 * IQR))).any(axis=1)]
print(df_out2.shape)


# In[56]:


df_out2.count


# In[ ]:


To execute data normalization


# In[ ]:





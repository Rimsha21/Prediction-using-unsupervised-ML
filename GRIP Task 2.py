
# coding: utf-8

# ## Author- Rimsha Virmani
# 
# ## GRIP @ The Sparks Foundation
# 
# ## Task 2: Prediction using Unsupervised Machine Learning
# 
# ## Task Aim: To predict the optimum no of clusters from the Iris Dataset.
# 

# ## Step1: Importing the data

# In[8]:


#Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets
from sklearn.cluster import KMeans
import seaborn as sns


# In[4]:


# Loading iris dataset
data= datasets.load_iris()
data= pd.DataFrame(data.data, columns= data.feature_names)
data.head()


# ## Step 2: Data Exploration

# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


# Checking for null values
data.isnull().any()


# ## Step 3 : Data Visualization

# In[10]:


#Plotting distplot
sns.distplot(data['sepal length (cm)'], color= "red")
sns.set(rc={'figure.figsize': (6,4)})


# In[12]:


sns.distplot(data['sepal width (cm)'], color= "blue")
sns.set(rc={'figure.figsize': (6,4)})


# In[13]:


sns.distplot(data['petal length (cm)'], color= "green")
sns.set(rc={'figure.figsize': (6,4)})


# In[14]:


sns.distplot(data['petal width (cm)'], color= "red")
sns.set(rc={'figure.figsize': (6,4)})


# In[16]:


#Plotting pairplot
sns.pairplot(data, kind='kde')


# In[20]:


# Plotting Correlation
plt.subplots(figsize= (12,10))
corrmat= data.corr()
sns.heatmap(corrmat, annot=True, cmap= None)


# ## Step 4: Finding Optimum Number of Clusters For K-Means

# In[21]:


#Using Elbow Method
x = data.iloc[:, [0,1,2,3]].values
wcss = []
for i in range(1,11):
    kmeans= KMeans(n_clusters=i , init = 'k-means++', max_iter= 200, n_init= 10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[23]:



plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

It can be seen from the above graph that elbow occurs between 2 and 4, This, I am selecting 3 as the optimum number of cluster.
# In[28]:


# Creating KMeans Classifier
kmeans= KMeans(n_clusters=3 , init = 'k-means++', max_iter= 200, n_init= 10, random_state=0)
y_kmeans= kmeans.fit_predict(x)
print(y_means)


# In[31]:


#Visualizing the Clusters 
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s= 100, c = 'red', label ='Iris-setosa')

plt.scatter(x[y_kmeans == 1,0], x[y_kmeans == 1,1], s= 100, c = 'blue', label ='Iris-versicolour')
plt.scatter(x[y_kmeans == 2,0], x[y_kmeans == 2,1], s= 100, c = 'green', label ='Iris-virginica')
# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s= 100, c= 'orange', label=' Centriods')
plt.legend()
sns.set(rc={'figure.figsize':(14,10)})


# ## Conclusion

# In[ ]:


This task has been performed successfully.


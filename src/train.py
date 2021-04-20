#!/usr/bin/env python
# coding: utf-8

# <a id='Q0'></a>
# <center><a target="_blank" href="http://www.propulsion.academy"><img src="https://drive.google.com/uc?id=1McNxpNrSwfqu1w-QtlOmPSmfULvkkMQV" width="200" style="background:none; border:none; box-shadow:none;" /></a> </center>
# <center> <h4 style="color:#303030"> Python for Data Science, Homework, template: </h4> </center>
# <center> <h1 style="color:#303030">Breast Cancer Selection</h1> </center>
# <p style="margin-bottom:1cm;"></p>
# <center style="color:#303030"><h4>Propulsion Academy, 2021</h4></center>
# <p style="margin-bottom:1cm;"></p>
# 
# <div style="background:#EEEDF5;border-top:0.1cm solid #EF475B;border-bottom:0.1cm solid #EF475B;">
#     <div style="margin-left: 0.5cm;margin-top: 0.5cm;margin-bottom: 0.5cm">
#         <p><strong>Goal:</strong> Practice binary classification on Breast Cancer data</p>
#         <strong> Sections:</strong>
#         <a id="P0" name="P0"></a>
#         <ol>
#             <li> <a style="color:#303030" href="#SU">Set Up </a> </li>
#             <li> <a style="color:#303030" href="#P1">Exploratory Data Analysis</a></li>
#             <li> <a style="color:#303030" href="#P2">Modeling</a></li>
#         </ol>
#         <strong>Topics Trained:</strong> Binary Classification.
#     </div>
# </div>
# 
# <nav style="text-align:right"><strong>
#         <a style="color:#00BAE5" href="https://monolith.propulsion-home.ch/backend/api/momentum/materials/intro-2-ds-materials/" title="momentum"> SIT Introduction to Data Science</a>|
#         <a style="color:#00BAE5" href="https://monolith.propulsion-home.ch/backend/api/momentum/materials/intro-2-ds-materials/weeks/week2/day1/index.html" title="momentum">Week 2 Day 1, Applied Machine Learning</a>|
#         <a style="color:#00BAE5" href="https://colab.research.google.com/drive/17X_OTM8Zqg-r4XEakCxwU6VN1OsJpHh7?usp=sharing" title="momentum"> Assignment, Classification of breast cancer cells</a>
# </strong></nav>

# <a id='SU' name="SU"></a>
# ## [Set up](#P0)

# **Package install**

# In[1]:


# get_ipython().system(u'sudo apt-get install build-essential swig')
# get_ipython().system(u'curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install')
# get_ipython().system(u'pip install auto-sklearn==0.12.5')


# In[2]:


# get_ipython().system(u'pip install joblib')


# In[3]:


# get_ipython().system(u'pip install pipelineprofiler')


# In[4]:


# get_ipython().system(u'pip install shap')


# In[5]:


# get_ipython().system(u'pip install --upgrade plotly')


# In[10]:


# get_ipython().system(u'pip3 install -U scikit-learn')


# In[77]:


# get_ipython().system(u'sudo apt-get install cookiecutter')
# get_ipython().system(u'pip install gdown')
# get_ipython().system(u'pip install dvc')
# get_ipython().system(u"pip install 'dvc[gdrive]'")


# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import plotly
plotly.__version__

import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
import logging
import joblib


# In[64]:


from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error
from sklearn import preprocessing


# In[3]:


import autosklearn.classification
import PipelineProfiler
import datetime


# In[ ]:





# In[4]:


import shap


# Connect to your Google Drive

# In[5]:


from google.colab import drive
drive.mount('/content/drive', force_remount=True)


# In[14]:


data_path = "/content/drive/MyDrive/Introduction2DataScience/w2d2/data/"


# In[15]:


model_path = "/content/drive/MyDrive/Introduction2DataScience/w2d2/model/"


# In[8]:


timestr =  str(datetime.datetime.now()).replace(' ','_')


# In[19]:


logging.basicConfig(filename = f'{model_path}log_{timestr}.log' , level = logging.INFO)


# In[10]:


pd.set_option('display.max_rows', 20)


# In[11]:


set_config(display='diagram')


# In[12]:


# get_ipython().magic(u'matplotlib inline')


# _Your Comments here_

# ### Data Structure and types

# **Load the csv file as a DataFrame using Pandas**

# In[17]:


dataset = pd.read_csv(f'{data_path}data-breast-cancer.csv')


# In[20]:


logging.info("Read the dataset")


# Now we separate the categories

# In[21]:


categories = ['diagnosis']


# In[22]:


X = dataset.drop(categories , axis = 1)
y = dataset[categories]

num_variables = ['id', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
       'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']


# 
# _Your Comments here_

# We now can do the test:train split

# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, # original dataframe to be split
                                                     y,
                                                     test_size=0.2, # proportion of the rows to put in the test set
                                                     stratify=y,
                                                     random_state=42) # for reproducibility (see explanation below)


# In[24]:


logging.info("Successfully divided data into training and testing set")


# ### Pipeline Definition

# In[25]:


numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                                      ('scaler', StandardScaler())])


# In[26]:


ohe_transformer = OneHotEncoder(handle_unknown='ignore')


# In[27]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_variables)
        ])


# In[28]:


classification_model = Pipeline(steps=[('preprocessor', preprocessor),
                                          ('classifier', LogisticRegression())])


# In[29]:


classification_model


# In[30]:


logging.info("Model prepared")


# _Your Comments here_

# ### Model Training

# In[31]:


classification_model.fit(X_train, y_train)


# In[32]:


logging.info('model fitted...')


# In[33]:


col_names = num_variables.copy()


# In[34]:


X_train_encoded = pd.DataFrame(classification_model['preprocessor'].transform(X_train), columns=col_names)


# Encode feature 'diagnosis' with label encoder

# In[45]:


le = preprocessing.LabelEncoder()
le.fit(y)
y_train_encoded = le.transform(y_train)


# In[46]:


automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60,
    per_run_time_limit=30,
)


# In[47]:


automl.fit(X_train_encoded, y_train_encoded)


# In[38]:


logging.info('automl fitted...')


# In[48]:


profiler_data= PipelineProfiler.import_autosklearn(automl)
PipelineProfiler.plot_pipeline_matrix(profiler_data)


# In[51]:


logging.info("Model successfully trained...")


# Now, we save the trained model using joblib.

# In[54]:


joblib.dump(automl , f'{model_path}model{timestr}.pkl')


# In[56]:


logging.info(f'Model saved successfully to {model_path} at {timestr}')


# ### Model Evaluation

# In[57]:


X_test_encoded = pd.DataFrame(classification_model['preprocessor'].transform(X_test), columns=col_names)


# In[58]:


y_pred = automl.predict(X_test_encoded)


# In[59]:


y_test_encoded = le.transform(y_test)


# In[66]:


logging.info(f"Mean Squared Error is {mean_squared_error(y_test_encoded, y_pred)}, \n R2 score is {automl.score(X_test_encoded, y_test_encoded)}")


# In[60]:


confusion_matrix(y_test_encoded,y_pred)


# In[67]:


ConfusionMatrixDisplay(confusion_matrix(y_test_encoded,y_pred))


# In[68]:


explainer = shap.KernelExplainer(model = automl.predict, data = X_test_encoded.iloc[:50, :], link = "identity")


# In[76]:


# Set the index of shap
X_idx = 0
shap_value_single = explainer.shap_values(X = X_test_encoded.iloc[X_idx:X_idx+1,:], nsamples = 100)
X_test.iloc[X_idx:X_idx+1,:]
# print the visualization 
shap.initjs()
shap.force_plot(base_value = explainer.expected_value,
                shap_values = shap_value_single,
                features = X_test_encoded.iloc[X_idx:X_idx+1,:], show=False,
                matplotlib=True
                )
plt.savefig(f"{model_path}model_shap{timestr}.png")
logging.info(f" Saved Shap as {model_path}model_shap{timestr}.png")


# In[70]:


shap_values = explainer.shap_values(X = X_test_encoded.iloc[0:50,:], nsamples = 100)


# In[74]:


#print summary
shap.initjs()
fig = shap.summary_plot(shap_values = shap_values,
                  features = X_test_encoded.iloc[0:50,:], show=False
                  )
plt.savefig(f"{model_path}shap_summary_{timestr}.png")
logging.info(f"Shapley summary saved as {model_path}shap_summary_{timestr}.png")


# In[ ]:


logging.info("Model evaluated")


# # Create a cookiecutter data science project directory in your google drive and track its evolution using git

# Change directory

# In[79]:


get_ipython().magic(u'cd /content/drive/MyDrive/Introduction2DataScience/w2d2')


# In[80]:


get_ipython().system(u'cookiecutter https://github.com/drivendata/cookiecutter-data-science')


# Check the project structure

# In[82]:


get_ipython().magic(u'cd w2d2_assignment/')
get_ipython().system(u'ls')


# ## Track Code Evolution using Git

# In[83]:


get_ipython().system(u'git init')


# In[84]:


get_ipython().system(u'git status')


# In[85]:


get_ipython().system(u'git add .')


# In[86]:


get_ipython().system(u'git status')


# In[88]:


get_ipython().system(u'git config --global user.email "dnewlife0@gmail.com"')
get_ipython().system(u'git config --global user.name "DSGroup"')


# In[90]:


get_ipython().system(u'git commit -m "adding cookiecutter project"')


# ## Place the raw data and the machine learning notebooks in the dedicated folders

# In[113]:


get_ipython().magic(u'cd data/raw')


# In[114]:



get_ipython().system(u'gdown https://drive.google.com/uc?id=1O-7VpweD_Ao8xGijJ7-9pd28MZLSoLqD')


# In[115]:


get_ipython().magic(u'cd ../../notebooks/')


# In[110]:


get_ipython().system(u'gdown https://drive.google.com/uc?id=1cbUK6zxWvDfXDPCUp3HAsv-Rp9ITzQB2')
#https://colab.research.google.com/drive/1cbUK6zxWvDfXDPCUp3HAsv-Rp9ITzQB2?usp=sharing


# In[ ]:





# --------------
# # End of This Notebook

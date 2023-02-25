#!/usr/bin/env python
# coding: utf-8

# ## Data Scrubbing , EDA, Naive Bayes, Logit, CART

# ### Name: Jeriel Wadjas, Yin Wang, Kieran Furse

# In[1]:


#import libraries
import pandas as pd #dataframe
import numpy as np #numerical
#visualization
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


cols=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','class']
adult=pd.read_csv('adult.csv',names=cols)
adult.head()


# In[3]:


# 


# In[4]:


# adult['fnlwgt']=adult['fnlwgt']/1000


# In[5]:


# adult['fnlwgt']


# In[6]:


# #Repeat rows based on column value
# # adult=adult.reindex(adult.index.repeat(adult['fnlwgt']))
# #adult=adult.loc[adult.index.repeat(adult['fnlwgt'])]
# adult.reset_index(drop=True)
# # adult


# In[7]:


# adult=adult.sample(frac=0.1)
# adult=adult.reset_index(drop=True)


# In[8]:


# adult=adult[~adult.duplicated()]


# In[9]:


# adult=adult.reset_index(drop=True)


# In[10]:


adult


# **Q1: How many observations (rows) and how many variables (columns) are there in the raw data?**

# In[14]:


adult.shape


# In[15]:


print(f'There is {adult.shape[0]} observations and {adult.shape[1]} variables in the raw data')


# In[16]:


#adult.drop(columns='fnlwgt',inplace=True)


# In[17]:


adult


# **Q2: Produce a table of variables showing their types**

# In[18]:


#adult.info()
pd.DataFrame(adult.dtypes,columns= ['Types'] )


# **Q3: Some of the variables appear to be numeric but should be treated as categorical. Your best clue is whether a variable has only a few discrete values. Which numeric variables should be treated as categorical?**

# In[19]:


numeric_col=adult.select_dtypes(include='int64').columns


# In[20]:


print('Number of discrete value\n')
for col in numeric_col:
    print(f'{col}: {len(adult[col].unique())}')
#print(adult['education-num'].unique())


# In[21]:


#print('age, capitalgain, capitalloss and hoursperweek have only 5 discrete values and should be treated as categorical')


# In[22]:


adult['education-num']=adult['education-num'].astype('category')


# In[23]:


# cat_col=[]
# for col in numeric_col:
#     if len(adult[col].unique())<10:
#         cat_col.append(col)


# In[24]:


# for col in cat_col:
#     adult[col]=adult[col].astype('category')


# In[25]:


adult.dtypes


# In[26]:


numeric_col=adult.select_dtypes(include='int64').columns
categorical_col=adult.select_dtypes(include=['category','object']).columns

# print(numeric_col)
# print(categorical_col)


# **Q4: For numeric variables, produce a table of statistics including missing values, min, max, median, mean, standard deviation, skewness and kurtosi**

# In[28]:


adult[numeric_col].agg(['min','max','median','mean','std','skew','kurt'])


# **Q5: How many outliers are present in each numeric variable? Show the tallies in a table. Set them to missing**

# In[29]:


outliers=[]
for col in numeric_col:
    Q1=np.nanpercentile(adult[col],25)
    Q3=np.nanpercentile(adult[col],75)
    #calculate the iqr
    iqr=Q3-Q1
    out=((adult[col]<(Q1-1.5*iqr)) | (adult[col]>(Q3 +1.5*iqr))).sum()
    outliers.append(out)
    


# In[30]:


outlier_dict=dict(zip(numeric_col,outliers))


# In[31]:


outlier_dict


# In[32]:


outliers_df=pd.DataFrame(data=outlier_dict,index=[0])
outliers_df


# In[33]:


numeric_col


# In[34]:


for col in numeric_col:
    #create lower and upper percentile
    Q1=np.nanpercentile(adult[col],25)
    Q3=np.nanpercentile(adult[col],75)
    #calculate the iqr
    iqr=Q3-Q1
    #replace the outlier
    adult[col]=np.where((adult[col]<(Q1-1.5*iqr)) | (adult[col]>(Q3 +1.5*iqr)),np.nan,adult[col])


# **Q6: Count the unique values of each categorical variable, including missing values. Are there any unusual values in any of the categorical variables?**

# In[36]:


for col in categorical_col:
    print(f'{col}:{len(adult[col].unique())} unique values')
    print(f'{col}:{adult[col].isna().sum()} missing values\n')


# In[38]:


#check if each categorical column has unusual values
for i in categorical_col:
    print(adult[i].value_counts())


# In[39]:


mask=adult[categorical_col].isna()
# adult[mask]


# **Q7: Impute the missing values. Be sure to explain how you did that in your presentation.**

# In[40]:


for col in numeric_col:
    adult[col].fillna(adult[col].mean(),inplace=True)


# In[41]:


numeric_col


# In[42]:


categorical_col


# In[43]:


adult['workclass'].value_counts().index[0]


# In[44]:


# replace unusual values (' ?')[unknown] with the mode in the column
for i in categorical_col:
    adult[i]= np.where(adult[i] == ' ?', adult[i].mode(), adult[i])


# In[45]:


#adult[categorical_col]=adult[categorical_col].apply(lambda x: x.fillna(x.value_counts().index[0]))


# In[47]:


#adult[numeric_col].isna().sum()
#adult[categorical_col].isna().sum()


# In[48]:


#Plot Parameters
sns.set_style('darkgrid')
plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.facecolor'] = '#00000000'


# **Q8: Produce a histogram or boxplot for each of the numeric variables.**

# In[49]:


#add a count column
adult['count']=1


# In[50]:


# plt.figure(figsize=(12,6))
# sns.barplot(x='day',y='total_bill',hue='sex',data=tips_df)
# plt.show()


# In[52]:


# plt.figure(figsize=(12,6))
# for col in numeric_col:
#     sns.histplot(data=adult,x=col, hue='class')
#     plt.show()


# In[53]:


#plot an histogram
for col in numeric_col:
    hist=px.histogram(adult,x=col,color='class',marginal='box')
    hist.update_layout(
    title_text=f'Distribution of {col}',
        bargap=0.1,
        bargroupgap=.2
    )
    hist.show()


# In[55]:


for col in numeric_col:
    bar=px.bar(x=[w for w, df in adult.groupby(col)],y=adult.groupby(col).count()['count'],color=adult.groupby(col).count()['class'])
    bar.update_layout(
    title_text=f'Distribution of {col}')
    bar.show()


# In[ ]:





# In[ ]:





# inferences

# In[56]:


#plot a boxplot
for col in numeric_col:
    box=px.box(adult[col])
    box.show()


# **Q9: Produce a bar chart for each of the categorical variables showing the counts for each unique value**

# In[57]:


len(categorical_col)


# In[58]:


plt.figure(figsize=(20,7))
for col in categorical_col:
    ax=sns.countplot(data=adult,x=col,hue='class')
    ax.set_xticklabels(ax.get_xticklabels(),rotation=60)
    plt.title(col)
    plt.show()


# In[59]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go


# In[60]:



# j=1
# d=1

# part1=categorical_col[0:5]
# part2=categorical_col[5:10]


# fig= make_subplots(rows=2, cols=5, start_cell ='top-left')


# for k in part2:
#     for i in part1:
#         if j<6:
#             fig.add_trace(go.Bar(x=adult[i],y=adult['count']),row=1,col=j)
#             j+=1
#     if d<6:
#         fig.add_trace(go.Bar(x=adult[k],y=adult['count']),row=2,col=d)
#         d+=1
# fig.update_layout(bargap=.2,width=1150,height=650)
# fig.show()


# In[61]:


adult.groupby('workclass').count()['class']


# In[62]:


for col in categorical_col:
    bar=px.bar(x=[w for w, df in adult.groupby(col)],y=adult.groupby(col).count()['count'],color=adult.groupby(col).count()['class'])
    bar.show()


# In[50]:


for col in categorical_col:
    plot=px.bar(adult[col].value_counts())
    plot.update_layout(title=col)
    plot.show()


# In[51]:


adult_corr=adult.corr()
plt.figure(figsize=(15,9))
sns.heatmap(adult_corr,annot=True,cmap='viridis');


# In[52]:


# scatter=px.scatter_matrix(adult,
#     dimensions=categorical_col)
# scatter.update_layout(height=1100,width=1100)
# scatter.show()


# In[53]:


categorical_col


# In[54]:


adult.describe()


# ### Machine Learning Workflow
# 
# Whether we're solving a regression problem using linear regression or a classification problem using logistic regression, the workflow for training a model is exactly the same:
# 
# 1. We initialize a model with random parameters (weights & biases).
# 2. We pass some inputs into the model to obtain predictions.
# 3. We compare the model's predictions with the actual targets using the loss function.  
# 4. We use an optimization technique (like least squares, gradient descent etc.) to reduce the loss by adjusting the weights & biases of the model
# 5. We repeat steps 1 to 4 till the predictions from the model are good enough.
# 
# 
# <img src="https://www.deepnetts.com/blog/wp-content/uploads/2019/02/SupervisedLearning.png" width="480">
# 
# 
# Classification and regression are both supervised machine learning problems, because they use labeled data. Machine learning applied to unlabeled data is known as unsupervised learning ([image source](https://au.mathworks.com/help/stats/machine-learning-in-matlab.html)). 
# 
# <img src="https://i.imgur.com/1EMQmAw.png" width="480">
# 
# 
# 

# ### Logistic Regression for Solving Classification Problems
# ​
# Logistic regression is a commonly used technique for solving binary classification problems. In a logistic regression model: 
# ​
# - we take linear combination (or weighted sum of the input features) 
# - we apply the sigmoid function to the result to obtain a number between 0 and 1
# - this number represents the probability of the input being classified as "Yes"
# - instead of RMSE, the cross entropy loss function is used to evaluate the results
# ​
# ​
# Here's a visual summary of how a logistic regression model is structured ([source](http://datahacker.rs/005-pytorch-logistic-regression-in-pytorch/)):
# ​
# ​
# <img src="https://i.imgur.com/YMaMo5D.png" width="480">
# ​
# The sigmoid function applied to the linear combination of inputs has the following formula:
# ​
# <img src="https://i.imgur.com/sAVwvZP.png" width="400">
# ​
# ​
# The output of the sigmoid function is called a logistic, hence the name _logistic regression_. For a mathematical discussion of logistic regression, sigmoid activation and cross entropy, check out [this YouTube playlist](https://www.youtube.com/watch?v=-la3q9d7AKQ&list=PLNeKWBMsAzboR8vvhnlanxCNr2V7ITuxy&index=1). Logistic regression can also be applied to multi-class classification problems, with a few modifications.

# ### Preprocessing

# In[55]:


#Select columns to encode
enc_col=adult.select_dtypes(include='object').columns[0:8]


# In[56]:


adult[enc_col]


# In[57]:


#Remove whitespace from categorical col
adult[enc_col]=adult[enc_col].apply(lambda x: x.str.strip())
adult['class']=adult['class'].apply(lambda x: x.strip())


# In[58]:


# #One hot encode the class variable
adult['class']=adult['class'].replace({'<=50K':0,'>50K':1})
# # adult['class'] = adult['class'].map({' <=50K':0,' >50K':1}) 


# In[60]:


#Create dummies variables
cat_enc= pd.get_dummies(adult[enc_col], drop_first=True)
#Concat dummies variable + numeric col
adult=pd.concat([adult[numeric_col],adult['education-num'],cat_enc,adult['class']],axis=1)
# adult


# In[55]:


# # #Label Encoding
# adult[enc_col]=pd.DataFrame({col: adult[col].astype('category').cat.codes for col in adult[enc_col]}, index=adult[enc_col].index)


# In[56]:


#adult[enc_col]


# In[57]:


#Create a mapping dictionary for categorical varibales
# cat_map={col: {n: cat for n, cat in enumerate(adult[col].astype('category').cat.categories)} 
#      for col in adult[enc_col]}


# In[58]:


# adult[enc_col]


# In[59]:


# from sklearn.preprocessing import OrdinalEncoder 
# oe=OrdinalEncoder()
# adult[enc_col]=oe.fit(adult[enc_col])
#enc=oe.transform(adult[enc_col])


# In[60]:


# adult[enc_col]


# In[61]:


adult


# In[62]:


#standardize the numeric values
from sklearn.preprocessing import MinMaxScaler 
MMS=MinMaxScaler() 
adult[numeric_col]=MMS.fit_transform(adult[numeric_col])


# In[63]:


#adult[numeric_col]
[numeric_col, categorical_col]


# In[64]:


#Create a X and y variables
# X=adult.iloc[:,:14] with label encoding
X=adult.iloc[:,:97] #with dummy variable
y=adult['class']
# X=adult.iloc[:,:14]


# In[65]:


from sklearn.model_selection import train_test_split


# In[66]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=3)


# In[67]:


X_train


# ### Q10: Naïve Bayes Model

# In[68]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,classification_report
clf = MultinomialNB()
clf.fit(X_train, y_train)


# In[69]:


train_preds=clf.predict(X_train)
accuracy_score(y_train, clf.predict(X_train))


# In[70]:


test_preds=clf.predict(X_test)
accuracy_score(y_test, clf.predict(X_test))


# In[71]:


clf.predict_proba(X_test)[:, 1]


# In[72]:


print(len(y_test))
print(len(clf.predict_proba(X_test)))


# In[73]:


import matplotlib.pyplot as plt


# In[74]:


#plotting ROC curve 
from sklearn.metrics import auc, roc_curve
fpr, tpr, threshold = roc_curve(y_test,clf.predict_proba(X_test)[:, 1], pos_label=1)
roc_auc= auc(fpr, tpr)

plt.figure(figsize= (6, 6))
plt.plot(fpr, tpr, color= 'red');
plt.plot([0,1],[0,1], linestyle= '--' );


# In[75]:


print(classification_report(y_test, test_preds))


# ### Q11: Logit Model

# **Q11.1 Build a model to predict income > $50K using logistic regression. Randomly partition the data into a training set (70%) and a validation set (30%)**

# In[76]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,classification_report
import statsmodels.api as sm


# In[77]:


#Train the logistic regression model
logistic_reg=LogisticRegression(solver='liblinear').fit(X_train,y_train)


# In[78]:


train_preds=logistic_reg.predict(X_train)


# In[79]:


accuracy_score(y_train,train_preds)


# In[80]:


X2 = sm.add_constant(X_train)
est = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())


# In[81]:


print('age,fnlwgt,hours-per-week,education-num,workclass_local-gov,workclass_private,workclass_Self-emp-not-inc,workclass_Self-emp-not-inc,workclass_State-gov,workclass_Without-pay,education_1st-4th')


# In[95]:


test_preds=logistic_reg.predict(X_test)
acc=accuracy_score(y_test,test_preds)
acc


# In[96]:


missclassification_rate= 1 - acc
missclassification_rate


# In[97]:


print(classification_report(y_test, test_preds))


# In[98]:


#y_test[y_test==1],y_test


# In[99]:


#Confusion Matrix
confusion_matrix(y_train,train_preds)


# In[100]:


#plotting ROC curve 
from sklearn.metrics import auc, roc_curve
fpr, tpr, threshold = roc_curve(y_test,logistic_reg.predict_proba(X_test)[:, 1], pos_label=1)
roc_auc= auc(fpr, tpr)

plt.figure(figsize= (3, 3))
plt.plot(fpr, tpr, color= 'red')
plt.plot([0,1],[0,1], linestyle= '--' )
plt.title('ROC')
plt.show()


# In[ ]:





# In[101]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[102]:


#Create confusion matrix and plot
def predict_and_plot(inputs,targets,name=''):
    preds= logistic_reg.predict(inputs)
    accuracy=accuracy_score(targets,preds)
    print(f"Accuracy: {accuracy*100}%")
    
    cf=confusion_matrix(targets,preds, normalize='true')
    plt.figure()
    sns.heatmap(cf, annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title(f'{name}')
    return preds


# In[103]:


train_preds= predict_and_plot(X_train,y_train,'Training')


# In[104]:


test_preds= predict_and_plot(X_test,y_test,'Test')


# **Q12: Tree Model (CART)**

# In[105]:


#Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier


# In[106]:


#Train the model
tree = DecisionTreeClassifier(random_state=2,max_depth=7).fit(X_train,y_train)


# In[107]:


#Make prediction
train_preds=tree.predict(X_train)


# In[108]:


#Evaluate the train_inputs
train_acc=accuracy_score(y_train,train_preds)
print('Train Accuracy:{:.2f}%'.format(train_acc*100))


# In[109]:


#Test the val_inputs
test_preds=tree.predict(X_test)
test_acc=accuracy_score(y_test,test_preds)
print('Test Accuracy:{:.2f}%'.format(test_acc*100))


# In[110]:


#Plot the tree
from sklearn.tree import plot_tree
plt.figure(figsize=(80,20))
plot_tree(tree,feature_names=X_train.columns,max_depth=2,filled=True);


# In[111]:


def predict_and_plot(inputs,targets,name=''):
    preds= tree.predict(inputs)
    accuracy=accuracy_score(targets,preds)
    print(f"Accuracy: {accuracy*100}%")
    
    cf=confusion_matrix(targets,preds, normalize='true')
    plt.figure()
    sns.heatmap(cf, annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title(f'{name}')
    return preds


# In[112]:


train_preds= predict_and_plot(X_train,y_train,'Training')


# In[113]:


test_preds= predict_and_plot(X_test,y_test,'Test')


# In[114]:


print(classification_report(y_test, test_preds))


# In[115]:


#plotting ROC curve 
fpr, tpr, threshold = roc_curve(y_test,tree.predict_proba(X_test)[:, 1], pos_label=1)
roc_auc= auc(fpr, tpr)

plt.figure(figsize= (3, 3))
plt.plot(fpr, tpr, color= 'red')
plt.plot([0,1],[0,1], linestyle= '--' )
plt.title('ROC')
plt.show()


# In[116]:


from sklearn.ensemble import RandomForestClassifier


# In[128]:


RF= RandomForestClassifier(n_jobs=-1,max_depth=19,n_estimators=500,random_state=42,).fit(X_train,y_train)


# In[131]:


#Make prediction
train_preds=RF.predict(X_train)


# In[132]:


#Evaluate the train_inputs
train_acc=accuracy_score(y_train,train_preds)
print('Train Accuracy:{:.2f}%'.format(train_acc*100))


# In[133]:


#Test the val_inputs
test_preds=RF.predict(X_test)
test_acc=accuracy_score(y_test,test_preds)
print('Test Accuracy:{:.2f}%'.format(test_acc*100))


# In[134]:


def predict_and_plot(inputs,targets,name=''):
    preds= RF.predict(inputs)
    accuracy=accuracy_score(targets,preds)
    print(f"Accuracy: {accuracy*100}%")
    
    cf=confusion_matrix(targets,preds, normalize='true')
    plt.figure()
    sns.heatmap(cf, annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title(f'{name}')
    return preds


# In[135]:


train_preds= predict_and_plot(X_train,y_train,'Training')


# In[123]:


test_preds= predict_and_plot(X_test,y_test,'Test')


# **Q13: Compare Models**

# In[124]:


model_comp={'models_name':['Naive Bayes',"Logistic Regression","Decision Tree"],
            'acc_train':[81.58,83.4,83.88],
            'acc_test':[81.38,82.87,82.05],
           }


# In[125]:


comp=pd.DataFrame(model_comp,index=None)


# In[126]:


comp


# In[127]:


print(f'logistic regression had the best accuracy for test set')


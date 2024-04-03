#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import datasets,metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
import pickle


# In[3]:


user=pd.read_csv("users.csv")
fake=pd.read_csv("fusers.csv")


# In[4]:


#adding a column for detecting fake or not
idNo_user=np.zeros(1481) #zero is adding for user 
idNo_fake=np.ones(1337)   #ones is adding for fake users


# In[5]:


user["isFake"]=idNo_user
fake["isFake"]=idNo_fake


# In[6]:


df_user=pd.concat([user,fake],ignore_index=True)
df_user.columns


# In[7]:


#shuffle the whole data
df_user=df_user.sample(frac=1).reset_index(drop=True)


# In[8]:


df_user.info()


# In[9]:


user_counts = df_user['isFake'].value_counts()

# Plot the bar graph
plt.bar(['Real', 'Fake'], user_counts.values, color=['green', 'red'])

# Add labels and title
plt.xlabel('User Type')
plt.ylabel('Number of Users')
plt.title('Distribution of Real and Fake Users')

# Add labels to the bars
for i, value in enumerate(user_counts.values):
    plt.text(i, value + 10, str(value), ha='center')

# Show the plot
plt.show()


# In[10]:


df_user=df_user[[ 
    
     'statuses_count',
     'followers_count',
     'friends_count',
     'listed_count',
     'favourites_count',
     'lang',
     'default_profile',
     'profile_use_background_image',
     'isFake'
     
     ]]


# In[11]:


df_user.info()


# In[12]:


null_counts = df_user.isnull().sum()

plt.figure(figsize=(10, 6))  

# Plot the bar graph
plt.bar(null_counts.index, null_counts.values, color='blue')

# Add labels and title
plt.xlabel('Columns')
plt.ylabel('Number of Null Values')
plt.title('Null Values in Dataset')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.show()


# In[13]:


df_user=df_user.fillna(0)


# In[14]:


null_counts = df_user.isnull().sum()

plt.figure(figsize=(10, 6))  

# Plot the bar graph
plt.bar(null_counts.index, null_counts.values, color='blue')

# Add labels and title
plt.xlabel('Columns')
plt.ylabel('Number of Null Values')
plt.title('Null Values in Dataset')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.show()


# In[15]:


def plotting(col, df=df_user):
    return df.groupby(col)['isFake'].value_counts().unstack().plot(kind='bar', figsize=(10,5), title=f'Distribution of isFake by {col}')

# Example usage:
plotting("statuses_count")


# In[16]:


plotting("followers_count")


# In[17]:


plotting("friends_count")


# In[18]:


plotting("listed_count")


# In[19]:


plotting("favourites_count")


# In[20]:


plotting("lang")


# In[21]:


plotting("default_profile")


# In[22]:


plotting("profile_use_background_image")


# In[23]:


plt.figure(figsize=(12, 3))

# Create the boxplot
sns.boxplot(data=df_user)

# Show the plot
plt.show()


# In[24]:


#data_df = pd.DataFrame(df_user)

Q1 = df_user.quantile(0.25)
Q3 = df_user.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[25]:


df_user = df_user[~((df_user < (Q1 - 1.5 * IQR)) |(df_user > (Q3 + 1.5 * IQR))).any(axis=1)]
df_user


# In[26]:


plt.figure(figsize=(12, 5))

# Create the boxplot
sns.boxplot(data=df_user)

# Show the plot
plt.show()


# In[27]:


numeric_columns = df_user.drop(columns=['isFake']).select_dtypes(include=[np.number])
plt.figure(figsize=(8, 5))

# Calculate correlation matrix
correlation_matrix = numeric_columns.corr()

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", square=True)

# Show plot
plt.show()


# In[28]:


for col in df_user.columns:
    le=LabelEncoder()
    df_user[col]=le.fit_transform(df_user[col])
df_user.head(20)


# In[29]:


y=df_user['isFake']
x=df_user.drop(['isFake'], axis=1)


# In[30]:


#dataset-2
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[31]:


x_train


# In[32]:


x_test


# In[33]:


y_train


# In[34]:


y_test


# In[35]:


sm = SMOTE(random_state=42, k_neighbors=5)
x, y = sm.fit_resample(x, y)
a=y.value_counts()
a


# In[36]:



#user_counts = df_user['isFake'].value_counts()

# Plot the bar graph
plt.bar(['Real', 'Fake'], a.values, color=['green', 'red'])

# Add labels and title
plt.xlabel('User Type')
plt.ylabel('Number of Users')
plt.title('Distribution of Real and Fake Users')

# Add labels to the bars
for i, value in enumerate(a.values):
    plt.text(i, value + 10, str(value), ha='center')

# Show the plot
plt.show()


# In[37]:


from sklearn.metrics import accuracy_score
from sklearn import metrics


# In[38]:


from sklearn.impute import SimpleImputer

# Replace NaN values with the mean
imputer = SimpleImputer(strategy='mean')
x_train_imputed = imputer.fit_transform(x_train)


# In[39]:


from sklearn.impute import SimpleImputer

# Create the imputer with strategy='mean' or 'median'
imputer = SimpleImputer(strategy='mean')

# Fit and transform the imputer on the training data
x_train_imputed = pd.DataFrame(imputer.fit_transform(x_train), columns=x_train.columns)

# Transform the test data using the imputer fitted on the training data
x_test_imputed = pd.DataFrame(imputer.transform(x_test), columns=x_test.columns)


# In[40]:


Training_Accuracy_L=[]
Test_Accuracy_L=[]
Sensitivity_L=[]
Specificity_L=[]
F1Score_L=[]
Precision_L=[]
Negative_Predictive_Value_L=[]
False_Negative_Rate_L=[]
False_Positive_Rate_L=[]
False_Discovery_Rate_L=[]
False_Omission_Rate_L=[]
average_cv_accuracy_L=[]


# In[70]:


Training_Accuracy_L=[]
Test_Accuracy_L=[]
Sensitivity_L=[]
Specificity_L=[]
F1Score_L=[]
Precision_L=[]
Negative_Predictive_Value_L=[]
False_Negative_Rate_L=[]
False_Positive_Rate_L=[]
False_Discovery_Rate_L=[]
False_Omission_Rate_L=[]
average_cv_accuracy_L=[]

import math


def rounder(n):
  try:
    return math.ceil(n * 1000) / 1000
  except:
    return n

def fun(model,name):
  test_pred = model.predict(x_test)
  train_pred = model.predict(x_train)
  
  train_acc=rounder(accuracy_score(y_train,train_pred)*100)
  test_acc=rounder(accuracy_score(y_test,test_pred)*100)

  Training_Accuracy_L.append(train_acc)
  Test_Accuracy_L.append(test_acc)

  print("\nTraining Accuracy:", train_acc)
  print("\nTesting Accuracy:",test_acc)

  print(classification_report(y_test,test_pred))
  test_conf_matrix = confusion_matrix(y_test,test_pred)
  plt.figure(figsize=(4, 4))
  sns.heatmap(test_conf_matrix, annot=True, fmt='g', cmap='Greens', cbar=False)
  t=name+' Confusion Matrix - Test Set'
  plt.title(t)
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.show()

  tn, fp,fn,tp = test_conf_matrix.ravel()

  Sensitivity=rounder((tp) / (tp + fn))
  Sensitivity_L.append(Sensitivity)

  Specificity=rounder((tn) / (tn + fp))
  Specificity_L.append(Specificity)

  F1Score=rounder( (2 * tp) / (2 * tp+ fp + fn))
  F1Score_L.append(F1Score)

  Precision=rounder((tp) / (tp +fp))
  Precision_L.append(Precision)

  Negative_Predictive_Value= rounder((tn) / (tn + fn))
  Negative_Predictive_Value_L.append(Negative_Predictive_Value)

  False_Negative_Rate=rounder((fn) / (fn + tp))
  False_Negative_Rate_L.append(False_Negative_Rate)

  False_Positive_Rate=rounder((fp) / (fp + tn))
  False_Positive_Rate_L.append(False_Positive_Rate)

  False_Discovery_Rate=rounder((fp) / (fp + tp))
  False_Discovery_Rate_L.append(False_Discovery_Rate)

  False_Omission_Rate=rounder((fn) / (fn+ tn))
  False_Omission_Rate_L.append(False_Omission_Rate)


  print('Sensitivity:', Sensitivity)
  print('Specificity:', Specificity)
  print('F1 Score:', F1Score)
  print('Precision:',Precision)
  print('Negative Predictive Value:', Negative_Predictive_Value)
  print('False Negative Rate:',False_Negative_Rate)
  print('False Positive Rate:',False_Positive_Rate)
  print('False Discovery Rate:',False_Discovery_Rate)
  print('False Omission Rate:', False_Omission_Rate)

  num_folds = 5
  kf = KFold(n_splits=num_folds, shuffle=True,random_state=42)
  cv_scores = cross_val_score(model, x_train, y_train, cv=kf, scoring='accuracy')
  print(f"\n{num_folds}-Fold Cross-Validation Scores:")
  print(cv_scores)
  average_cv_accuracy = rounder(np.mean(cv_scores))
  print(f"\nAverage Cross-Validation Accuracy: {average_cv_accuracy * 100:.2f}%")
  average_cv_accuracy_L.append(average_cv_accuracy)


# In[71]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score, KFold


# Assuming x_train, y_train, x_test, y_test are defined

# Use SimpleImputer to fill in missing values
imputer = SimpleImputer(strategy='mean')  # You can use other strategies like 'median' or 'most_frequent'
x_train_imputed = imputer.fit_transform(x_train)
x_test_imputed = imputer.transform(x_test)

# Create a Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust the number of trees (n_estimators) as needed

# Fit the model
rf.fit(x_train, y_train)

# Predict on the test set
y_pred_rf = rf.predict(x_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(" Random Forest Accuracy:", accuracy_rf)

cv_accuracy_rf = cross_val_score(rf, x_train, y_train, cv=5)
print(f"Cross-validated Random Forest Accuracy: {cv_accuracy_rf.mean()*100}")
try:
    y_prob_rf = rf.predict_proba(x_test_imputed)[:, 1]


# Compute ROC curve and ROC area for each class
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)

# Plot ROC curve
    plt.figure()
    plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label='ROC curve - rf (area = %0.2f)' % roc_auc_rf)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Random Forest')
    plt.legend(loc="lower right")
    plt.show()
except AttributeError as e:
    print("AttributeError:", e)


# In[72]:


fun(rf,'Random Forest Classifier')


# In[44]:


from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Train the model
rf.fit(x_train, y_train.values.ravel())

# Get predictions
y_pred_train = rf.predict(x_train)
y_pred_test = rf.predict(x_test)

# Calculate metrics for training data
report_train = classification_report(y_train, y_pred_train, output_dict=True)
accuracy_train = accuracy_score(y_train, y_pred_train)

# Calculate metrics for test data
report_test = classification_report(y_test, y_pred_test, output_dict=True)
accuracy_test = accuracy_score(y_test, y_pred_test)

# Extracting metrics for training data
precision_train = report_train['macro avg']['precision']
recall_train = report_train['macro avg']['recall']
f1_score_train = report_train['macro avg']['f1-score']

# Extracting metrics for test data
precision_test = report_test['macro avg']['precision']
recall_test = report_test['macro avg']['recall']
f1_score_test = report_test['macro avg']['f1-score']

# Plotting
labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
train_scores = [accuracy_train, precision_train, recall_train, f1_score_train]
test_scores = [accuracy_test, precision_test, recall_test, f1_score_test]

x = range(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x, train_scores, width, label='Train')
rects2 = ax.bar([p + width for p in x], test_scores, width, label='Test')

ax.set_ylabel('Scores')
ax.set_title('Random Forest Classifier')
ax.set_xticks([p + width/2 for p in x])
ax.set_xticklabels(labels)
ax.legend()

plt.show()


# In[45]:


from sklearn.linear_model import LogisticRegression

# Assuming x_train, y_train, x_test, y_test are defined

# Create a logistic regression model
lr = LogisticRegression()

# Use SimpleImputer to fill in missing values
imputer = SimpleImputer(strategy='mean')  # You can use other strategies like 'median' or 'most_frequent'
x_train = imputer.fit_transform(x_train)
x_test= imputer.transform(x_test)

# Fit the model
lr.fit(x_train, y_train)

# Predict on the test set
y_pred_log_reg = lr.predict(x_test)

# Evaluate the model
accuracy_lr = accuracy_score(y_test, y_pred_log_reg)
print("Logistic Regression Accuracy:", accuracy_lr)

cv_log_reg = cross_val_score(lr, x_train_imputed, y_train, cv=5)
print(f"Cross-validated Logistic Regression Accuracy: {cv_log_reg.mean()*100}")
try:
    y_prob_log_reg = lr.predict_proba(x_test_imputed)[:, 1]

# Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_test, y_prob_log_reg)
    roc_auc = auc(fpr, tpr)

# Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve - lr (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
except AttributeError as e:
    print("AttributeError:", e)


# In[46]:


fun(lr,'Logistic Regression')


# In[47]:


from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Train the model
lr.fit(x_train, y_train.values.ravel())

# Get predictions
y_pred_train = lr.predict(x_train)
y_pred_test = lr.predict(x_test)

# Calculate metrics for training data
report_train = classification_report(y_train, y_pred_train, output_dict=True)
accuracy_train = accuracy_score(y_train, y_pred_train)

# Calculate metrics for test data
report_test = classification_report(y_test, y_pred_test, output_dict=True)
accuracy_test = accuracy_score(y_test, y_pred_test)

# Extracting metrics for training data
precision_train = report_train['macro avg']['precision']
recall_train = report_train['macro avg']['recall']
f1_score_train = report_train['macro avg']['f1-score']

# Extracting metrics for test data
precision_test = report_test['macro avg']['precision']
recall_test = report_test['macro avg']['recall']
f1_score_test = report_test['macro avg']['f1-score']

# Plotting
labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
train_scores = [accuracy_train, precision_train, recall_train, f1_score_train]
test_scores = [accuracy_test, precision_test, recall_test, f1_score_test]

x = range(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x, train_scores, width, label='Train')
rects2 = ax.bar([p + width for p in x], test_scores, width, label='Test')

ax.set_ylabel('Scores')
ax.set_title('Logistic Regression')
ax.set_xticks([p + width/2 for p in x])
ax.set_xticklabels(labels)
ax.legend()

plt.show()


# In[48]:


from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

# Assuming x_train, y_train, x_test, y_test are defined

# Create a Gaussian Naive Bayes model
nb = GaussianNB()

# Use SimpleImputer to fill in missing values
imputer = SimpleImputer(strategy='mean')  # You can use other strategies like 'median' or 'most_frequent'
x_train = imputer.fit_transform(x_train)
x_test= imputer.transform(x_test)

# Fit the model
nb.fit(x_train, y_train)

# Predict on the test set
y_pred_nb = nb.predict(x_test)

# Evaluate the model
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Navie Bayes Accuracy:", accuracy_nb)

cv_accuracy = cross_val_score(nb, x_train, y_train, cv=5)
print(f"Cross-validated naive bayes Accuracy: {cv_accuracy.mean()*100}")
try:
    y_prob_nb = nb.predict_proba(x_test)[:, 1]

# Compute ROC curve and ROC area for each class
    fpr_nb, tpr_nb, _ = roc_curve(y_test, y_prob_nb)
    roc_auc_nb = auc(fpr_nb, tpr_nb)

# Plot ROC curve
    plt.figure()
    plt.plot(fpr_nb, tpr_nb, color='darkorange', lw=2, label='ROC curve - nb (area = %0.2f)' % roc_auc_nb)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
except AttributeError as e:
    print("AttributeError:", e)


# In[49]:


fun(nb,'Navie Bayes')


# In[50]:


from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Train the model
nb.fit(x_train, y_train.values.ravel())

# Get predictions
y_pred_train = nb.predict(x_train)
y_pred_test = nb.predict(x_test)

# Calculate metrics for training data
report_train = classification_report(y_train, y_pred_train, output_dict=True)
accuracy_train = accuracy_score(y_train, y_pred_train)

# Calculate metrics for test data
report_test = classification_report(y_test, y_pred_test, output_dict=True)
accuracy_test = accuracy_score(y_test, y_pred_test)

# Extracting metrics for training data
precision_train = report_train['macro avg']['precision']
recall_train = report_train['macro avg']['recall']
f1_score_train = report_train['macro avg']['f1-score']

# Extracting metrics for test data
precision_test = report_test['macro avg']['precision']
recall_test = report_test['macro avg']['recall']
f1_score_test = report_test['macro avg']['f1-score']

# Plotting
labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
train_scores = [accuracy_train, precision_train, recall_train, f1_score_train]
test_scores = [accuracy_test, precision_test, recall_test, f1_score_test]

x = range(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x, train_scores, width, label='Train')
rects2 = ax.bar([p + width for p in x], test_scores, width, label='Test')

ax.set_ylabel('Scores')
ax.set_title('Navie Bayes')
ax.set_xticks([p + width/2 for p in x])
ax.set_xticklabels(labels)
ax.legend()

plt.show()


# In[51]:


from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

# Assuming x_train, y_train, x_test, y_test are defined
gbc= GradientBoostingClassifier(n_estimators = 15, max_features = None, min_samples_split = 2)


# Use SimpleImputer to fill in missing values
imputer = SimpleImputer(strategy='mean')  # You can use other strategies like 'median' or 'most_frequent'
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)

# Fit the model
gbc.fit(x_train, y_train)

# Predict on the test set
y_pred_gbc = gbc.predict(x_test)

# Evaluate the model
accuracy_gbc = accuracy_score(y_test, y_pred_gbc)
print("Gradient Boosting Accuracy:", accuracy_gbc)

cv_accuracy_gbc = cross_val_score(gbc, x_train, y_train, cv=5)
print(f"Cross-validated Gradient Bossting Classifier Accuracy: {cv_accuracy_gbc.mean()*100}")
try:
    y_prob_gbc = gbc.predict_proba(x_test)[:, 1]

# Compute ROC curve and ROC area for each class
    fpr_gbc, tpr_gbc, _ = roc_curve(y_test, y_prob_gbc)
    roc_auc_gbc = auc(fpr_gbc, tpr_gbc)

# Plot ROC curve
    plt.figure()
    plt.plot(fpr_nb, tpr_nb, color='darkorange', lw=2, label='ROC curve - gbc (area = %0.2f)' % roc_auc_gbc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
except AttributeError as e:
    print("AttributeError:", e)


# In[52]:


fun(gbc,'Gradient Boosting Classifier')


# In[53]:


from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Train the model
gbc.fit(x_train, y_train.values.ravel())

# Get predictions
y_pred_train = gbc.predict(x_train)
y_pred_test = gbc.predict(x_test)

# Calculate metrics for training data
report_train = classification_report(y_train, y_pred_train, output_dict=True)
accuracy_train = accuracy_score(y_train, y_pred_train)

# Calculate metrics for test data
report_test = classification_report(y_test, y_pred_test, output_dict=True)
accuracy_test = accuracy_score(y_test, y_pred_test)

# Extracting metrics for training data
precision_train = report_train['macro avg']['precision']
recall_train = report_train['macro avg']['recall']
f1_score_train = report_train['macro avg']['f1-score']

# Extracting metrics for test data
precision_test = report_test['macro avg']['precision']
recall_test = report_test['macro avg']['recall']
f1_score_test = report_test['macro avg']['f1-score']

# Plotting
labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
train_scores = [accuracy_train, precision_train, recall_train, f1_score_train]
test_scores = [accuracy_test, precision_test, recall_test, f1_score_test]

x = range(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x, train_scores, width, label='Train')
rects2 = ax.bar([p + width for p in x], test_scores, width, label='Test')

ax.set_ylabel('Scores')
ax.set_title('Gradient Boosting Classifier')
ax.set_xticks([p + width/2 for p in x])
ax.set_xticklabels(labels)
ax.legend()

plt.show()


# In[54]:


compare = pd.DataFrame({'Model': ['RANDOM FOREST','GRADIENT BOOSTING CLASSIFIER','LOGISTIC REGRESSION','NAIVE BAYES'],
                        'Accuracy': [accuracy_rf*100,accuracy_gbc*100, accuracy_lr*100, accuracy_nb*100],
                        'Training Accuracy':Training_Accuracy_L,
                        'Test Accuracy':Test_Accuracy_L,
                        'Sensitivity':Sensitivity_L,
                        'Specificity':Specificity_L,
                        'F1 Score':F1Score_L,
                        'Precision':Precision_L,
                        'Negative Predictive Value':Negative_Predictive_Value_L,
                        'False Negative Rate':False_Negative_Rate_L,
                        'False Positive Rate':False_Positive_Rate_L,
                        'False Discovery Rate':False_Discovery_Rate_L,
                        'False Omission Rate':False_Omission_Rate_L,
                        'Average cv-accuracy':average_cv_accuracy_L })
compare.sort_values(by='Accuracy', ascending=False)


# In[55]:


colors = ['skyblue', 'orange', 'green', 'red', 'purple', 'pink']
plt.figure(figsize=(10, 6))
plt.barh(compare['Model'], compare['Accuracy'], color=colors)
plt.xlabel('Accuracy (%)')
plt.title('Model Comparison - Accuracy')
plt.xlim(0, 100)
for index, value in enumerate(compare['Accuracy']):
    plt.text(value, index, f'{value:.2f}', va='center', fontsize=10)

plt.show()


# In[73]:


import pickle
with open('model.pkl','wb') as f:
    pickle.dump(rf,f)


# In[74]:


model=pickle.load(open('model.pkl','rb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





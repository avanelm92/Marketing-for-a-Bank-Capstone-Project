#!/usr/bin/env python
# coding: utf-8

# # Modulo 5 Tarea 3-Credit One Final
# ## Ana Vanessa López Monge

# # Libraries Import

# In[32]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns 
from matplotlib import pylab
import scipy
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
pylab.rcParams['figure.figsize'] = (10.0, 8.0)


# # Define estimators

# In[33]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.feature_selection import RFE


# # Set Model Metrics and Cross Validation 

# In[34]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split


# # Upload Dataset

# In[118]:


credit = pd.read_csv('CreditCard.csv', header =1)


# In[119]:


creditOriginal=pd.read_csv('CreditCard.csv', header =1)


# # General Visualizations

# In[120]:


credit.head() #Para visualiaza el numero de atributos


# In[121]:


credit.describe() #Para visualizar estadistica descriptiva de la data


# In[122]:


credit.info() #Tipo de variables


# In[123]:


credit.isna() #Para validar la calidad de la data y ver si hay celdas vacias


# In[124]:


credit.isna().sum() #Resumen de los NA por columna


# # Data Preparation

# In[125]:


#Renombrando variable dependiente
credit.rename(columns={'default payment next month':'DefaultPaymentNextMonth'}, inplace = True)
#Renombrando variables independientes
credit.rename(columns={'PAY_0':'PaySet05'}, inplace = True)
credit.rename(columns={'PAY_2':'PayAgo05'}, inplace = True)
credit.rename(columns={'PAY_3':'PayJul05'}, inplace = True)
credit.rename(columns={'PAY_4':'PayJun05'}, inplace = True)
credit.rename(columns={'PAY_5':'PayMay05'}, inplace = True)
credit.rename(columns={'PAY_6':'PayApr05'}, inplace = True)


# In[126]:


credit.columns


# In[127]:


#Cambiando los niveles de las variables independientes estudios, genero y status
credit['STUDY'] = credit.EDUCATION.map({0:'Others', 1:'School', 2:'University', 3:'High Sch.', 4:'Others', 
                                        5:'Others', 6:'Others'})
credit['GENDER'] = credit.SEX.map({1:'Male', 2:'Female'})
credit['MARITAL_ST'] = credit.MARRIAGE.map({0:'Others', 1:'Married', 2:'Single', 3:'Divorce'})


# In[128]:


#Realizando intervalos para la variables edad y limite de credito
#Edad
binsAge = [20,30,40,50,60,70,80]
groupNamesAge = ['20-29','30-39','40-49','50-59','60-69','>70']
credit['AGE_RANGE'] = pd.cut(credit['AGE'], binsAge, labels = groupNamesAge)


# In[129]:


#limite de  credito
binsCredit=[10000, 50000, 100000, 150000, 200000, 250000, 300000, 400000, 500000, 1000000]
groupNamesCredit = ['10K-50K', '51K-100K', '101K-150K', '151K-200K', '201K-250K', 
                '251K-300K', '301K-400K', '401K-500K', '>501K']
credit['LOAN_RANGE'] = pd.cut(credit['LIMIT_BAL'], binsCredit, labels = groupNamesCredit)


# In[130]:


#Cambiando el tipo de atributo
credit['DefaultPaymentNextMonth'] = pd.Categorical(credit.DefaultPaymentNextMonth)
credit['GENDER'] = pd.Categorical(credit.GENDER)
credit['STUDY'] = pd.Categorical(credit.STUDY)
credit['MARITAL_ST'] = pd.Categorical(credit.MARITAL_ST)
credit['PaySet05'] = pd.Categorical(credit.PaySet05)
credit['PayAgo05'] = pd.Categorical(credit.PayAgo05)
credit['PayJul05'] = pd.Categorical(credit.PayJul05)
credit['PayJun05'] = pd.Categorical(credit.PayJun05)
credit['PayMay05'] = pd.Categorical(credit.PayMay05)
credit['PayApr05'] = pd.Categorical(credit.PayApr05)


# In[131]:


credit['DefaulPaymentNextMonth'] = credit.DefaultPaymentNextMonth.map({0:'No', 1:'Yes'})


# In[132]:


credit.info()


# In[133]:


credit.describe()


# # Initial Analysis and Graph Visualizations

# In[134]:


# Realizar un analisis de correlacion
correlationCredit=creditOriginal.corr()
print(correlationCredit)


# In[135]:


#Realizar un análisis de covarianza
covarianceCredit = creditOriginal.cov()
print(covarianceCredit)


# In[136]:


#Histograma para limite de prestamo
plt.hist(credit['LIMIT_BAL'])
plt.xlabel('Amount Loans')
plt.ylabel('Frequency')
plt.title('Loans Amount Frecuency')
plt.show()


# In[137]:


# Grafico de barras para rango de limite de prestamo
gbarLoanRange = sns.factorplot('LOAN_RANGE', data=credit, kind='count', aspect=2)
gbarLoanRange.fig.suptitle('Loan Range Frecuency')
gbarLoanRange.set_xlabels('Loan Range')
gbarLoanRange.set_ylabels('Frecuency')


# In[138]:


# Agrupar y contabilizar por rango de prestamo
credit.groupby('LOAN_RANGE')['LOAN_RANGE'].count()


# In[139]:


# Grafico de barras para incumplimiento de pago para el proximo periodo
gbarLoanRange = sns.factorplot('DefaultPaymentNextMonth', data=credit, kind='count', aspect=2)
gbarLoanRange.fig.suptitle('Default Payment Next Month')
gbarLoanRange.set_xlabels('Default Payment')
gbarLoanRange.set_ylabels('Frecuency')


# In[140]:


# Agrupar y contabilizar por incumplimiento de pago
credit.groupby('DefaultPaymentNextMonth')['DefaultPaymentNextMonth'].count()


# In[141]:


# Grafico de proporciones para incumplimiento de pago
labels = 'Yes', 'No'
sizes = [6636,23364]
fig1, DefaultPaymentProp = plt.subplots()
DefaultPaymentProp.pie(sizes,labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
DefaultPaymentProp.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[142]:


# Grafico de barras para nivel academico
gbarStudy= sns.factorplot('STUDY', data=credit, kind='count', aspect=2,order=['University','High Sch.','School','Others'])
gbarStudy.set_xlabels('Education')
gbarStudy.set_ylabels('Frecuency')
gbarStudy.fig.suptitle('Education Level')


# In[143]:


# Agrupar y contabilizar por nivel academico
credit.groupby('STUDY')['STUDY'].count()


# In[144]:


# Calcular proporcion de incumplimiento de pago por nivel academico
propNivelAcademico=credit.groupby("STUDY")['DefaultPaymentNextMonth'].value_counts(normalize=True).unstack()
propNivelAcademico


# In[145]:


# Grafico proporcion de incumplimiento de pago por nivel academico
propStudies = credit.groupby("STUDY")['DefaultPaymentNextMonth'].value_counts(normalize=True).unstack()
propStudies.plot(kind='bar', stacked='True')


# In[146]:


# Grafico por genero
gbarSex = sns.factorplot('GENDER', data=credit, kind='count', aspect=2)
gbarSex.set_xlabels('Gender')
gbarSex.set_ylabels('Frecuency')
gbarSex.fig.suptitle('Clients per Gender')


# In[147]:


# Calcular proporcion de incumplimiento de pago por genero
propSex=credit.groupby("GENDER")['DefaultPaymentNextMonth'].value_counts(normalize=True).unstack()
propSex


# In[148]:


propSex.plot(kind='bar', stacked='True')


# In[149]:


# Grafico de barras para status matrimonial
gbarSM = sns.factorplot('MARITAL_ST', data=credit, kind='count', aspect=2, order=['Married','Single','Divorce','Others'])
gbarSM.set_xlabels('Marital Status')
gbarSM.set_ylabels('Frecuency')
gbarSM.fig.suptitle('Clients per Marital Status')


# In[150]:


# Calcular proporcion de incumplimiento de pago por estatus civil
propSM=credit.groupby("MARITAL_ST")['DefaultPaymentNextMonth'].value_counts(normalize=True).unstack()
propSM


# In[151]:


propSM.plot(kind='bar', stacked='True')


# In[152]:


# Grafico de barras para rango de edad
gbarAgeRange = sns.factorplot('AGE_RANGE', data=credit, kind='count', aspect=2)
gbarAgeRange.fig.suptitle('Age Range Frecuency')
gbarAgeRange.set_xlabels('Age Range')
gbarAgeRange.set_ylabels('Frecuency')


# In[153]:


# Agrupar por rango de edad
credit.groupby('AGE_RANGE')['AGE_RANGE'].count()


# In[154]:


# Calcular proporcion de incumplimiento de pago por rango de edad
propAge=credit.groupby("AGE_RANGE")['DefaultPaymentNextMonth'].value_counts(normalize=True).unstack()
propAge


# In[155]:


propAge.plot(kind='bar', stacked='True')


# In[156]:


#Visulizar combinaciones de variables Edad y Genero
sns.factorplot('AGE_RANGE', data=credit, kind='count', palette='Pastel2', hue='GENDER', col='DefaultPaymentNextMonth', 
               aspect=2, size=5)


# In[157]:


#Visualizar combinaciones de variables Estudio y Estado Civil
sns.factorplot('STUDY', data=credit, kind='count', palette='Pastel2', hue='MARITAL_ST', col='DefaultPaymentNextMonth', hue_order=['Single','Married','Divorce','Others'], 
               order=['University','School','High Sch.','Others'],aspect=2, size=5)


# # Select the features

# In[188]:


#Visualizar columnas
credit.columns


# In[278]:


#Selecciona todas las variables disponibles
AllFeatures=credit.loc[ : , ['LIMIT_BAL','AGE','SEX', 'EDUCATION', 'MARRIAGE', 'PaySet05',
       'PayAgo05', 'PayJul05', 'PayJun05', 'PayMay05', 'PayApr05', 'BILL_AMT1',
       'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
       'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]
AllFeatures.head()


# # Define dependent variable

# In[207]:


depVar = credit['DefaulPaymentNextMonth']
depVar


# # Experiment 1

# Se toman unicamente aquellas variables/atributos conocidos que se pueden tener en el momento que una persona solicita un credito. Edad, genero, estatus matrimonial y limite de credito son variables que se pueden tener desde un inicio y permitiran predecir si el cliente pagará su prestamo antes de ser autorizado. Para este experimento, se evalua el desempeño del modelo mediante la aplicación de diferentes algoritmos tales como Decision Tree, Super Vector Machine, Knn y Random Forest.

# ## Features Selection-Experiment 1

# In[273]:


#Selecciona limite de credito, sexo, educacion, estado civil y edad
FilteredFeatures=credit.loc[ : , ['LIMIT_BAL','AGE','SEX', 'MARRIAGE', 'EDUCATION']]
FilteredFeatures.head()


# ## Training and testing datasets-Experiment 1

# In[209]:


#Variables independientes Training Set
XTrain = (FilteredFeatures[:])
XTrain.head()


# In[210]:


#Variable Dependiente Training Set
YTrain = depVar[:]
YTrainCount = len(YTrain.index)
print('The number of observations in the Y training set are:',str(YTrainCount))
YTrain.head()


# In[211]:


#Variables Independientes Testing Set
XTest = FilteredFeatures[-100:]
XTestCount = len(XTest.index)
print('The number of observations in the feature testing set is:',str(XTestCount))
print(XTest.head())


# In[212]:


# Variable Dependiente Testing Set.Y Truth
YTest = depVar[-100:]
YTestCount = len(YTest.index)
print('The number of observations in the Y training set are:',str(YTestCount))
YTest.head()


# In[300]:


#Cross Validation
XTrain, XTest, YTrain, YTest = train_test_split(XTrain, YTrain, test_size=0.3)


# In[214]:


XTrain.shape, XTest.shape


# ## Predictive Models -Experiment 1

# ## Decision Tree

# In[ ]:


# Decision Tree
modelDecisionTree = DecisionTreeClassifier(criterion='gini', max_depth=10)


# In[218]:


#Training
modelDecisionTree.fit(XTrain,YTrain)


# In[219]:


#Cross Validation
print(cross_val_score(modelDecisionTree, XTrain, YTrain, cv=10, n_jobs=2)) 
modelDecisionTree.score(XTrain,YTrain)


# In[224]:


#Testing
predictionDecisionTree= modelDecisionTree.predict(XTest)


# In[225]:


#Confusion Matrix
confusion_matrix(YTest,predictionDecisionTree, labels=['No', 'Yes'])
pd.crosstab(YTest,predictionDecisionTree, rownames=['True'], colnames=['Prediction'], margins=True)


# In[230]:


#Classification Report
target_names = ['No', 'Yes']
print(classification_report(YTest, predictionDecisionTree,target_names=target_names))


# ## Super Vector Machine

# In[233]:


#SVM
modelSVM=svm.SVC(C=1, kernel='rbf')


# In[234]:


#Training
modelSVM.fit(XTrain,YTrain)


# In[247]:


#Cross Validation
print(cross_val_score(modelSVM, XTrain, YTrain, cv=10, n_jobs=2)) 
modelSVM.score(XTrain,YTrain)


# In[248]:


#Testing
predictionSVM= modelSVM.predict(XTest)


# In[249]:


#Confusion Matrix
confusion_matrix(YTest,predictionSVM, labels=['No', 'Yes'])
pd.crosstab(YTest,predictionSVM, rownames=['True'], colnames=['Prediction'], margins=True)


# In[250]:


#Classification Report
target_names = ['No', 'Yes']
print(classification_report(YTest, predictionSVM,target_names=target_names))


# # KNN

# In[253]:


#KNN
modelKnn = KNeighborsClassifier(n_jobs=2, n_neighbors=13, weights='uniform', p=1)


# In[254]:


#Training
modelKnn.fit(XTrain,YTrain)


# In[255]:


#Cross Validation
print(cross_val_score(modelKnn, XTrain, YTrain, cv=10, n_jobs=2)) 
modelKnn.score(XTrain,YTrain)


# In[256]:


#Testing
predictionKnn= modelKnn.predict(XTest)


# In[257]:


#Confusion Matrix
confusion_matrix(YTest,predictionKnn, labels=['No', 'Yes'])
pd.crosstab(YTest,predictionKnn, rownames=['True'], colnames=['Prediction'], margins=True)


# In[258]:


#Classification Report
target_names = ['No', 'Yes']
print(classification_report(YTest, predictionKnn,target_names=target_names))


# # Random Forest

# In[259]:


#RF
modelRF= RandomForestClassifier(max_depth=7, n_estimators=70, n_jobs=2, criterion='gini')


# In[260]:


#Training
modelRF.fit(XTrain,YTrain)


# In[261]:


#Cross Validation
print(cross_val_score(modelRF, XTrain, YTrain, cv=10, n_jobs=2)) 
modelRF.score(XTrain,YTrain)


# In[263]:


#Testing
predictionRf= modelRF.predict(XTest)


# In[265]:


#Confusion Matrix
confusion_matrix(YTest,predictionRf, labels=['No', 'Yes'])
pd.crosstab(YTest,predictionRf, rownames=['True'], colnames=['Prediction'], margins=True)


# In[266]:


#Classification Report
target_names = ['No', 'Yes']
print(classification_report(YTest, predictionRf,target_names=target_names))


# # Experiment 2

# Se toman unicamente aquellas variables/atributos que sugiere la función RFE (Recursive Feature Elimination). Para este experimento, se evalua el desempeño del modelo mediante la aplicación de diferentes algoritmos tales como Decision Tree, Super Vector Machine, Knn y Random Forest.

# ## Features Selection RFE-Experiment 2

# In[291]:


rfeRF=RandomForestClassifier()
rfe = RFE(rfeRF, n_features_to_select=10)
rfe.fit(AllFeatures,depVar)
print(rfe.support_)
print(rfe.ranking_)


# Las 10 variables independientes con un efecto más significativo sobre la variable a predecir segun RFE son: 'LIMIT_BAL','AGE','PaySet05', 'BILL_AMT1','BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6','PAY_AMT1'.

# In[290]:


# Variables con RFE
featuresRFE = credit.loc[ : , ['LIMIT_BAL', 'AGE', 'PaySet05','BILL_AMT1','BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6','PAY_AMT1']]
featuresRFE.head()


# ## Training and testing datasets-Experiment 2

# In[292]:


#Variables independientes Training Set
XTrain1 = (featuresRFE[:])
XTrain1.head()


# In[294]:


#Variable Dependiente Training Set
YTrain1 = depVar[:]
YTrain1Count = len(YTrain.index)
print('The number of observations in the Y training set are:',str(YTrain1Count))
YTrain1.head()


# In[295]:


#Variables Independientes Testing Set
XTest1 = featuresRFE[-100:]
XTest1Count = len(XTest1.index)
print('The number of observations in the feature testing set is:',str(XTest1Count))
print(XTest1.head())


# In[297]:


# Variable Dependiente Testing Set.Y Truth
YTest1 = depVar[-100:]
YTest1Count = len(YTest1.index)
print('The number of observations in the Y training set are:',str(YTest1Count))
YTest1.head()


# In[298]:


#Cross Validation
XTrain1, XTest1, YTrain1, YTest1 = train_test_split(XTrain1, YTrain1, test_size=0.3)


# In[299]:


XTrain.shape, XTest.shape


# ## Predictive Models -Experiment 2
# 

# ## Decision Tree

# In[301]:


# Decision Tree
modelDecisionTree1 = DecisionTreeClassifier(criterion='gini', max_depth=10)


# In[302]:


#Training
modelDecisionTree1.fit(XTrain1,YTrain1)


# In[304]:


#Cross Validation
print(cross_val_score(modelDecisionTree1, XTrain1, YTrain1, cv=10, n_jobs=2)) 
modelDecisionTree1.score(XTrain1,YTrain1)


# In[305]:


#Testing
predictionDecisionTree1= modelDecisionTree1.predict(XTest1)


# In[306]:


#Confusion Matrix
confusion_matrix(YTest1,predictionDecisionTree1, labels=['No', 'Yes'])
pd.crosstab(YTest1,predictionDecisionTree1, rownames=['True'], colnames=['Prediction'], margins=True)


# In[308]:


#Classification Report
target_names = ['No', 'Yes']
print(classification_report(YTest1, predictionDecisionTree1,target_names=target_names))


# ## Super Vector Machine

# In[309]:


#SVM
modelSVM1=svm.SVC(C=1, kernel='rbf')


# In[310]:


#Training
modelSVM1.fit(XTrain1,YTrain1)


# In[312]:


#Cross Validation
print(cross_val_score(modelSVM1, XTrain1, YTrain1, cv=10, n_jobs=2)) 
modelSVM1.score(XTrain1,YTrain1)


# In[313]:


#Testing
predictionSVM1= modelSVM1.predict(XTest1)


# In[314]:


#Confusion Matrix
confusion_matrix(YTest1,predictionSVM1, labels=['No', 'Yes'])
pd.crosstab(YTest1,predictionSVM1, rownames=['True'], colnames=['Prediction'], margins=True)


# In[315]:


#Classification Report
target_names = ['No', 'Yes']
print(classification_report(YTest1, predictionSVM1,target_names=target_names))


# ## KNN

# In[316]:


#KNN
modelKnn1 = KNeighborsClassifier(n_jobs=2, n_neighbors=13, weights='uniform', p=1)


# In[317]:


#Training
modelKnn.fit(XTrain1,YTrain1)


# In[318]:


#Cross Validation
print(cross_val_score(modelKnn1, XTrain1, YTrain1, cv=10, n_jobs=2)) 
modelKnn.score(XTrain1,YTrain1)


# In[319]:


#Testing
predictionKnn1= modelKnn.predict(XTest1)


# In[320]:


#Confusion Matrix
confusion_matrix(YTest1,predictionKnn1, labels=['No', 'Yes'])
pd.crosstab(YTest1,predictionKnn1, rownames=['True'], colnames=['Prediction'], margins=True)


# In[321]:


#Classification Report
target_names = ['No', 'Yes']
print(classification_report(YTest1, predictionKnn1,target_names=target_names))


# ## Random Forest

# In[322]:


#RF
modelRF1= RandomForestClassifier(max_depth=7, n_estimators=70, n_jobs=2, criterion='gini')


# In[323]:


#Training
modelRF.fit(XTrain1,YTrain1)


# In[324]:


#Cross Validation
print(cross_val_score(modelRF1, XTrain1, YTrain1, cv=10, n_jobs=2)) 
modelRF.score(XTrain1,YTrain1)


# In[325]:


#Testing
predictionRf1= modelRF.predict(XTest1)


# In[326]:


#Confusion Matrix
confusion_matrix(YTest1,predictionRf1, labels=['No', 'Yes'])
pd.crosstab(YTest1,predictionRf1, rownames=['True'], colnames=['Prediction'], margins=True)


# In[328]:


#Classification Report
target_names = ['No', 'Yes']
print(classification_report(YTest1, predictionRf1,target_names=target_names))


# In[ ]:





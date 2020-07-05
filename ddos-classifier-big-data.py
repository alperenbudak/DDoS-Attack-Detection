# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:02:25 2019

@authors: Batuhan, Alperen, Merve
Data-Communication-DDOS Attack Pretection using Machine Learning
"""
#importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#data preprocessing
cuma_ddos = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
x0 = cuma_ddos.iloc[:,:14]
x1 = cuma_ddos.iloc[:,16:77]
x = pd.concat([x0, x1], axis = 1)
y = cuma_ddos.iloc[:,78:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_min = scaler.fit_transform(x)


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k=20)
fit = bestfeatures.fit(x_min,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(15,'Score'))  #print 15 best features
print(featureScores.nsmallest(10,'Score'))  #print 10 worst features


#Concatenate selected features to obtain new dataframe
x_new = x.iloc[:,40:41] #Packet Length Variance
x_new_2 = x.iloc[:,1:2] #Flow Duration
x_new_3 = x.iloc[:,70:71]#Active Max
x_new_4 = x.iloc[:,2:3]#Total Fwd Packets
x_new_5 = x.iloc[:,:1] #Destination Port
x_new_6 = x.iloc[:,13:14] #Bwd Packet Length Std
x_new_7 = x.iloc[:,50:51]#Average Packet Size
x_new_8 = x.iloc[:,10:11]#Bwd Packet Length Max
x_new_9 = x.iloc[:,52:53] #Avg Bwd Segment size
x_new_10 = x.iloc[:,37:38] #Max Packet Length



x_new   = np.asarray(x_new)
x_new_2 = np.asarray(x_new_2)
x_new_3 = np.asarray(x_new_3)
x_new_4 = np.asarray(x_new_4)
x_new_5 = np.asarray(x_new_5)
x_new_6 = np.asarray(x_new_6)
x_new_7 = np.asarray(x_new_7)
x_new_8 = np.asarray(x_new_8)
x_new_9 = np.asarray(x_new_9)
x_new_10 = np.asarray(x_new_10)


x_new_conc = np.concatenate((x_new, x_new_2, x_new_3, x_new_4, x_new_5, x_new_6,x_new_7,x_new_8,x_new_9,x_new_10), axis=1)



#splitting data to test and train
x_train, x_test, y_train, y_test = train_test_split(x_new_conc, y, 
                                                    test_size = 0.20,
                                                    random_state=0)
from sklearn import preprocessing
lab_enc = preprocessing.LabelEncoder()
y_train_enc = lab_enc.fit_transform(y_train)
y_test_enc = lab_enc.fit_transform(y_test)
y_enc = lab_enc.fit_transform(y)
#convert df to array to use in learning
y_train_enc = np.asarray(y_train_enc).reshape(-1,1)
y_test_enc = np.asarray(y_test_enc).reshape(-1,1)
y_enc = np.asarray(y_enc).reshape(-1,1)
# Feature Scaling in order to use in machine learning algorithms
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


#K-NN model 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 750, metric = 'minkowski')
knn.fit(X_train, y_train_enc.ravel())
# Predicting the Test set results
y_pred_knn = knn.predict(X_test)
y_pred_knn = np.asarray(y_pred_knn).reshape(-1,1)
# Making the Confusion Matrix
cm_knn= confusion_matrix(y_test_enc, y_pred_knn)
print('cm-KNN: \n',cm_knn)


# Fitting Linear Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear' , random_state = 42)
classifier.fit(X_train,y_train_enc.ravel())
svc_pred = classifier.predict(X_test)
cm_svc= confusion_matrix(y_test_enc, svc_pred)
print('cm-SVC: \n',cm_svc)

# Fitting RBF Kernel SVM to the Training set
classifier_rbf = SVC(kernel = 'rbf' , random_state = 42)
classifier_rbf.fit(X_train,y_train_enc.ravel())
svc_pred_rbf = classifier_rbf.predict(X_test)
cm_svc= confusion_matrix(y_test_enc, svc_pred_rbf)
print('cm-SVC: \n',cm_svc)


#Accuracy and F1 score calculation for analysis
from sklearn.metrics import f1_score
y_true = y_test_enc
f1_knn = f1_score(y_true, y_pred_knn,average = 'macro')
f1_svc = f1_score(y_true, svc_pred, average = 'macro')
f1_svc_rbf = f1_score(y_true, svc_pred_rbf, average = 'macro')
print('F1-SVC-LINEAR: \n',f1_svc)
print('F1-SVC-RBF: \n',f1_svc_rbf)
print('F1-KNN: \n',f1_knn)
print('Accuracy of SVC-Linear - Test -  : \n' ,accuracy_score(y_true, 
                                                              svc_pred))
print('Accuracy of SVC-RBF kernel - Test - : \n' ,accuracy_score(y_true, 
                                                                 svc_pred_rbf))
print('Accuracy of KNN - Test -:  \n' , accuracy_score(y_true, y_pred_knn))


#Testing The Obtained Algorithm with Kaggle Data Set 
kaggle_data = pd.read_csv('test_mosaic.csv')

x_kaggle   = kaggle_data.iloc[:50000,42:43] #Packet Length Variance
x_kaggle_2 = kaggle_data.iloc[:50000,1:2] #Flow Duration
x_kaggle_3 = kaggle_data.iloc[:50000,71:72]#Active Max
x_kaggle_4 = kaggle_data.iloc[:50000,2:3]#Total Fwd Packets
x_kaggle_5 = kaggle_data.iloc[:50000,:1] #Destination Port
x_kaggle_6 = kaggle_data.iloc[:50000,13:14] #Bwd Packet Length Std
x_kaggle_7 = kaggle_data.iloc[:50000,52:53] #Average Packet Size
x_kaggle_8 = kaggle_data.iloc[:50000,10:11] #Bwd Packet Length Max
x_kaggle_9 = kaggle_data.iloc[:50000,54:55] #Avg Bwd Segment size
x_kaggle_10 = kaggle_data.iloc[:50000,39:40] #Max Packet Length
 
x_kaggle = np.asarray(x_kaggle)
x_kaggle_2 = np.asarray(x_kaggle_2)
x_kaggle_3 = np.asarray(x_kaggle_3)
x_kaggle_4 = np.asarray(x_kaggle_4)
x_kaggle_5 = np.asarray(x_kaggle_5)
x_kaggle_6 = np.asarray(x_kaggle_6)
x_kaggle_7 = np.asarray(x_kaggle_7)
x_kaggle_8 = np.asarray(x_kaggle_8)
x_kaggle_9 = np.asarray(x_kaggle_9)
x_kaggle_10 = np.asarray(x_kaggle_10)

x_kaggle_conc = np.concatenate((x_kaggle, x_kaggle_2, x_kaggle_3, x_kaggle_4, x_kaggle_5, x_kaggle_6, x_kaggle_7, x_kaggle_8, x_kaggle_9, x_kaggle_10), axis=1)

y_kaggle = kaggle_data.iloc[:50000,77:]


y_kolon_kaggle = []
for i in range(0,len(y_kaggle)):
    if y_kaggle['Label'][i] == 'DoS Hulk' or y_kaggle['Label'][i] == 'DoS slowloris':
    
        y_kolon_kaggle.append('DDoS')
    elif  y_kaggle['Label'][i] == 'BENIGN' :
        y_kolon_kaggle.append( 'BENIGN')


y_new_kaggle = np.asarray(y_kolon_kaggle).reshape(-1,1)

lab_enc_kaggle = preprocessing.LabelEncoder()
y_kaggle_enc = lab_enc_kaggle.fit_transform(y_new_kaggle.ravel())

y_kaggle_enc = np.asarray(y_kaggle_enc).reshape(-1,1)


X_test_kaggle = sc.transform(x_kaggle_conc)
y_kaggle_enc = lab_enc_kaggle.fit_transform(y_new_kaggle.ravel())
# Predicting the KNN - the KAGGLE results
y_kaggle_knn = knn.predict(X_test_kaggle)
y_kaggle_knn = np.asarray(y_kaggle_knn).reshape(-1,1)
# Making the Confusion Matrix
cm_knn= confusion_matrix(y_kaggle_enc, y_kaggle_knn)
print('cm-KNN: \n',cm_knn)

# Predict Linear Kernel SVM - the KAGGLE set
svc_pred_kaggle = classifier.predict(X_test_kaggle)
cm_svc_kaggle = confusion_matrix(y_kaggle_enc, svc_pred_kaggle)
print('cm-SVC-KAGGLE: \n',cm_svc_kaggle)

# Predict RBF Kernel SVM to the KAGGLE set
svc_pred_rbf_kaggle = classifier_rbf.predict(X_test_kaggle)
cm_svc_kaggle_rbf= confusion_matrix(y_kaggle_enc, svc_pred_rbf_kaggle)
print('cm-SVC-KAGGLE-RBF: \n',cm_svc_kaggle_rbf)



f1_svc_kaggle = f1_score(y_kaggle_enc, svc_pred_kaggle, average = 'macro')
f1_svc_rbf_kaggle = f1_score(y_kaggle_enc, svc_pred_rbf_kaggle, average = 'macro')
f1_knn_kaggle = f1_score(y_kaggle_enc, y_kaggle_knn, average = 'macro')

print('F1-SVC-LINEAR-KAGGLE: \n',f1_svc_kaggle)
print('F1-SVC-RBF-KAGGLE: \n',f1_svc_rbf_kaggle)
print('F1-KNN-KAGGLE: \n',f1_knn_kaggle)

print('Accuracy of SVC-Linear - Kaggle -  : \n' ,accuracy_score(y_kaggle_enc, svc_pred_kaggle))
print('Accuracy of SVC-RBF kernel - Kaggle - : \n' ,accuracy_score(y_kaggle_enc, svc_pred_rbf_kaggle))
print('Accuracy of KNN - Kaggle -:  \n' , accuracy_score(y_kaggle_enc, y_kaggle_knn))

#YENİ DATA SETİYLE DENEME

monday = pd.read_csv('Monday-WorkingHours.pcap_ISCX.csv')


x_monday   = monday.iloc[:40000,42:43] #Packet Length Variance
x_monday_2 = monday.iloc[:40000,1:2] #Flow Duration
x_monday_3 = monday.iloc[:40000,71:72]#Active Max
x_monday_4 = monday.iloc[:40000,2:3]#Total Fwd Packets
x_monday_5 = monday.iloc[:40000,:1] #Destination Port
x_monday_6 = monday.iloc[:40000,13:14] #Bwd Packet Length Std
x_monday_7 = monday.iloc[:40000,53:54]#Average Packet Size
x_monday_8 = monday.iloc[:40000,10:11]#Bwd Packet Length Max
x_monday_9 = monday.iloc[:40000,54:55] #Avg Bwd Segment size
x_monday_10 = monday.iloc[:40000,39:40] #Max Packet Length

 
x_monday   = np.asarray(x_monday)
x_monday_2 = np.asarray(x_monday_2)
x_monday_3 = np.asarray(x_monday_3)
x_monday_4 = np.asarray(x_monday_4)
x_monday_5 = np.asarray(x_monday_5)
x_monday_6 = np.asarray(x_monday_6)
x_monday_7 = np.asarray(x_monday_7)
x_monday_8 = np.asarray(x_monday_8)
x_monday_9 = np.asarray(x_monday_9)
x_monday_10 = np.asarray(x_monday_10)

x_monday_conc = np.concatenate((x_monday, x_monday_2, x_monday_3, x_monday_4, x_monday_5, x_monday_6, x_monday_7, x_monday_8, x_monday_9, x_monday_10), axis=1)

y_monday = monday.iloc[:40000,78:]



lab_enc_monday = preprocessing.LabelEncoder()
y_monday_enc = lab_enc_monday.fit_transform(y_monday)

y_monday_enc = y_monday_enc.reshape(-1,1)



X_test_monday = sc.transform(x_monday_conc)

# Predicting the KNN set results
y_monday_knn = knn.predict(X_test_monday)
y_monday_knn = np.asarray(y_monday_knn).reshape(-1,1)
# Making the Confusion Matrix
cm_knn_monday= confusion_matrix(y_monday_enc, y_monday_knn)
print('cm-KNN: \n',cm_knn_monday)
# Predict Linear Kernel SVM to the Monday set
svc_pred_monday = classifier.predict(X_test_monday)
cm_svc_monday= confusion_matrix(y_monday_enc, svc_pred_monday)
print('cm-SVC-MONDAY: \n',cm_svc_monday)
# Predict RBF Kernel SVM to the Monday set
svc_pred_rbf_monday = classifier_rbf.predict(X_test_monday)
cm_svc_monday_rbf= confusion_matrix(y_monday_enc, svc_pred_rbf_monday)
print('cm-SVC-MONDAY-RBF: \n',cm_svc_monday_rbf)

f1_svc_monday = f1_score(y_monday_enc, svc_pred_monday, average = 'macro')
f1_svc_rbf_monday = f1_score(y_monday_enc, svc_pred_rbf_monday, average = 'macro')
f1_knn_monday = f1_score(y_monday_enc, y_monday_knn, average = 'macro')


print('F1-SVC-LINEAR-MONDAY: \n',f1_svc_monday)
print('F1-SVC-RBF-MONDAY: \n',f1_svc_rbf_monday)
print('F1-knn-MONDAY: \n',f1_knn_monday)

print('Accuracy of SVC-Linear - Monday -  : \n' ,accuracy_score(y_monday_enc, svc_pred_monday))
print('Accuracy of SVC-RBF kernel - Monday - : \n' ,accuracy_score(y_monday_enc, svc_pred_rbf_monday))
print('Accuracy of KNN - Monday -:  \n' , accuracy_score(y_monday_enc, y_monday_knn))


#import seaborn as sn
#df_cm = pd.DataFrame(cm_svc_monday, columns = np.unique(y_monday ), index = np.unique(y_monday))

#plt.figure(figsize = (10,7))
#sn.set(font_scale = 1.4) #for label size
#ax = sn.heatmap(df_cm, cmap = "Blues", annot = True, annot_kws = {"size" : 16}, fmt='g')






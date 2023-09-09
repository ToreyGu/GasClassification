
'''
@filename              :main.py
@createFileTime        :2023/02/27 16:44:34
@author                :Tianyi Gu
@version               :1.0.1
@description           :A Data Process Program Based on Sklearn
Created in Key Laboratory of Advanced Gas Sensors,Jilin University
'''

import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xlrd
from mpl_toolkits import mplot3d
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import cross_val_score

import matplotlib.colors as mcolors



class LoadData(object):

    def __init__(self, worksheet):
        self._worksheet = worksheet
        self._accuracy = None
        self._predict = []

        print("Load File Success")


    def getRow(self, row):
        cols = []
        sheet_names = self._worksheet.sheet_names()
        for sheet_name in sheet_names:
            sheet = self._worksheet.sheet_by_name(sheet_name)
   
            cols.append(sheet.col_values(row))
            return np.array(cols)[:, 1:].T


    def getPartMatrix(self, startPosition, steps=3, intervals=2):

        matrix = self.getRow(startPosition)
        for step in range(steps-1):
            matrix = np.hstack((matrix, self.getRow(
                startPosition + (step+1)*intervals)))
        return matrix


    def getMatrix(self, startPosition, steps=3, intervals=2):

        FullMatrix = self.getPartMatrix(startPosition, steps)
        for interval in range(intervals-1):
            FullMatrix = np.vstack(
                (FullMatrix, self.getPartMatrix(startPosition + interval+1)))
        return FullMatrix

    def getRangeMatrix(self, startPosition, StopPosition):
 
        rangeMatrix = self.getRow(startPosition)
        while startPosition < StopPosition:
            rangeMatrix = np.hstack(
                (rangeMatrix, self.getRow(startPosition + 1)))
            startPosition += 1
        return rangeMatrix

    def drawBundary(self,  data_input, model, pixel=50):

        data_input1_max = max(data_input[:, 0])
        data_input1_min = min(data_input[:, 0])
        data_input2_max = max(data_input[:, 1])
        data_input2_min = min(data_input[:, 1])
        scanx = np.linspace(data_input1_min-1, data_input1_max+1, int(
            pixel*(data_input1_max-data_input1_min)))
        scany = np.linspace(data_input2_min-1, data_input2_max+1, int(
            pixel*(data_input2_max-data_input2_min)))
        predict = []
        for i in scany:
            temp = []
            for q in scanx:

                temp.append(model.predict([[q, i]])[0])

            predict.append(temp)
        plt.contourf(scanx, scany, predict, cmap=plt.cm.Accent, alpha=0.2)
        plt.show()
    def pcaData2D(self, matrix, scaler):
 
        matrix = scaler.fit_transform(matrix)
        pca = PCA(n_components=2)
        matrix = pca.fit_transform(matrix)

        return matrix

    def attachMatrix(self, FrontMatrix, BackMatrix):

        attachedMatrix = np.hstack((FrontMatrix, BackMatrix))
        return attachedMatrix

    def drawPcaScatters(self, fullMatrix, spicesList, title=None):


        width = 3
        
        pca1 = np.array(fullMatrix[: , 0]).astype(float)
        pca2 = np.array(fullMatrix[: , 1]).astype(float)
        colors = fullMatrix[: , 2]
        spicesIndex = np.unique(fullMatrix[: , 3],return_index=True)[1]
        spicesIndex = np.sort(spicesIndex)
        spicesName  = []
        for i in range(len(spicesIndex)):    
            index = spicesIndex[i]
            spicesName.append(fullMatrix[index,3])
        plt.figure(figsize=(12,9) , dpi=300)
        scatter = plt.scatter(pca1, pca2, c= [colors] , cmap= plt.cm.Accent)
        font1 = {'family' : 'Times New Roman','weight' : 'normal','size' : 20}       
        plt.title(title)
        plt.legend(handles=scatter.legend_elements()[0] ,labels= spicesName,prop=font1, loc = 'upper right')
        plt.yticks(fontproperties = 'Times New Roman', size = 18)
        plt.xticks(fontproperties = 'Times New Roman', size = 18)
        ax=plt.gca();
        ax.spines['bottom'].set_linewidth(width)
        ax.spines['left'].set_linewidth(width)
        ax.spines['top'].set_linewidth(width)
        ax.spines['right'].set_linewidth(width)
        
    def initCal(self):
        if self._accuracy or self._predict is not None:
            self._accuracy = None
            self._predict = []

    def accuracyCal(self, fullMatrix, model):
        self.initCal()
        accuracy = 0
        self._predict = model.predict(fullMatrix[:, 0:2])
        for i in range(len(self._predict)):
            if self._predict[i] == fullMatrix[i, 2]:
                accuracy += 1
        self._accuracy = float(accuracy/len(self._predict))
        print('Accuracy in %s model is: %3.2f Percent' %
              (model, (self._accuracy * 100)))

    def crossCal(self, fullMatrix, model, epoch=10):
        scores = cross_val_score(
            model, fullMatrix[:, 0:2], fullMatrix[:, 2], cv=epoch)
        print("Accuracy of %s model is: %0.2f (+/- %0.2f)" %
              (model, scores.mean(), scores.std() * 2))
        return scores.mean(), (2*scores.std())


worksheet = xlrd.open_workbook('data.xls')

data = LoadData(worksheet)
responseRaw = data.getRangeMatrix(1,3)
spices = data.getRow(7)
spices = spices.astype(float)
spices = spices.astype(int)

spicesName = data.getRow(8)
spicesName = spicesName.astype(str)

response = data.pcaData2D(responseRaw, scaler=StandardScaler())

spicesFull = data.attachMatrix(spices, spicesName)
spicesName = np.unique(spicesName)
spicesList = np.unique(spices)

fullMatrix = data.attachMatrix(response, spicesFull)
data.drawPcaScatters(fullMatrix, 
                      spicesList=spicesList)
rawMatrix = data.attachMatrix(response, spices)

from sklearn import tree  
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(response, spices[:,0])

LDA = LinearDiscriminantAnalysis(n_components=2)
LDA.fit(response, spices[:,0])

lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(response, spices[:,0])

svm_lin = SVC(kernel='linear', C=10.0, random_state=10)
svm_lin.fit(response,spices[:,0])

svm_rbf = SVC(kernel='rbf', random_state=1,
              gamma=0.20, C=100.0)  
svm_rbf.fit(response, spices[:,0])

decTree = tree.DecisionTreeClassifier()  
decTree.fit(response, spices[:,0]) 

radFor = RandomForestClassifier()
radFor.fit(response, spices[:,0])

data.crossCal(rawMatrix, lr)
data.crossCal(rawMatrix, LDA)
data.crossCal(rawMatrix, KNN)
data.crossCal(rawMatrix, svm_lin)
data.crossCal(rawMatrix, svm_rbf)
data.crossCal(rawMatrix, decTree)
data.crossCal(rawMatrix, radFor)



data.drawPcaScatters(fullMatrix,spicesList=spicesList)
data.drawBundary(rawMatrix, svm_rbf, pixel=500)


type(fullMatrix[:,0:2])
type(response)


from sklearn.model_selection import cross_val_score
scores = cross_val_score(LDA, rawMatrix [:,0:2], spices[:,0], cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



data.drawPcaScatters(fullMatrix,spicesList=spicesList)
data.drawBundary(rawMatrix, lr, pixel=500)


data.drawPcaScatters(fullMatrix,spicesList=spicesList)
data.drawBundary(rawMatrix, svm_lin, pixel=500)


data.drawPcaScatters(fullMatrix,spicesList=spicesList)
data.drawBundary(rawMatrix, LDA, pixel=500)

data.drawPcaScatters(fullMatrix,spicesList=spicesList)
data.drawBundary(rawMatrix, KNN, pixel=500)


data.drawPcaScatters(fullMatrix,spicesList=spicesList)
data.drawBundary(rawMatrix, decTree, pixel=500)


data.drawPcaScatters(fullMatrix,spicesList=spicesList)
data.drawBundary(rawMatrix, radFor, pixel=50)


spicesIndex = np.unique(fullMatrix[: , 3],return_index=True)[1]
spicesIndex = np.sort(spicesIndex)

spicesLabel  = []
for i in range(len(spicesIndex)):    
    index = spicesIndex[i]
    spicesLabel.append(fullMatrix[index,3])

from sklearn.model_selection import train_test_split
import joblib
x_train , x_test , y_train , y_test = train_test_split(response ,spices[:,0],train_size=0.7)
svm_rbf =  radFor  
svm_rbf.fit(x_train, y_train)


from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

y_train_predict = svm_rbf.predict(x_train)
train_confuse_matrix = confusion_matrix(y_true = y_train, y_pred =y_train_predict)
train_confuse_matrix_norm = train_confuse_matrix.astype('float') / train_confuse_matrix.sum(axis=1)[:, np.newaxis]   
train_confuse_matrix_norm = np.around(train_confuse_matrix_norm, decimals=2)
cm_display = ConfusionMatrixDisplay(train_confuse_matrix_norm, display_labels = spicesLabel)

cm_display.plot(cmap = 'Blues')

plt.show()
plt.clf()



from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
y_test_predict = svm_rbf.predict(x_test)
test_confuse_matrix = confusion_matrix(y_true = y_test, y_pred =y_test_predict)
test_confuse_matrix_norm = test_confuse_matrix.astype('float') / test_confuse_matrix.sum(axis=1)[:, np.newaxis]    
test_confuse_matrix_norm = np.around(test_confuse_matrix_norm, decimals=2)
cm_display = ConfusionMatrixDisplay(test_confuse_matrix_norm, display_labels = spicesLabel)
cm_display.plot(cmap = 'Blues')

plt.show()
plt.clf()



from sklearn.metrics import f1_score
f1_score(y_test , y_test_predict , average='macro')


y_predict = svm_rbf.predict(response)
confuse_matrix = confusion_matrix(y_true = spices[:,0], y_pred =y_predict)
confuse_matrix_norm = test_confuse_matrix.astype('float') / test_confuse_matrix.sum(axis=1)[:, np.newaxis]    
confuse_matrix_norm = np.around(confuse_matrix_norm, decimals=2)
cm_display = ConfusionMatrixDisplay(confuse_matrix_norm, display_labels = spicesLabel)
cm_display.plot(cmap = 'Blues')


import joblib
joblib.dump(decTree, 'dectree3.pkl')



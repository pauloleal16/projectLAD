import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.basemap import Basemap
from geopy.geocoders import Nominatim

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.linear_model import RidgeCV
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier



eq = pd.read_csv("earthquakes.csv")
eq = eq.drop("Unnamed: 0", axis=1)

eq.dropna(subset=["magType"], inplace=True)
mediana_nst = eq["nst"].median()
eq["nst"].fillna(mediana_nst, inplace=True)
media_gap = eq["gap"].mean()
eq["gap"].fillna(media_gap, inplace=True)
media_dmin = eq["dmin"].mean()
eq["dmin"].fillna(media_dmin, inplace=True)
media_rms = eq["rms"].mean()
eq["rms"].fillna(media_rms, inplace=True)
eq.dropna(subset=["place"], inplace=True)
media_horizontalError = eq["horizontalError"].mean()
eq["horizontalError"].fillna(media_horizontalError, inplace=True)
media_depthError = eq["depthError"].mean()
eq["depthError"].fillna(media_depthError, inplace=True)
media_magError = eq["magError"].mean()
eq["magError"].fillna(media_magError, inplace=True)
mediana_magNst = eq["magNst"].median()
eq["magNst"].fillna(mediana_magNst, inplace=True)
eq.dropna(subset=["status"], inplace=True)
eq.dropna(subset=["locationSource"], inplace=True)
eq.dropna(subset=["magSource"], inplace=True)


# COMPONENTE 2
#remover as indesejadas
dfInput = eq.drop(columns = ['time','magType','gap','dmin','rms','net','id','updated','place','type','horizontalError','depthError','magError','magNst','status','locationSource','magSource', 'mag'])
dfOutput = eq['mag']
print(dfInput)
print(dfOutput)

dfN = np.array([[41.927, 20.543, 10.0, 18]])
dfPred = pd.DataFrame(dfN)

X_train, X_test, Y_train, Y_test = train_test_split(dfInput.values, dfOutput.values, test_size=0.3)

lr = LinearRegression()
lr.fit(X_train, Y_train)
scores = cross_val_score (lr, X_train, Y_train, cv=5)

print("-> Linear Regression")
print("mean:", scores.mean())
print("std:", scores.std())
print("score:", lr.score(X_test,Y_test))

pred = lr.predict(dfPred)
print("predict:", pred)

print("----------------------")

#Ridge

X = dfInput
Y = eq['mag']

lrr = Ridge()
lrr.fit(X,Y)

print("-> Ridge")
print("R2={:.2f}".format(r2_score(Y,lrr.predict(X))))
print("MAE={:.2f}".format(mean_absolute_error(Y,lrr. predict(X))))

rcv = RidgeCV(alphas=np.arange(0.1,1,0.1))
rcv.fit(X,Y)

#print("Alpha:{:.2f}".format(rcv.alpha_))
#o valor de alpha fornece os mesmos valores
print(rcv.coef_)
print(rcv.score(X,Y))
print("----------------------")


#Lasso

clf = linear_model.Lasso(alpha=1)
clf.fit(X,Y)

print("-> Lasso")
print("R2={:.2f}".format(r2_score(Y, clf.predict(X))))
print("MAE={:.2f}".format(mean_absolute_error(Y, clf.predict(X))))
print("score:", clf.score(X,Y))
print("----------------------")

''' 
#SVR regressor

# print the names of the 13 features (X)
print("-> SVR")
print("Features: \n", dfInput)
# print the label type
print("Labels: \n", eq['mag'])
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(dfInput, eq['mag'],test_size=0.3,random_state=109) # 70% training and 30% test

#Create a svm Classifier
clfSVR = svm.SVR(kernel='linear') # Linear Kernel
#####clf = svm.SVC(kernel='poly') # Poly Kernel

#Train the model using the training sets
clfSVR.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = clfSVR.predict(X_test)
# Model Accuracy: how often is the classifier correct?
print("----------Métricas de avaliação-----------")
print("MAE={:.2f}".format(mean_absolute_error(y_test, y_pred)))
print("R2={:.2f}".format(r2_score(y_test, y_pred)))

plt.figure(figsize=(8, 8))
sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train)
plt.title("SVR")
plt.show()
'''


#KNN

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(dfInput, eq['mag'], test_size=0.3,random_state=109) # 70% training and 30% test

#Create the Classifier
clf = KNeighborsClassifier(n_neighbors=8)
#Train the model using the training sets
clf.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("-> KNN")
print("----------Classification report-----------")
print(classification_report(y_test, y_pred))


xvalues= [2, 4, 6, 8, 10, 12, 14, 16, 18] #number of neighbours
values = [0.91, 0.94, 0.97, 0.98, 0.98, 0.97, 0.98, 0.98, 0.98] #accuracy
plt.plot(xvalues, values)
plt.xlabel('Number of neighbours')
plt.ylabel('Accuracy')
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.basemap import Basemap
from geopy.geocoders import Nominatim

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA

#TRATAMENTO DE DADOS
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
################################################


#COMPONENTE_2

#DADOS A UTILIZAR
dfInput = eq.drop(columns = ['time','magType','gap','dmin','rms','net','id','updated','place','type','horizontalError','depthError','magError','magNst','status','locationSource','magSource', 'mag'])
dfOutput = eq['mag']
print(dfInput)
print(dfOutput)

#Valores para previsão
dfN = np.array([[41.927, 20.543, 10.0, 18]])
dfPred = pd.DataFrame(dfN)

#Linear Regression

X_train, X_test, Y_train, Y_test = train_test_split(dfInput.values, dfOutput.values, test_size=0.3)

lr = LinearRegression()
lr.fit(X_train, Y_train)
scores = cross_val_score (lr, X_train, Y_train, cv=5)

print("\n-> Linear Regression")
print("std:", scores.std())
print("score:", lr.score(X_test,Y_test))
print("mean:", scores.mean())
pred = lr.predict(dfPred)
print("predict:", pred)
print("----------------------")


#Logistic -> não é possível de fazer pois serve para categorização


#Ridge

X = dfInput
Y = eq['mag']
lrr = Ridge()
lrr.fit(X,Y)
print("\n-> Ridge")
print("R2={:.2f}".format(r2_score(Y,lrr.predict(X))))
print("MAE={:.2f}".format(mean_absolute_error(Y,lrr. predict(X))))

scores = cross_val_score(lrr, X, Y, cv=5)

rcv = RidgeCV(alphas=np.arange(0.1,1,0.1))
rcv.fit(X,Y)
print("Alpha:{:.2f}".format(rcv.alpha_)) #o valor de alpha não altera o resultado
print(rcv.coef_)
print("score:", rcv.score(X,Y))
print("mean scores", scores.mean())
print("Predict:", rcv.predict(dfN))
print("----------------------")

#Lasso

clf = linear_model.Lasso(alpha=1)
clf.fit(X,Y)
scores = cross_val_score(clf, X, Y, cv=5)

print("\n-> Lasso")
print("R2={:.2f}".format(r2_score(Y, clf.predict(X))))
print("MAE={:.2f}".format(mean_absolute_error(Y, clf.predict(X))))
print("score:", clf.score(X,Y))
print("mean scores", scores.mean())
print("Predict:",clf.predict(dfN))
print("----------------------")

''' 
#SVR regressor without PCA

print("-> SVR")

X_train, X_test, y_train, y_test = train_test_split(dfInput, eq['mag'],test_size=0.3,random_state=109) # 70% training and 30% test

clfSVR = svm.SVR(kernel='linear') # Linear Kernel

clfSVR.fit(X_train, y_train)
y_pred = clfSVR.predict(X_test)
print("MAE={:.2f}".format(mean_absolute_error(y_test, y_pred)))
print("R2={:.2f}".format(r2_score(y_test, y_pred)))

plt.figure(figsize=(8, 8))
sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train)
plt.title("SVR")
plt.show()
'''

'''
#SVR regressor with PCA

print("-> SVR")
X_train, X_test, y_train, y_test = train_test_split(dfInput, eq['mag'],test_size=0.3,random_state=109) # 70% training and 30% test

clfSVR = svm.SVR(kernel='linear') # Linear Kernel

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca=pca.transform(X_test)

clfSVR.fit(X_train_pca, y_train)
y_pred = clfSVR.predict(X_test_pca)
print("MAE={:.2f}".format(mean_absolute_error(y_test, y_pred)))
print("R2={:.2f}".format(r2_score(y_test, y_pred)))

plt.figure(figsize=(8, 8))
sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train)
plt.title("SVR")
plt.show()
'''

#KNN

X_train, X_test, y_train, y_test = train_test_split(dfInput, eq['mag'], test_size=0.3,random_state=109) # 70% training and 30% test
clfKNN = KNeighborsRegressor(n_neighbors=8)
clfKNN.fit(X_train, y_train)
y_pred = clfKNN.predict(X_test)

scores = cross_val_score(clfKNN, X, Y, cv=5)

print("\n-> KNN")
print("MAE= {:.2f}".format(mean_absolute_error(y_test, y_pred)))
print("R2= {:.2f}".format(r2_score(y_test, y_pred)))
print("Score:", clfKNN.score(X,Y))
print("mean scores", scores.mean())
print("Predict:", clfKNN.predict(dfN))

#Dados para o gráfico
xvalues = [2, 4, 6, 8, 10, 12, 14, 16, 18] #number of neighbours
values = [0.91, 0.94, 0.97, 0.98, 0.98, 0.97, 0.98, 0.98, 0.98] #accuracy
plt.plot(xvalues, values)
plt.xlabel('Number of neighbours')
plt.ylabel('Accuracy')
plt.show()
print("----------------------")

#Decision Trees
X_train, X_test, y_train, y_test = train_test_split(dfInput, eq['mag'], test_size=0.5, random_state=0)
clfDecisionTrees = DecisionTreeRegressor(max_depth=2)
#clf=tree.DecisionTreeClassifier(criterion='entropy',max_depth=2)
clfDecisionTrees.fit(X_train, y_train)

fig = plt.figure(figsize=(10,10))
_ = plot_tree(clfDecisionTrees, filled=True)
#fig.savefig("decistion_tree.png")
plt.show()
print("\n-> Decision Trees")
print("----------------------")


# Random Forest
X_train, X_test, Y_train, Y_test = train_test_split(dfInput, eq['mag'], test_size=0.3)
rf = RandomForestRegressor(max_depth=10, random_state=0)
clfRF = rf.fit(X_train,Y_train)

scores = cross_val_score(clfRF, X, Y, cv=5)

print("\n-> Randoom Forest regressor")
print("R2={:.2f}".format(r2_score(Y_test,clfRF.predict(X_test))))
print("MAE={:.2f}".format(mean_absolute_error(Y_test, clfRF.predict(X_test))))
print("score:", clfRF.score(X,Y))
print("mean scores", scores.mean())
print("Predict:",clfRF.predict(dfN))
print("----------------------")

#Boosting
X_train,X_test,y_train,y_test=train_test_split(dfInput,eq['mag'],test_size=0.3,random_state=0)
rf=AdaBoostRegressor(n_estimators=10,random_state=0)

clfBR = rf.fit(X_train,y_train)
y_pred = clfBR.predict(X_test)

scores = cross_val_score(clfBR, X, Y, cv=5)

print("\n-> Boosting")
print("MAE={:.2f}".format(mean_absolute_error(Y_test, clfBR.predict(X_test))))
print("R2={:.2f}".format(r2_score(Y_test,clfBR.predict(X_test))))
print("score:", clfBR.score(X,Y))
print("mean scores", scores.mean())
print("Predict:",clfBR.predict(dfN))


#Neural networks
X_train,X_test,y_train,y_test=train_test_split(dfInput,eq['mag'],test_size=0.3)

rfNeural = MLPRegressor(max_iter=200, solver='lbfgs', alpha=0.01)
clfNeural=rfNeural.fit(X_train,Y_train)

#scores = cross_val_score(clfNeural, dfInput, eq['mag'], cv=5)

print("-> Neural networks")
print("MAE={:.2f}".format(mean_absolute_error(Y_test, clfNeural.predict(X_test))))
print("R2={:.2f}".format(r2_score(Y_test,clfNeural.predict(X_test))))
print("score:", clfNeural.score(X_test,Y_test))
#print("mean scores", scores.mean)
print("Predict:",clfNeural.predict(dfN))


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

eq = pd.read_csv("earthquakes.csv")
eq = eq.drop("Unnamed: 0", axis=1)

print(eq)
print("\n INFO:")
print(eq.info())
print("\n SHAPE:")
print(eq.shape)
print("\n COLUNAS:")
print(eq.columns)
print("\n TAMANHO:")
print(eq.size)
print("\n DESCRIBE:")
print(eq.describe())
print("\n NULOS:")
print(eq.isnull().sum())

#Mediana para variaveis discretas e media para variaveis continuas

print("\n SUBSTITUIÇÃO DOS NULOS")

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

print(eq.isnull().sum())



# COMPONENTE 2

dfInput = eq.drop(columns = ['longitude','latitude','depth','nst'])
#remover as indesejadas
dfOutput = eq['mag']
print(dfInput)

dfN = np.array([[41.927, 20.543, 10.0, 18]])
dfPred = pd.DataFrame(dfN)

X_train, X_test, Y_train, Y_test = train_test_split(dfInput, dfOutput, test_size=0.3)

lr = LinearRegression()
lr.fit(X_train, Y_train)
scores = cross_val_score (lr, X_train, Y_train, cv=5)

print(scores.mean())
print(scores.std())
print(lr.score(X_test,Y_test))

pred = lr.predict(dfPred)
print(pred)
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns



eq = pd.read_csv("earthquakes.csv")
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

#mediana para variaveis discretas e media para continuas

print("\n SUBSTITUIÇÃO DOS NULOS")

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


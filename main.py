import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

eq = pd.read_csv("earthquakes.csv")
eq = eq.drop("Unnamed: 0",axis=1)

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

print("\n Correlações")

print(eq.corr(numeric_only=True))

sns.heatmap(eq.corr(numeric_only=True), annot=True)
plt.figure()

'''

#Gráficos das variáveis

#Displot magnitude
sns.displot(eq["mag"])
plt.figure()

#Displot latitude
sns.displot(eq["latitude"])
plt.figure()

#Displot longitude
sns.displot(eq["longitude"])
plt.figure()

#Histograma depth
plt.hist(eq["depth"])
plt.xlabel("depth")
plt.ylabel("Count")
plt.figure()

#Histograma nst
plt.hist(eq["nst"])
plt.xlabel("nst")
plt.ylabel("Count")
plt.figure()

#Displot gap
sns.displot(eq["gap"])
plt.figure()

#Displot dmin
sns.displot(eq["dmin"])
plt.xlim(0, 15)
plt.figure()

#Displot rms
sns.displot(eq["rms"])
plt.xlim(0, 5)
plt.figure()

#Displot horizontalError
sns.displot(eq["horizontalError"])
plt.figure()

#Histograma depthError
plt.hist(eq["depthError"])
plt.xlabel("depthError")
plt.ylabel("Count")
plt.figure()

#Displot magError
sns.displot(eq["magError"])
plt.figure()

#Displot magNst
sns.displot(eq["magNst"])
plt.show()

'''

#comentários para fins de desempenho

'''
print("\n Covariancias")

df = eq["mag"].cov(eq["nst"])
print("mag x nst:", df)
plt.scatter(eq["mag"], eq["nst"])
plt.xlabel("mag")
plt.ylabel("nst")
plt.figure()

df = eq["mag"].cov(eq["horizontalError"])
print("mag x horizontalError:", df)
plt.scatter(eq["mag"], eq["horizontalError"])
plt.xlabel("mag")
plt.ylabel("horizontalError")
plt.figure()

df = eq["dmin"].cov(eq["horizontalError"])
print("dmin x horizontalError:", df)
plt.scatter(eq["dmin"], eq["horizontalError"])
plt.xlabel("dmin")
plt.ylabel("horizontalError")
plt.figure()

df = eq["mag"].cov(eq["rms"])
print("mag x rms:", df)
plt.scatter(eq["mag"], eq["rms"])
plt.xlabel("mag")
plt.ylabel("rms")
plt.figure()
'''

'''
eq.groupby(["mag"])["depth"].count().plot(kind="bar")
plt.show()

sns.set_style("whitegrid")
sns.barplot(x='mag', y ='depth', data=eq, estimator=len)
plt.show()
'''

# ax = sns.barplot(x='mag',y ='depth', hue='place', data=eq, estimator=len)
# plt.figure()

sns.violinplot(x="mag", data=eq)
plt.figure()

sns.jointplot(x='mag', y='depth', data=eq)
plt.figure()

sns.regplot(x='mag', y='depth', data=eq)
plt.figure()

sns.boxplot(x='mag', y='depth', data=eq)
plt.figure()


# sns.boxplot(x='mag',y='depth', hue='place', data=eq)
# plt.figure()


sns.stripplot(x='mag', y='depth', data=eq) #SWARMPLOT SUPOSTAMENTE MAS ERRO DISSE STRIPPLOT
plt.figure()

'''
sns.pairplot(eq)
plt.show()
'''

'''
#NORMALIZAÇÃO
normMinMax=MinMaxScaler()
#Transform data
norm=normMinMax.fit_transform(eq[["mag"]].values)
print(norm)
plt.plot(norm)
plt.figure()


#STANDARDIZAÇÃO
scale=StandardScaler()
#standardization of dependent variables
scaled_data=scale.fit_transform(eq.reshape(-1, 1))
print(eq.mean())
plt.hist(scaled_data, 100)
plt.show()
'''

#Quantos sismos ocorreram por país e fazer um histograma
#Numero de sismos por ano e fazer histograma
#Países com mais sismos

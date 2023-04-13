import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


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


""""
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
plt.figure()
"""


#comentários para fins de desempenho

"""
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


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

eq.groupby(["mag"])["depth"].count().plot(kind="bar")
plt.figure()

sns.set_style("whitegrid")
sns.barplot(x='mag', y ='depth', data=eq, estimator=len)
plt.show()


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


sns.violinplot(x="mag", data=eq)
plt.figure()

sns.jointplot(x='mag', y='nst', data=eq)
plt.figure()

sns.regplot(x='mag', y='nst', data=eq)
plt.figure()

eq["magC"] = pd.cut(eq["mag"], bins=[2,3,4,5,6,7,8,9], labels=["2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9"])
sns.boxplot(x='magC', y='depth', data=eq, width=1)
plt.figure()

eq["magC"] = pd.cut(eq["mag"], bins=[2,3,4,5,6,7,8,9], labels=["2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9"])
sns.stripplot(x='magC', y='depth', data=eq)
plt.figure()
"""

#PAIRPLOT
colunas_selecionadas = ['mag', 'nst', 'depth', 'horizontalError', 'dmin', 'rms']
sns.pairplot(eq[colunas_selecionadas])
plt.show()


"""
#NORMALIZAÇÃO
scaler=MinMaxScaler()
coluna = eq['mag']
scaler.fit(coluna.values.reshape(-1, 1))
coluna_normalizada = scaler.transform(coluna.values.reshape(-1, 1))
eq_normalizado = pd.DataFrame(coluna_normalizada, columns=['mag'])
eq_normalizado.hist()
plt.figure()

#Padronização
scaler=StandardScaler()
coluna = eq['gap']
scaler.fit(coluna.values.reshape(-1,1))
coluna_padronizada = scaler.transform(coluna.values.reshape(-1,1))
eq_padronizado = pd.DataFrame(coluna_padronizada, columns=['gap'])
eq_padronizado.hist()
plt.show()


#TESTE
valor_procurado = '2000'
coluna = 'time'
numero_de_ocorrencias = (eq[coluna] == valor_procurado).sum()
print("No ano",valor_procurado,"ocorreram", numero_de_ocorrencias, "sismos")
"""

#Quantos sismos ocorreram por país e fazer um histograma  ???
#Numero de sismos por ano e fazer histograma
#Países com mais sismos ???
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.basemap import Basemap
from geopy.geocoders import Nominatim

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

print("\n Correlações")
print(eq.corr(numeric_only=True))
sns.heatmap(eq.corr(numeric_only=True), annot=True)
plt.figure()


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


print("\n Covariancias")

#mag x nst
df = eq["mag"].cov(eq["nst"])
print("mag x nst:", df)
plt.scatter(eq["mag"], eq["nst"])
plt.xlabel("mag")
plt.ylabel("nst")
plt.figure()

#mag x horizontalError
df = eq["mag"].cov(eq["horizontalError"])
print("mag x horizontalError:", df)
plt.scatter(eq["mag"], eq["horizontalError"])
plt.xlabel("mag")
plt.ylabel("horizontalError")
plt.figure()

#dmin x horizontalError
df = eq["dmin"].cov(eq["horizontalError"])
print("dmin x horizontalError:", df)
plt.scatter(eq["dmin"], eq["horizontalError"])
plt.xlabel("dmin")
plt.ylabel("horizontalError")
plt.figure()

#mag x rms
df = eq["mag"].cov(eq["rms"])
print("mag x rms:", df)
plt.scatter(eq["mag"], eq["rms"])
plt.xlabel("mag")
plt.ylabel("rms")
plt.figure()


#violinplot para analisar a distribuição da mag ao longo do tempo
sns.violinplot(x="mag", data=eq)
plt.figure()


#Reta regressão linear -> mag x nst
sns.regplot(x='mag', y='nst', data=eq)
plt.figure()


#Relação entre magnitude e profundidade
#Análise através de boxplot
eq["magC"] = pd.cut(eq["mag"], bins=[2,3,4,5,6,7,8,9], labels=["2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9"])
sns.boxplot(x='magC', y='depth', data=eq, width=1)
plt.figure()


#Análise através de stripplot
sns.stripplot(x='magC', y='depth', data=eq)
plt.figure()


#Gráfico pairplot com as variáveis com maior correlação
colunas_selecionadas = ['mag', 'nst', 'depth', 'horizontalError', 'dmin', 'rms']
sns.pairplot(eq[colunas_selecionadas])
plt.figure()


#Normalização para a variável magnitude
scaler=MinMaxScaler()
coluna = eq['mag']
scaler.fit(coluna.values.reshape(-1, 1))
coluna_normalizada = scaler.transform(coluna.values.reshape(-1, 1))
eq_normalizado = pd.DataFrame(coluna_normalizada, columns=['mag'])
eq_normalizado.hist()
plt.figure()


#Padronização para a variável gap
scaler=StandardScaler()
coluna = eq['gap']
scaler.fit(coluna.values.reshape(-1,1))
coluna_padronizada = scaler.transform(coluna.values.reshape(-1,1))
eq_padronizado = pd.DataFrame(coluna_padronizada, columns=['gap'])
eq_padronizado.hist()
plt.figure()

#Top 10 de sismos ordenados por magnitude
print("\n TOP 10 Earthquakes")
top10 = ['mag', 'place', 'time']
top = eq.sort_values('mag', ascending=False)[top10].head(10).reset_index(drop=True)
top.index = top.index + 1
print(top)


#Mapa mundo com a localização dos sismos
#Função criada com o apoio do ChatGPT
m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
longitudes = eq["longitude"].tolist()
latitudes = eq["latitude"].tolist()
x,y = m(longitudes,latitudes)
fig = plt.figure(figsize=(12,10))
plt.title("All affected areas")
m.plot(x, y, "o", markersize = 2, color = 'blue')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
m.drawmapboundary()
m.drawcountries()
plt.figure()


#Sismos por ano
eq["year"] = eq["time"].apply(lambda x: x[:4])
sns.displot(eq["year"],kde=True)
plt.xlabel("year")
plt.ylabel("Count")
plt.figure()


eq["month"] = eq["time"].apply(lambda x: x[5:7])
sns.displot(eq["month"])
plt.xlabel("month")
plt.ylabel("Count")
plt.figure()


#Sismos por país
eq["country"] = eq["place"].apply(lambda x: x.split(", ")[-1])
sns.displot(eq["country"])
plt.xlabel("country")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.figure()


#Função para criar um displot que apresenta o número de sismos por país
#Devido ao elevado número de dados está limitado aos 100 primeiros
#Função criada com o apoio do ChatGPT

geolocator = Nominatim(user_agent="my_app_name")
eq['country'] = ''
for i, row in eq.head(100).iterrows():
    lat = row['latitude']
    lon = row['longitude']
    location = geolocator.reverse(f"{lat},{lon}", language='en')
    if location is not None:
        print("\n",location.raw )
        country_name = location.raw['address'].get('country', 'N/A')
        eq.at[i, 'country'] = country_name

sns.displot(eq.head(100)['country'])
plt.xticks(rotation=90)
plt.show()

#regressors; rmse e mae

# Linear regression
# Lasso & Ridge
# SVM regressor
# KNN regressor
# Decision Trees
# Random Forest
# Logistic regression
# boosting

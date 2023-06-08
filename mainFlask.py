import numpy as np
import pandas as pd
from flask import Flask,request, render_template
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

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

app = Flask ( __name__ )

@app.route ("/")
def hello_world ():
    return render_template('index.html')

@app.route('/', methods =['GET'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    #Ciclos while para evitar o erro por falta de algum dos dados
    latitude = request.form.get('latitude')
    while (latitude==''):
        latitude = request.form.get('latitude')
        output = 'Insira dados para o campo da latitude!'
        return render_template('index.html', prediction_text=output)
    longitude = request.form.get('longitude')
    while (longitude==''):
        longitude = request.form.get('longitude')
        output = 'Insira dados para o campo da longitude!'
        return render_template('index.html', prediction_text=output)
    prof = request.form.get('profundidade')
    while (prof==''):
        prof = request.form.get('profundidade')
        output = 'Insira dados para o campo da profundidade!'
        return render_template('index.html', prediction_text=output)
    nst = request.form.get('nst')
    while (nst==''):
        nst = request.form.get('nst')
        output = 'Insira dados para o campo nst!\n(Número de estações a detetar o sismo)'
        return render_template('index.html', prediction_text=output)
    #Valores inseridos pelo utilizador para a previsão do sismo
    final_features = np.array([[float(latitude), float(longitude), float(prof), float(nst)]])

    dfInput = eq.drop(
        columns=['time', 'magType', 'gap', 'dmin', 'rms', 'net', 'id', 'updated', 'place', 'type', 'horizontalError',
                 'depthError', 'magError', 'magNst', 'status', 'locationSource', 'magSource', 'mag'])

    X_train, X_test, y_train, y_test = train_test_split(dfInput.values, eq['mag'], test_size=0.3,
                                                        random_state=109)  # 70% training and 30% test
    clfKNN = KNeighborsRegressor(n_neighbors=8)
    clfKNN.fit(X_train, y_train)

    # Resultado da previsão
    output = clfKNN.predict(final_features)
    #Retirar os parenteses retos
    output = np.squeeze(output)

    #return do valor da previsão
    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run()
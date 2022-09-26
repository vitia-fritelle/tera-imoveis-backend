'''Application for prediction price of real state'''
import os
import pickle
import traceback

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from waitress import serve

from class_previsao import Precificacao

load_dotenv()
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY')
CORS(app)
objeto = Precificacao()

# Inicialização dos arquivos auxiliares
with open('./data/model_price.pkl', 'rb') as arquivo:
    modelo = pickle.load(arquivo)
bairro_area_conversor = pd.read_csv('./data/bairro_por_area.csv')
bairro_preco_conversor = pd.read_csv('./data/Bairro_valor_metro2.csv')

class InputModel:
    '''Class for transformation of input data'''
    def __init__(
        self,
        rooms: int,
        bathrooms: int,
        garage: int,
        area: float,
        category: str,
        neighborhood: str):
        self.bathrooms = bathrooms
        self.garage = garage
        self.rooms_by_area = rooms/area
        self.log_area = np.log1p(area)
        self.bairro_por_area = float(
            bairro_area_conversor[
                (bairro_area_conversor['Bairro'] == neighborhood)
                & (bairro_area_conversor['Categoria'] == category)
            ]['bairro_por_area']
        )
        self.preco_por_area = float(
            bairro_preco_conversor[
                (bairro_preco_conversor['Bairro'] == neighborhood)
            ]['preco_area']
        )
        self.neighborhood = neighborhood

    def set_predict_model(self):
        '''Return a DataFrame from the attributes'''
        return pd.DataFrame({
            'Banheiros':[self.bathrooms],
            'Vagas':[self.garage],
            'Bairro':[self.neighborhood],
            'preco_por_metro':[self.preco_por_area],
            'QuartosporAreaConstruida':[self.rooms_by_area],
            'LogAreaConstruida':[self.log_area],
            'bairro_por_area':[self.bairro_por_area]
        })

@app.route('/api', methods=['POST'])
def index():
    '''POST /api route to get the price'''
    if request.method == 'POST':
        try:
            rooms = int(request.json['quartos'])
            bathrooms = int(request.json['banheiros'])
            garage = int(request.json['vagas'])
            area = float(request.json['area'])
            category = request.json['categoria']
            neighborhood = request.json['bairro']
            inputs = InputModel(
                rooms,
                bathrooms,
                garage,
                area,
                category,
                neighborhood
            )
            transformado = objeto.data_preparation(inputs.set_predict_model())
            preco = objeto.get_prediction(transformado)
            return jsonify(data={"price": round(preco[0],2)}), 200
        except TypeError as error:
            return f"{error}\nData in wrong format", 422
        except Exception as error:
            traceback.print_exc()
            return f"{error}", 400

if __name__ == "__main__":
    serve(
        app,
        host='0.0.0.0',
        port=os.environ.get('PORT') if os.environ.get('PORT') else 5000
    )

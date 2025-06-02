from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

class request_body(BaseModel):
    tempo_de_experiencia: int
    numero_de_vendas: int
    fator_sazonal: int

modelo_poly = joblib.load('./modelo_receita.pkl')

@app.post('/predict')
def predict(data: request_body):

    input_features = {
        'tempo_de_experiencia': data.tempo_de_experiencia,
        'numero_de_vendas': data.numero_de_vendas,
        'fator_sazonal': data.fator_sazonal
    }

    pred_df = pd.DataFrame(input_features, index=[1])

    y_pred = modelo_poly.predict(pred_df)[0].astype(float)

    return {'receita_em_reais': y_pred.tolist()}
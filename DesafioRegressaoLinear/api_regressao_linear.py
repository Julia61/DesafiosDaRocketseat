from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn 
import joblib 

app = FastAPI()

class RequestBody(BaseModel):
    tempo_irrigacao: float

regressao_linear = joblib.load('./regressaoLinear.pkl')

@app.post('/predict')
def predict(data: RequestBody):
    input_feature = [[data.tempo_irrigacao]]
    y_pred = int(regressao_linear.predict(input_feature)[0])
    return {'Área irrigada': y_pred}

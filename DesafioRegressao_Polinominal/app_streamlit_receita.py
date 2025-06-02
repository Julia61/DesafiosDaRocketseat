import streamlit as st
import json
import requests

st.title("Modelo de Predição de receita")

st.write("Quanto tempo experiência: ")
tempo_de_experiencia = st.slider("Experiência", min_value=1, max_value=120, value=5, step=1)

st.write("Qual o número de vendas: ")
numero_de_vendas = st.slider("Vêndas", min_value=1, max_value=100, value=10, step=1)

st.write("Qual o fator Sazonal: ")
fator_sazonal = st.slider("Fator Sazonal", min_value=1, max_value=10, value=5, step=1)

input_features = {'tempo_de_experiencia': tempo_de_experiencia,
                  'numero_de_vendas': numero_de_vendas, 
                  'fator_sazonal': fator_sazonal}

if st.button('Estimar Receita'):
    res = requests.post(url="http://127.0.0.1:8000/predict", data=json.dumps(input_features))
    res_json = json.loads(res.text)
    receita_em_reais = round(res_json['receita_em_reais'], 2)
    st.subheader(f'A receita estimada é de R$ {receita_em_reais}')
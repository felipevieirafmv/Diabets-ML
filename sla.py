import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Função para fazer previsões com encoders e scaler já treinados
def fazer_previsao(df, model_path="random_forest_model.joblib", label_encoders=None, scaler=None, selected_columns=None):
    # Carregar o modelo
    model = load(model_path)
    
    # Se o dataframe contiver variáveis categóricas, aplicar os encoders
    if label_encoders:
        for column in selected_columns:  # Usar as colunas selecionadas
            if column in df.select_dtypes(include="object").columns:
                if column in label_encoders:
                    le = label_encoders[column]
                    df[column] = le.transform(df[column])  # Aplica a transformação

    # Se o scaler foi fornecido, escalar as features
    if scaler:
        df_scaled = scaler.transform(df[selected_columns])  # Usar o scaler treinado
    else:
        df_scaled = df[selected_columns]  # Caso não tenha um scaler, usa os dados sem escalonar
    
    # Realizar as previsões
    predictions = model.predict(df_scaled)
    
    return predictions

# Exemplo de como usar:
# 1. Carregar o dataframe com os dados que você quer prever
df_novo = pd.read_csv("novo_dataset.csv")

# 2. Prever com o modelo, usando seus encoders e scaler já treinados
# Certifique-se de que você já tem os encoders e scaler treinados e carregados:
# label_encoders = { ... }  # Dicionário com seus LabelEncoders
# scaler = StandardScaler()  # Seu scaler treinado

# 3. Prever com o modelo
predicoes = fazer_previsao(df_novo, model_path="random_forest_model.joblib", 
                            label_encoders=label_encoders, scaler=scaler, selected_columns=selected_columns)

print(predicoes)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Carregar o dataset
df = pd.read_csv("diabetes_dataset00.csv")

# Selecionar as colunas relevantes, excluindo 'Age' e 'BMI'
selected_columns = [
    "Family History",
    "Physical Activity",
    "Dietary Habits",
    "Socioeconomic Factors",
    "Smoking Status",
    "Alcohol Consumption",
    "Pregnancy History",
    "Steroid Use History",
    "Genetic Testing",
    "Liver Function Tests"
]

X = df[selected_columns]
y = df["Target"]

# Aplicando o LabelEncoder nas colunas categóricas
label_encoder = LabelEncoder()

# Lista de colunas categóricas
categorical_columns = X.select_dtypes(include=['object']).columns

# Transformar as colunas categóricas
for col in categorical_columns:
    X.loc[:, col] = label_encoder.fit_transform(X[col])

# Definir explicitamente as colunas numéricas
numerical_columns = [
    "Family History", 
    "Physical Activity", 
    "Dietary Habits", 
    "Socioeconomic Factors", 
    "Smoking Status", 
    "Alcohol Consumption", 
    "Pregnancy History", 
    "Steroid Use History", 
    "Genetic Testing", 
    "Liver Function Tests"
]

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pré-processar os dados numéricos
scaler = QuantileTransformer()

# Escalonar as variáveis numéricas
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Criar e treinar o modelo
random_forest_model = RandomForestClassifier(
    n_estimators=500,        # Mais árvores
    max_depth=10,            # Limitar a profundidade das árvores
    max_features='sqrt',     # Usar a raiz quadrada das features
    criterion='gini',        # Usar o índice de Gini
    class_weight='balanced', # Lidar com classes desbalanceadas
    random_state=42
)

# Treinar o modelo
random_forest_model.fit(X_train, y_train)

# Avaliar a acurácia
accuracy = random_forest_model.score(X_test, y_test)
print(f"Acurácia do RandomForest: {accuracy:.2f}")

# Avaliar as importâncias das features
feature_importances = random_forest_model.feature_importances_
print("Importâncias das features:", feature_importances)

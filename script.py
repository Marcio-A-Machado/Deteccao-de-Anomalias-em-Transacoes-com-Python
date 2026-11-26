import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregamento dos Dados
print("Carregando os dados...")
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)
print("Dados carregados. 5 primeiras linhas:")
print(df.head())
print("Distribuição da classe (Fraude vs. Não Fraude):")
print(df["Class"].value_counts(normalize=True))

 2. Engenharia de Features
print("Realizando engenharia de features...")
# Aplica a transformação logarítmica na feature 'Amount' para reduzir assimetria
df["Amount_log"] = np.log1p(df["Amount"])
# Normaliza a feature 'Amount' usando StandardScaler
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))

# Separa features (X) e target (y)
X = df.drop("Class", axis=1)
y = df["Class"]
print("Features 'Amount_log' e 'Amount_scaled' criadas.")

# 3. Divisão Inicial dos Dados (para Otimização de Hiperparâmetros)
# Esta divisão usa os dados originais e desbalanceados para buscar os melhores hiperparâmetros
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)
print("Dados divididos inicialmente para otimização de hiperparâmetros (dados originais).")

# 4. Otimização de Hiperparâmetros para XGBoost (com RandomizedSearchCV)
print("Executando RandomizedSearchCV para otimização de hiperparâmetros do XGBoost...")
param_dist = {
    'n_estimators': randint(50, 200),        # Número de árvores
    'max_depth': randint(3, 10),             # Profundidade máxima de cada árvore
    'learning_rate': uniform(0.01, 0.3),     # Taxa de aprendizado
    'subsample': uniform(0.6, 0.4),          # Fração de amostras usadas para treinar cada árvore
    'colsample_bytree': uniform(0.6, 0.4),   # Fração de features usadas para treinar cada árvore
    'gamma': uniform(0, 0.5)                 # Redução mínima de perda necessária para fazer uma nova partição em um nó folha da árvore.
}

random_search = RandomizedSearchCV(
    XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=50, # Número de configurações de parâmetros que serão amostradas. Aumente para uma busca mais completa.
    cv=3,
    scoring='recall', # Otimiza para recall, importante para detecção de fraudes
    random_state=42,
    n_jobs=-1, # Usa todos os cores disponíveis
    verbose=0
)

random_search.fit(X_train, y_train)

best_params = random_search.best_params_
print("Melhores hiperparâmetros encontrados:", best_params)
print("Melhor recall score a partir do RandomizedSearchCV:", random_search.best_score_)


# 5. Balanceamento dos Dados Usando SMOTE
print("Aplicando SMOTE para balancear o dataset...")
smote = SMOTE(random_state=42)
# Aplica SMOTE no conjunto completo X e y para criar um dataset balanceado
X_res, y_res = smote.fit_resample(X, y)
print(f"Formato do dataset original: {X.shape}, {y.shape}")
print(f"Formato do dataset reamostrado por SMOTE: {X_res.shape}, {y_res.shape}")
print("Distribuição da classe após SMOTE:")
print(y_res.value_counts(normalize=True))

# 6. Divisão dos Dados Balanceados (SMOTE)
# Divide o dataset balanceado pelo SMOTE em conjuntos de treino e teste
X_res_train, X_res_test, y_res_train, y_res_test = train_test_split(
    X_res, y_res, test_size=0.3, random_state=42, stratify=y_res
)
print("Dados balanceados por SMOTE divididos em conjuntos de treino e teste.")

# 7. Treinamento do Modelo XGBoost com Dados Balanceados e Hiperparâmetros Otimizados
print("Treinando o modelo XGBoost nos dados balanceados por SMOTE com hiperparâmetros otimizados...")
xgb_smote = XGBClassifier(
    objective='binary:logistic', # Função objetivo para classificação binária
    eval_metric='logloss',       # Métrica de avaliação
    use_label_encoder=False,     # Suprime o aviso de depreciação
    random_state=42,
    n_jobs=-1,                   # Usa todos os cores para treinamento
    **best_params                # Desempacota os melhores parâmetros encontrados
)
# Treina o modelo nos dados de treino balanceados
xgb_smote.fit(X_res_train, y_res_train)
# Realiza previsões no conjunto de teste balanceado
y_pred_smote = xgb_smote.predict(X_res_test)

# 8. Avaliação do Modelo (XGBoost com SMOTE)
print("
--- Avaliação do Modelo (XGBoost com SMOTE) ---")
print("Relatório de Classificação:")
print(classification_report(y_res_test, y_pred_smote))

# Gera a matriz de confusão
cm_smote = confusion_matrix(y_res_test, y_pred_smote)
# Calcula o recall para a classe de fraude (1) manualmente a partir da matriz
true_positives_smote = cm_smote[1, 1]
false_negatives_smote = cm_smote[1, 0]
recall_score_smote = true_positives_smote / (true_positives_smote + false_negatives_smote)
print(f"
Recall para a classe de fraude (1) com SMOTE: {recall_score_smote:.4f}")

# Plota a matriz de confusão
plt.figure(figsize=(6, 5))
sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Não Fraude (0)', 'Fraude (1)'],
            yticklabels=['Não Fraude (0)', 'Fraude (1)'])
plt.xlabel('Valor Predito')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusão para o Modelo XGBoost (SMOTE)')
plt.show()

print("
Execução do script concluída. O gráfico da Matriz de Confusão será exibido.")

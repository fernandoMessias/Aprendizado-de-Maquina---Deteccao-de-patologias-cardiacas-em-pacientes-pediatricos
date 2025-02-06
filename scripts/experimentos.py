# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Aprendizado de Maquina
# Prof. Tiago A. Almeida
#
#
# Nome: Fernando Messias da Silva - RA: 489450
# Nome: Josie de Assis Francisco Henriques do Nascimento - RA: 840214
# ################################################################
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import tree

from scripts.analise_exploratoria import train_data, test_data, rhp_data, columns_to_drop
from scripts.preprocessamento import rhp_data_processed_normalized, new_columns_to_drop

# Arquivo com todas as funcoes e codigos referentes aos experimentos


# Cria o arquivo de treinamento
train_dataset = train_data.merge(rhp_data_processed_normalized, on="Id", how="left")

train_dataset = train_dataset.drop(columns=["CLASSE_x"]).rename(columns={"CLASSE_y": "CLASSE"})

# Criar nova feature: Relação Peso/Altura
#train_dataset["Peso_Altura_Ratio"] = train_dataset["Peso"] / train_dataset["Altura"]
# Criar Pressão de Pulso Arterial (PPA numérica)
train_dataset["PPA_num"] = train_dataset["PA SISTOLICA"] - train_dataset["PA DIASTOLICA"]

train_dataset = train_dataset.drop(columns=new_columns_to_drop, errors='ignore')

# Remove NaN from train dataset
train_dataset['Peso'].fillna(train_dataset['Peso'].median(), inplace=True)
train_dataset['Altura'].fillna(train_dataset['Altura'].median(), inplace=True)
train_dataset['IMC'].fillna(train_dataset['IMC'].median(), inplace=True)
train_dataset['IDADE'].fillna(train_dataset['IDADE'].median(), inplace=True)
#train_dataset['PULSOS'].fillna(train_dataset['PULSOS'].mode()[0], inplace=True)
train_dataset['PA SISTOLICA'].fillna(train_dataset['PA SISTOLICA'].median(), inplace=True)
train_dataset['PA DIASTOLICA'].fillna(train_dataset['PA DIASTOLICA'].median(), inplace=True)
#train_dataset['PPA'].fillna(train_dataset['PPA'].mode()[0], inplace=True)
#train_dataset['B2'].fillna(train_dataset['B2'].mode()[0], inplace=True)
train_dataset['FC'].fillna(train_dataset['FC'].median(), inplace=True)
# train_dataset['SEXO'].fillna(train_dataset['SEXO'].mode()[0], inplace=True)
# train_dataset['MOTIVO1'].fillna(train_dataset['MOTIVO1'].mode()[0], inplace=True)
# train_dataset['MOTIVO2'].fillna(train_dataset['MOTIVO2'].median(), inplace=True)
# train_dataset['SOPRO'].fillna(train_dataset['SOPRO'].mode()[0], inplace=True)


train_dataset['SEXO_Feminino'].fillna(train_dataset['SEXO_Feminino'].mode()[0], inplace=True)
train_dataset['SEXO_Indeterminado'].fillna(train_dataset['SEXO_Indeterminado'].mode()[0], inplace=True)
train_dataset['SEXO_M'].fillna(train_dataset['SEXO_M'].mode()[0], inplace=True)
train_dataset['SEXO_Masculino'].fillna(train_dataset['SEXO_Masculino'].mode()[0], inplace=True)
train_dataset['SEXO_masculino'].fillna(train_dataset['SEXO_masculino'].mode()[0], inplace=True)
#train_dataset['PULSOS_Amplos'].fillna(train_dataset['PULSOS_Amplos'].mode()[0], inplace=True)
train_dataset['PULSOS_Diminuídos '].fillna(train_dataset['PULSOS_Diminuídos '].mode()[0], inplace=True)
train_dataset['PULSOS_Femorais diminuidos'].fillna(train_dataset['PULSOS_Femorais diminuidos'].mode()[0], inplace=True)
#train_dataset['PULSOS_NORMAIS'].fillna(train_dataset['PULSOS_NORMAIS'].mode()[0], inplace=True)
train_dataset['PULSOS_Normais'].fillna(train_dataset['PULSOS_Normais'].mode()[0], inplace=True)
train_dataset['PULSOS_Outro'].fillna(train_dataset['PULSOS_Outro'].mode()[0], inplace=True)
train_dataset['PPA_HAS-1 PAS'].fillna(train_dataset['PPA_HAS-1 PAS'].mode()[0], inplace=True)
train_dataset['PPA_HAS-2 PAD'].fillna(train_dataset['PPA_HAS-2 PAD'].mode()[0], inplace=True)
train_dataset['PPA_HAS-2 PAS'].fillna(train_dataset['PPA_HAS-2 PAS'].mode()[0], inplace=True)
train_dataset['PPA_Normal'].fillna(train_dataset['PPA_Normal'].mode()[0], inplace=True)
train_dataset['PPA_Não Calculado'].fillna(train_dataset['PPA_Não Calculado'].mode()[0], inplace=True)
train_dataset['PPA_Pre-Hipertensão PAD'].fillna(train_dataset['PPA_Pre-Hipertensão PAD'].mode()[0], inplace=True)
train_dataset['PPA_Pre-Hipertensão PAS'].fillna(train_dataset['PPA_Pre-Hipertensão PAS'].mode()[0], inplace=True)
train_dataset['B2_Hiperfonética'].fillna(train_dataset['B2_Hiperfonética'].mode()[0], inplace=True)
train_dataset['B2_Normal'].fillna(train_dataset['B2_Normal'].mode()[0], inplace=True)
train_dataset['B2_Outro'].fillna(train_dataset['B2_Outro'].mode()[0], inplace=True)
train_dataset['B2_Única'].fillna(train_dataset['B2_Única'].mode()[0], inplace=True)
#train_dataset['SOPRO_Sistolico e diastólico'].fillna(train_dataset['SOPRO_Sistolico e diastólico'].mode()[0], inplace=True)
train_dataset['SOPRO_Sistólico'].fillna(train_dataset['SOPRO_Sistólico'].mode()[0], inplace=True)
train_dataset['SOPRO_ausente'].fillna(train_dataset['SOPRO_ausente'].mode()[0], inplace=True)
train_dataset['SOPRO_contínuo'].fillna(train_dataset['SOPRO_contínuo'].mode()[0], inplace=True)
train_dataset['SOPRO_diastólico'].fillna(train_dataset['SOPRO_diastólico'].mode()[0], inplace=True)
train_dataset['SOPRO_sistólico'].fillna(train_dataset['SOPRO_sistólico'].mode()[0], inplace=True)
train_dataset['MOTIVO1_2 - Check-up'].fillna(train_dataset['MOTIVO1_2 - Check-up'].mode()[0], inplace=True)
train_dataset['MOTIVO1_5 - Parecer cardiológico'].fillna(train_dataset['MOTIVO1_5 - Parecer cardiológico'].mode()[0], inplace=True)
train_dataset['MOTIVO1_6 - Suspeita de cardiopatia'].fillna(train_dataset['MOTIVO1_6 - Suspeita de cardiopatia'].mode()[0], inplace=True)
train_dataset['MOTIVO1_7 - Outro'].fillna(train_dataset['MOTIVO1_7 - Outro'].mode()[0], inplace=True)
train_dataset['MOTIVO2_1 - Cardiopatia congenica'].fillna(train_dataset['MOTIVO2_1 - Cardiopatia congenica'].mode()[0], inplace=True)
train_dataset['MOTIVO2_5 - Atividade física'].fillna(train_dataset['MOTIVO2_5 - Atividade física'].mode()[0], inplace=True)
train_dataset['MOTIVO2_5 - Cirurgia'].fillna(train_dataset['MOTIVO2_5 - Cirurgia'].mode()[0], inplace=True)
train_dataset['MOTIVO2_5 - Uso de cisaprida'].fillna(train_dataset['MOTIVO2_5 - Uso de cisaprida'].mode()[0], inplace=True)
train_dataset['MOTIVO2_6 - Alterações de pulso/perfusão'].fillna(train_dataset['MOTIVO2_6 - Alterações de pulso/perfusão'].mode()[0], inplace=True)
train_dataset['MOTIVO2_6 - Cansaço'].fillna(train_dataset['MOTIVO2_6 - Cansaço'].mode()[0], inplace=True)
train_dataset['MOTIVO2_6 - Cardiopatia na familia'].fillna(train_dataset['MOTIVO2_6 - Cardiopatia na familia'].mode()[0], inplace=True)
train_dataset['MOTIVO2_6 - Cianose'].fillna(train_dataset['MOTIVO2_6 - Cianose'].mode()[0], inplace=True)
#train_dataset['MOTIVO2_6 - Cianose e dispnéia'].fillna(train_dataset['MOTIVO2_6 - Cianose e dispnéia'].mode()[0], inplace=True)
train_dataset['MOTIVO2_6 - Dispnéia'].fillna(train_dataset['MOTIVO2_6 - Dispnéia'].mode()[0], inplace=True)
train_dataset['MOTIVO2_6 - Dor precordial'].fillna(train_dataset['MOTIVO2_6 - Dor precordial'].mode()[0], inplace=True)
train_dataset['MOTIVO2_6 - HAS/dislipidemia/obesidade'].fillna(train_dataset['MOTIVO2_6 - HAS/dislipidemia/obesidade'].mode()[0], inplace=True)
train_dataset['MOTIVO2_6 - Palpitação/taquicardia/arritmia'].fillna(train_dataset['MOTIVO2_6 - Palpitação/taquicardia/arritmia'].mode()[0], inplace=True)
train_dataset['MOTIVO2_6 - Sopro'].fillna(train_dataset['MOTIVO2_6 - Sopro'].mode()[0], inplace=True)
train_dataset['MOTIVO2_Outro'].fillna(train_dataset['MOTIVO2_Outro'].mode()[0], inplace=True)
#train_dataset['Peso_Altura_Ratio'].fillna(train_dataset['Peso_Altura_Ratio'].median(), inplace=True)
train_dataset['PPA_num'].fillna(train_dataset['PPA_num'].mode()[0], inplace=True)

train_dataset['CLASSE'].fillna(train_dataset['CLASSE'].mode()[0], inplace=True)


# Aplicar a padronização (Z-score) aos atributos numéricos
numerical_columns = ["IDADE", "Peso", "Altura", "IMC", "PA SISTOLICA", "PA DIASTOLICA", "FC", "PPA_num"]

# Calcular a média e desvio padrão de cada atributo
means = train_dataset[numerical_columns].mean()
stds = train_dataset[numerical_columns].std()

# Aplicar a padronização
train_dataset[numerical_columns] = (train_dataset[numerical_columns] - means) / stds

# Salvar dados pre-processados
train_dataset.to_csv('dados/train_dataset.csv', index=False)

### Começar a aplicar os algoritmos para fazer o treinamento
X = train_dataset.drop(['CLASSE', 'Id'], axis=1)
X.fillna(0)
y = train_dataset['CLASSE']


# Dividir em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Aplicar SMOTE para balancear as classes
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train.astype(int), y_train.astype(int))

# Dicionário para armazenar resultados
results_auc_score = {}

# Treinar modelo
# k-NN
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train_balanced, y_train_balanced)
y_pred_knn = knn.predict_proba(X_val)[:, 1]
results_auc_score['k-NN'] = roc_auc_score(y_val, y_pred_knn)

# Treinar modelo
# RandomForest
randomForest = RandomForestClassifier(n_estimators=250, max_depth=20, min_samples_split=5, random_state=42)
randomForest.fit(X_train_balanced, y_train_balanced)
y_pred_random_forest = randomForest.predict_proba(X_val)[:, 1]
results_auc_score['Radom Forest'] = roc_auc_score(y_val, y_pred_random_forest)

# Treinar modelo
# Logistic Regression
logisticRegression = LogisticRegression(max_iter=250)
logisticRegression.fit(X_train_balanced, y_train_balanced)
y_pred_logistic_regression = logisticRegression.predict_proba(X_val)[:, 1]
results_auc_score['Logistic Regression'] = roc_auc_score(y_val, y_pred_logistic_regression)

# Naive Bayes
nb = GaussianNB()
nb_calibrated = CalibratedClassifierCV(nb, method="sigmoid")
nb_calibrated.fit(X_train_balanced, y_train_balanced)
y_pred_nb = nb_calibrated.predict_proba(X_val)[:, 1]
results_auc_score['Naive Bayes'] = roc_auc_score(y_val, y_pred_nb)

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=4, random_state=42)
gb.fit(X_train_balanced, y_train_balanced)
y_pred_gb = gb.predict_proba(X_val)[:, 1]
results_auc_score['Gradient Boosting'] = roc_auc_score(y_val, y_pred_gb)

# SVM
svm = SVC(probability=True, random_state=42)
svm.fit(X_train_balanced, y_train_balanced)
y_pred_svm = svm.predict_proba(X_val)[:, 1]
results_auc_score['SVM'] = roc_auc_score(y_val, y_pred_svm)

# Redes Neurais
mlp = MLPClassifier(random_state=42, max_iter=250)
mlp.fit(X_train_balanced, y_train_balanced)
y_pred_mlp = mlp.predict_proba(X_val)[:, 1]
results_auc_score['Redes Neurais'] = roc_auc_score(y_val, y_pred_mlp)

# Definir o grid de hiperparâmetros
param_grid = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "criterion": ["gini", "entropy"],
    "max_features": ["sqrt", "log2", None],
    "class_weight": [None, "balanced"]
}

# Decision Tree
dt = tree.DecisionTreeClassifier(criterion='gini', max_depth=3)
dt.fit(X_train_balanced, y_train_balanced)
y_pred_dt = dt.predict_proba(X_val)[:, 1]
results_auc_score['Decision Tree'] = roc_auc_score(y_val, y_pred_dt)

# Aplicar GridSearchCV
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train_balanced, y_train_balanced)
y_pred_gs = grid_search.predict_proba(X_val)[:, 1]
results_auc_score['Grid Search'] = roc_auc_score(y_val, y_pred_gs)


print(results_auc_score)


# Filtrar os dados de teste baseados nos IDs do test.csv
test_filtered = test_data.merge(rhp_data, on="Id", how="left")

# Remove columns dropped during training phase
test_filtered = test_filtered.drop(columns=columns_to_drop, errors='ignore')

# Selecionar colunas categóricas para conversão
categorical_cols = ["SEXO", "PULSOS", "PPA", "B2", "SOPRO", "MOTIVO1", "MOTIVO2"]

# Aplicar One-Hot Encoding
test_filtered = pd.get_dummies(test_filtered, columns=categorical_cols, drop_first=True)

# Exibir as primeiras linhas após a codificação
test_filtered.head()


# Criar nova feature: Relação Peso/Altura
#test_filtered["Peso_Altura_Ratio"] = test_filtered["Peso"] / test_filtered["Altura"]

# Criar Pressão de Pulso Arterial (PPA numérica)
test_filtered["PPA_num"] = test_filtered["PA SISTOLICA"] - test_filtered["PA DIASTOLICA"]



test_filtered['Peso'] = pd.to_numeric(test_filtered['Peso'], errors='coerce')
test_filtered['Altura'] = pd.to_numeric(test_filtered['Altura'], errors='coerce')
test_filtered['IMC'] = pd.to_numeric(test_filtered['IMC'], errors='coerce')
test_filtered['IDADE'] = pd.to_numeric(test_filtered['IDADE'], errors='coerce')
#test_filtered['PULSOS'] = pd.to_numeric(test_filtered['PULSOS'], errors='coerce')
test_filtered['PA SISTOLICA'] = pd.to_numeric(test_filtered['PA SISTOLICA'], errors='coerce')
test_filtered['PA DIASTOLICA'] = pd.to_numeric(test_filtered['PA DIASTOLICA'], errors='coerce')
#test_filtered['PPA'] = pd.to_numeric(test_filtered['PPA'], errors='coerce')
#test_filtered['B2'] = pd.to_numeric(test_filtered['B2'], errors='coerce')
test_filtered['FC'] = pd.to_numeric(test_filtered['FC'], errors='coerce')
#test_filtered['SEXO'] = pd.to_numeric(test_filtered['SEXO'], errors='coerce')
#test_filtered['MOTIVO1'] = pd.to_numeric(test_filtered['MOTIVO1'], errors='coerce')
#test_filtered['MOTIVO2'] = pd.to_numeric(test_filtered['MOTIVO2'], errors='coerce')
#test_filtered['SOPRO'] = pd.to_numeric(test_filtered['SOPRO'], errors='coerce')

test_filtered['SEXO_Feminino'] = pd.to_numeric(test_filtered['SEXO_Feminino'], errors='coerce')
test_filtered['SEXO_Indeterminado'] = pd.to_numeric(test_filtered['SEXO_Indeterminado'], errors='coerce')
test_filtered['SEXO_M'] = pd.to_numeric(test_filtered['SEXO_M'], errors='coerce')
test_filtered['SEXO_Masculino'] = pd.to_numeric(test_filtered['SEXO_Masculino'], errors='coerce')
test_filtered['SEXO_masculino'] = pd.to_numeric(test_filtered['SEXO_masculino'], errors='coerce')
#test_filtered['PULSOS_Amplos'] = pd.to_numeric(test_filtered['PULSOS_Amplos'], errors='coerce')
test_filtered['PULSOS_Diminuídos '] = pd.to_numeric(test_filtered['PULSOS_Diminuídos '], errors='coerce')
test_filtered['PULSOS_Femorais diminuidos'] = pd.to_numeric(test_filtered['PULSOS_Femorais diminuidos'], errors='coerce')
#test_filtered['PULSOS_NORMAIS'] = pd.to_numeric(test_filtered['PULSOS_NORMAIS'], errors='coerce')
test_filtered['PULSOS_Normais'] = pd.to_numeric(test_filtered['PULSOS_Normais'], errors='coerce')
test_filtered['PULSOS_Outro'] = pd.to_numeric(test_filtered['PULSOS_Outro'], errors='coerce')
test_filtered['PPA_HAS-1 PAS'] = pd.to_numeric(test_filtered['PPA_HAS-1 PAS'], errors='coerce')
test_filtered['PPA_HAS-2 PAD'] = pd.to_numeric(test_filtered['PPA_HAS-2 PAD'], errors='coerce')
test_filtered['PPA_HAS-2 PAS'] = pd.to_numeric(test_filtered['PPA_HAS-2 PAS'], errors='coerce')
test_filtered['PPA_Normal'] = pd.to_numeric(test_filtered['PPA_Normal'], errors='coerce')
test_filtered['PPA_Não Calculado'] = pd.to_numeric(test_filtered['PPA_Não Calculado'], errors='coerce')
test_filtered['PPA_Pre-Hipertensão PAD'] = pd.to_numeric(test_filtered['PPA_Pre-Hipertensão PAD'], errors='coerce')
test_filtered['PPA_Pre-Hipertensão PAS'] = pd.to_numeric(test_filtered['PPA_Pre-Hipertensão PAS'], errors='coerce')
test_filtered['B2_Hiperfonética'] = pd.to_numeric(test_filtered['B2_Hiperfonética'], errors='coerce')
test_filtered['B2_Normal'] = pd.to_numeric(test_filtered['B2_Normal'], errors='coerce')
test_filtered['B2_Outro'] = pd.to_numeric(test_filtered['B2_Outro'], errors='coerce')
test_filtered['B2_Única'] = pd.to_numeric(test_filtered['B2_Única'], errors='coerce')
#test_filtered['SOPRO_Sistolico e diastólico'] = pd.to_numeric(test_filtered['SOPRO_Sistolico e diastólico'], errors='coerce')
test_filtered['SOPRO_Sistólico'] = pd.to_numeric(test_filtered['SOPRO_Sistólico'], errors='coerce')
test_filtered['SOPRO_ausente'] = pd.to_numeric(test_filtered['SOPRO_ausente'], errors='coerce')
test_filtered['SOPRO_contínuo'] = pd.to_numeric(test_filtered['SOPRO_contínuo'], errors='coerce')
test_filtered['SOPRO_diastólico'] = pd.to_numeric(test_filtered['SOPRO_diastólico'], errors='coerce')
test_filtered['SOPRO_sistólico'] = pd.to_numeric(test_filtered['SOPRO_sistólico'], errors='coerce')
test_filtered['MOTIVO1_2 - Check-up'] = pd.to_numeric(test_filtered['MOTIVO1_2 - Check-up'], errors='coerce')
test_filtered['MOTIVO1_5 - Parecer cardiológico'] = pd.to_numeric(test_filtered['MOTIVO1_5 - Parecer cardiológico'], errors='coerce')
test_filtered['MOTIVO1_6 - Suspeita de cardiopatia'] = pd.to_numeric(test_filtered['MOTIVO1_6 - Suspeita de cardiopatia'], errors='coerce')
test_filtered['MOTIVO1_7 - Outro'] = pd.to_numeric(test_filtered['MOTIVO1_7 - Outro'], errors='coerce')
test_filtered['MOTIVO2_1 - Cardiopatia congenica'] = pd.to_numeric(test_filtered['MOTIVO2_1 - Cardiopatia congenica'], errors='coerce')
test_filtered['MOTIVO2_5 - Atividade física'] = pd.to_numeric(test_filtered['MOTIVO2_5 - Atividade física'], errors='coerce')
test_filtered['MOTIVO2_5 - Cirurgia'] = pd.to_numeric(test_filtered['MOTIVO2_5 - Cirurgia'], errors='coerce')
test_filtered['MOTIVO2_5 - Uso de cisaprida'] = pd.to_numeric(test_filtered['MOTIVO2_5 - Uso de cisaprida'], errors='coerce')
test_filtered['MOTIVO2_6 - Alterações de pulso/perfusão'] = pd.to_numeric(test_filtered['MOTIVO2_6 - Alterações de pulso/perfusão'], errors='coerce')
test_filtered['MOTIVO2_6 - Cansaço'] = pd.to_numeric(test_filtered['MOTIVO2_6 - Cansaço'], errors='coerce')
test_filtered['MOTIVO2_6 - Cardiopatia na familia'] = pd.to_numeric(test_filtered['MOTIVO2_6 - Cardiopatia na familia'], errors='coerce')
test_filtered['MOTIVO2_6 - Cianose'] = pd.to_numeric(test_filtered['MOTIVO2_6 - Cianose'], errors='coerce')
#test_filtered['MOTIVO2_6 - Cianose e dispnéia'] = pd.to_numeric(test_filtered['MOTIVO2_6 - Cianose e dispnéia'], errors='coerce')
test_filtered['MOTIVO2_6 - Dispnéia'] = pd.to_numeric(test_filtered['MOTIVO2_6 - Dispnéia'], errors='coerce')
test_filtered['MOTIVO2_6 - Dor precordial'] = pd.to_numeric(test_filtered['MOTIVO2_6 - Dor precordial'], errors='coerce')
test_filtered['MOTIVO2_6 - HAS/dislipidemia/obesidade'] = pd.to_numeric(test_filtered['MOTIVO2_6 - HAS/dislipidemia/obesidade'], errors='coerce')
test_filtered['MOTIVO2_6 - Palpitação/taquicardia/arritmia'] = pd.to_numeric(test_filtered['MOTIVO2_6 - Palpitação/taquicardia/arritmia'], errors='coerce')
test_filtered['MOTIVO2_6 - Sopro'] = pd.to_numeric(test_filtered['MOTIVO2_6 - Sopro'], errors='coerce')
test_filtered['MOTIVO2_Outro'] = pd.to_numeric(test_filtered['MOTIVO2_Outro'], errors='coerce')


test_filtered['Peso'].fillna(test_filtered['Peso'].median(), inplace=True)
test_filtered['Altura'].fillna(test_filtered['Altura'].median(), inplace=True)
test_filtered['IMC'].fillna(test_filtered['IMC'].median(), inplace=True)
test_filtered['IDADE'].fillna(test_filtered['IDADE'].median(), inplace=True)
#test_filtered['PULSOS'].fillna(test_filtered['PULSOS'].mode()[0], inplace=True)
test_filtered['PA SISTOLICA'].fillna(test_filtered['PA SISTOLICA'].median(), inplace=True)
test_filtered['PA DIASTOLICA'].fillna(test_filtered['PA DIASTOLICA'].median(), inplace=True)
#test_filtered['PPA'].fillna(test_filtered['PPA'].mode()[0], inplace=True)
#test_filtered['B2'].fillna(test_filtered['B2'].mode()[0], inplace=True)
test_filtered['FC'].fillna(test_filtered['FC'].median(), inplace=True)
#test_filtered['SEXO'].fillna(test_filtered['SEXO'].mode()[0], inplace=True)
#test_filtered['MOTIVO1'].fillna(test_filtered['MOTIVO1'].mode()[0], inplace=True)
#test_filtered['MOTIVO2'].fillna(test_filtered['MOTIVO2'].mode()[0], inplace=True)
#test_filtered['SOPRO'].fillna(test_filtered['SOPRO'].mode()[0], inplace=True)


test_filtered['SEXO_Feminino'].fillna(test_filtered['SEXO_Feminino'].mode()[0], inplace=True)
test_filtered['SEXO_Indeterminado'].fillna(test_filtered['SEXO_Indeterminado'].mode()[0], inplace=True)
test_filtered['SEXO_M'].fillna(test_filtered['SEXO_M'].mode()[0], inplace=True)
test_filtered['SEXO_Masculino'].fillna(test_filtered['SEXO_Masculino'].mode()[0], inplace=True)
test_filtered['SEXO_masculino'].fillna(test_filtered['SEXO_masculino'].mode()[0], inplace=True)
#test_filtered['PULSOS_Amplos'].fillna(test_filtered['PULSOS_Amplos'].mode()[0], inplace=True)
test_filtered['PULSOS_Diminuídos '].fillna(test_filtered['PULSOS_Diminuídos '].mode()[0], inplace=True)
test_filtered['PULSOS_Femorais diminuidos'].fillna(test_filtered['PULSOS_Femorais diminuidos'].mode()[0], inplace=True)
#test_filtered['PULSOS_NORMAIS'].fillna(test_filtered['PULSOS_NORMAIS'].mode()[0], inplace=True)
test_filtered['PULSOS_Normais'].fillna(test_filtered['PULSOS_Normais'].mode()[0], inplace=True)
test_filtered['PULSOS_Outro'].fillna(test_filtered['PULSOS_Outro'].mode()[0], inplace=True)
test_filtered['PPA_HAS-1 PAS'].fillna(test_filtered['PPA_HAS-1 PAS'].mode()[0], inplace=True)
test_filtered['PPA_HAS-2 PAD'].fillna(test_filtered['PPA_HAS-2 PAD'].mode()[0], inplace=True)
test_filtered['PPA_HAS-2 PAS'].fillna(test_filtered['PPA_HAS-2 PAS'].mode()[0], inplace=True)
test_filtered['PPA_Normal'].fillna(test_filtered['PPA_Normal'].mode()[0], inplace=True)
test_filtered['PPA_Não Calculado'].fillna(test_filtered['PPA_Não Calculado'].mode()[0], inplace=True)
test_filtered['PPA_Pre-Hipertensão PAD'].fillna(test_filtered['PPA_Pre-Hipertensão PAD'].mode()[0], inplace=True)
test_filtered['PPA_Pre-Hipertensão PAS'].fillna(test_filtered['PPA_Pre-Hipertensão PAS'].mode()[0], inplace=True)
test_filtered['B2_Hiperfonética'].fillna(test_filtered['B2_Hiperfonética'].mode()[0], inplace=True)
test_filtered['B2_Normal'].fillna(test_filtered['B2_Normal'].mode()[0], inplace=True)
test_filtered['B2_Outro'].fillna(test_filtered['B2_Outro'].mode()[0], inplace=True)
test_filtered['B2_Única'].fillna(test_filtered['B2_Única'].mode()[0], inplace=True)
#test_filtered['SOPRO_Sistolico e diastólico'].fillna(test_filtered['SOPRO_Sistolico e diastólico'].mode()[0], inplace=True)
test_filtered['SOPRO_Sistólico'].fillna(test_filtered['SOPRO_Sistólico'].mode()[0], inplace=True)
test_filtered['SOPRO_ausente'].fillna(test_filtered['SOPRO_ausente'].mode()[0], inplace=True)
test_filtered['SOPRO_contínuo'].fillna(test_filtered['SOPRO_contínuo'].mode()[0], inplace=True)
test_filtered['SOPRO_diastólico'].fillna(test_filtered['SOPRO_diastólico'].mode()[0], inplace=True)
test_filtered['SOPRO_sistólico'].fillna(test_filtered['SOPRO_sistólico'].mode()[0], inplace=True)
test_filtered['MOTIVO1_2 - Check-up'].fillna(test_filtered['MOTIVO1_2 - Check-up'].mode()[0], inplace=True)
test_filtered['MOTIVO1_5 - Parecer cardiológico'].fillna(test_filtered['MOTIVO1_5 - Parecer cardiológico'].mode()[0], inplace=True)
test_filtered['MOTIVO1_6 - Suspeita de cardiopatia'].fillna(test_filtered['MOTIVO1_6 - Suspeita de cardiopatia'].mode()[0], inplace=True)
test_filtered['MOTIVO1_7 - Outro'].fillna(test_filtered['MOTIVO1_7 - Outro'].mode()[0], inplace=True)
test_filtered['MOTIVO2_1 - Cardiopatia congenica'].fillna(test_filtered['MOTIVO2_1 - Cardiopatia congenica'].mode()[0], inplace=True)
test_filtered['MOTIVO2_5 - Atividade física'].fillna(test_filtered['MOTIVO2_5 - Atividade física'].mode()[0], inplace=True)
test_filtered['MOTIVO2_5 - Cirurgia'].fillna(test_filtered['MOTIVO2_5 - Cirurgia'].mode()[0], inplace=True)
test_filtered['MOTIVO2_5 - Uso de cisaprida'].fillna(test_filtered['MOTIVO2_5 - Uso de cisaprida'].mode()[0], inplace=True)
test_filtered['MOTIVO2_6 - Alterações de pulso/perfusão'].fillna(test_filtered['MOTIVO2_6 - Alterações de pulso/perfusão'].mode()[0], inplace=True)
test_filtered['MOTIVO2_6 - Cansaço'].fillna(test_filtered['MOTIVO2_6 - Cansaço'].mode()[0], inplace=True)
test_filtered['MOTIVO2_6 - Cardiopatia na familia'].fillna(test_filtered['MOTIVO2_6 - Cardiopatia na familia'].mode()[0], inplace=True)
test_filtered['MOTIVO2_6 - Cianose'].fillna(test_filtered['MOTIVO2_6 - Cianose'].mode()[0], inplace=True)
#test_filtered['MOTIVO2_6 - Cianose e dispnéia'].fillna(test_filtered['MOTIVO2_6 - Cianose e dispnéia'].mode()[0], inplace=True)
test_filtered['MOTIVO2_6 - Dispnéia'].fillna(test_filtered['MOTIVO2_6 - Dispnéia'].mode()[0], inplace=True)
test_filtered['MOTIVO2_6 - Dor precordial'].fillna(test_filtered['MOTIVO2_6 - Dor precordial'].mode()[0], inplace=True)
test_filtered['MOTIVO2_6 - HAS/dislipidemia/obesidade'].fillna(test_filtered['MOTIVO2_6 - HAS/dislipidemia/obesidade'].mode()[0], inplace=True)
test_filtered['MOTIVO2_6 - Palpitação/taquicardia/arritmia'].fillna(test_filtered['MOTIVO2_6 - Palpitação/taquicardia/arritmia'].mode()[0], inplace=True)
test_filtered['MOTIVO2_6 - Sopro'].fillna(test_filtered['MOTIVO2_6 - Sopro'].mode()[0], inplace=True)
test_filtered['MOTIVO2_Outro'].fillna(test_filtered['MOTIVO2_Outro'].mode()[0], inplace=True)
#test_filtered['Peso_Altura_Ratio'].fillna(train_dataset['Peso_Altura_Ratio'].median(), inplace=True)
test_filtered['PPA_num'].fillna(test_filtered['PPA_num'].mode()[0], inplace=True)


# Aplicar a padronização (Z-score) aos atributos numéricos
numerical_columns = ["IDADE", "Peso", "Altura", "IMC", "PA SISTOLICA", "PA DIASTOLICA", "FC", "PPA_num"]

# Calcular a média e desvio padrão de cada atributo
means = test_filtered[numerical_columns].mean()
stds = test_filtered[numerical_columns].std()

# Aplicar a padronização
test_filtered[numerical_columns] = (test_filtered[numerical_columns] - means) / stds


# Remover a coluna CLASSE do conjunto de teste, pois essa é a variável a ser predita
if "CLASSE" in test_filtered.columns:
    test_filtered = test_filtered.drop(columns=["CLASSE"])

# Separar features (X) e rótulos (y) do conjunto de treinamento
X_train = train_data.drop(columns=["Id", "CLASSE"])
y_train = train_data["CLASSE"]

# Features do conjunto de teste
X_test = test_filtered.drop(columns=["Id"])

# Verificar as dimensões dos conjuntos
X_train.shape, y_train.shape, X_test.shape


# Fazer previsões no conjunto de teste
pred_knn = knn.predict_proba(X_test)
pred_nb = nb_calibrated.predict_proba(X_test)
pred_svm = svm.predict_proba(X_test)
pred_mlp = mlp.predict_proba(X_test)
pred_rf = randomForest.predict_proba(X_test)
pred_lr = logisticRegression.predict_proba(X_test)
pred_gb = gb.predict_proba(X_test)
pred_dt = dt.predict_proba(X_test)
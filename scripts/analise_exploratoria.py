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

# Arquivo com todas as funcoes e codigos referentes a analise exploratoria

#Import Packages
import warnings
import pandas as pd
import os

from matplotlib import pyplot as plt
import seaborn


warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.2f}'.format

# Funcao para carregar os dados
FILES_DIRECTORY = "dados"

rhp_data = pd.read_csv(os.path.join(FILES_DIRECTORY, "RHP_data.csv"), sep=',', encoding='utf-8')
train_data = pd.read_csv(os.path.join(FILES_DIRECTORY, "train.csv"), sep=',', index_col=None)
test_data = pd.read_csv(os.path.join(FILES_DIRECTORY, 'test.csv'), sep=',', index_col=None)
rhp_data_classe = pd.merge(rhp_data, train_data, on='Id')

# Cria um novo dataframe sem os atributos Idade, IMC, Atendimento, Convenio, Sexo, HDA1 e HDA2. Esses atributos são reduntantes ou não são relevantes para a análise a ser feito
columns_to_drop = ['DN', 'Atendimento', 'Convenio', 'HDA 1', 'HDA2']
rhp_data_processed = rhp_data_classe.drop(columns=columns_to_drop, errors='ignore')


# Contar a distribuição das classes no conjunto de treino
class_distribution = rhp_data_processed["CLASSE"].value_counts(dropna=False)

# Visualizar a distribuição das classes
plt.figure(figsize=(6,4))
seaborn.barplot(x=class_distribution.index, y=class_distribution.values, palette="coolwarm")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.title("Distribuição das Classes no Conjunto de Treinamento")
plt.show()

# Identifying outliers
# Histogram for each attribute:
for attribute in rhp_data_processed.columns:
    n, bins, patches = plt.hist(rhp_data_processed[attribute], bins=10, color='blue', alpha=0.7, rwidth=0.85)
    plt.title(f"Histograma do Atributo {attribute}")
    plt.show()

# Density chart for each attribute (Help to identify outliers)
for attribute in rhp_data_processed.columns:
    densityPlot = rhp_data_processed[attribute].plot(kind='density', subplots=True, layout=(1, 1), sharex=False, sharey=False)
    plt.title(f"Gráfico de Densidade do Atributo {attribute}")
    plt.show()

# Check for null values in the dataset
missing_values = rhp_data.isnull().sum().sort_values(ascending=False)

# Show attributes with higher count of null values
missing_values = missing_values[missing_values > 0]
missing_values


# Análise Estatística
# Select only numeric columns
numerical_columns = rhp_data_processed.select_dtypes(include=["int64", "float64"]).columns

# Generate descriptive statistics
stats_summary = rhp_data_processed[numerical_columns].describe()

# Display the summary
print(stats_summary)

# Distribution plot
# Main numeric variables plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

seaborn.histplot(rhp_data_processed["Peso"].dropna(), bins=50, kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Distribuição do Peso")

seaborn.histplot(rhp_data_processed["Altura"].dropna(), bins=50, kde=True, ax=axes[0, 1])
axes[0, 1].set_title("Distribuição da Altura")

seaborn.histplot(rhp_data_processed["IMC"].dropna(), bins=50, kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Distribuição do IMC")

seaborn.histplot(rhp_data_processed["PA SISTOLICA"].dropna(), bins=50, kde=True, ax=axes[1, 1])
axes[1, 1].set_title("Distribuição da Pressão Sistólica")

plt.tight_layout()
plt.show()


# Correlation Matrix
# Calcular a matriz de correlação entre as variáveis numéricas
correlation_matrix = rhp_data_processed[numerical_columns].corr()

# Gerar um mapa de calor para visualização das correlações
plt.figure(figsize=(10, 8))
seaborn.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Matriz de Correlação entre Variáveis Numéricas")
plt.show()
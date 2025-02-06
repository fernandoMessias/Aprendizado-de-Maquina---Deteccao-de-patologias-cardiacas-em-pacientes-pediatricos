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

from scripts.analise_exploratoria import rhp_data_classe

# Arquivo com todas as funcoes e codigos referentes ao preprocessamento

# Cria um novo dataframe sem os atributos Idade, IMC, Atendimento, Convenio, Sexo, HDA1 e HDA2. Esses atributos são reduntantes ou não são relevantes para a análise a ser feito
columns_to_drop = ['DN', 'Atendimento', 'Convenio', 'HDA 1', 'HDA2']
rhp_data_processed = rhp_data_classe.drop(columns=columns_to_drop, errors='ignore')

# Verificar a distribuição das classes no conjunto de treinamento
class_distribution = rhp_data_processed['CLASSE'].value_counts(normalize=True)
print(class_distribution)

rhp_data_processed = rhp_data_processed.dropna(subset=['CLASSE'])

pulsos_distribution = rhp_data_processed['PULSOS'].value_counts(normalize=True)
ppa_distribution = rhp_data_processed['PPA'].value_counts(normalize=True)
b2_distribution = rhp_data_processed['B2'].value_counts(normalize=True)
sopro_distribution = rhp_data_processed['SOPRO'].value_counts(normalize=True)
sexo_distribution = rhp_data_processed['SEXO'].value_counts(normalize=True)
motivo1_distribution = rhp_data_processed['MOTIVO1'].value_counts(normalize=True)
motivo2_distribution = rhp_data_processed['MOTIVO2'].value_counts(normalize=True)


# Replace #VALUE! to NaN
rhp_data_processed.replace("#VALUE!", pd.NA, inplace=True)

rhp_data_processed['CLASSE'] = rhp_data_processed['CLASSE'].replace({
    'Normal': '0',
    'Normais': '0',
    'Anormal': '1'
}).astype('Int64')

# Selecionar colunas categóricas para conversão
categorical_cols = ["SEXO", "PULSOS", "PPA", "B2", "SOPRO", "MOTIVO1", "MOTIVO2"]

# Aplicar One-Hot Encoding
rhp_data_processed = pd.get_dummies(rhp_data_processed, columns=categorical_cols, drop_first=True)

# Exibir as primeiras linhas após a codificação
rhp_data_processed.head()

new_columns_to_drop = ['MOTIVO2_6 - Cianose e dispnéia', 'PULSOS_Amplos', 'PULSOS_NORMAIS', 'SOPRO_Sistolico e diastólico']
rhp_data_processed = rhp_data_processed.drop(columns=new_columns_to_drop, errors='ignore')


rhp_data_processed_normalized = rhp_data_processed
rhp_data_processed_normalized['Peso'] = pd.to_numeric(rhp_data_processed_normalized['Peso'], errors='coerce')
rhp_data_processed_normalized['Altura'] = pd.to_numeric(rhp_data_processed_normalized['Altura'], errors='coerce')
rhp_data_processed_normalized['IMC'] = pd.to_numeric(rhp_data_processed_normalized['IMC'], errors='coerce')
rhp_data_processed_normalized['IDADE'] = pd.to_numeric(rhp_data_processed_normalized['IDADE'], errors='coerce')
#rhp_data_processed_normalized['PULSOS'] = pd.to_numeric(rhp_data_processed_normalized['PULSOS'], errors='coerce')
rhp_data_processed_normalized['PA SISTOLICA'] = pd.to_numeric(rhp_data_processed_normalized['PA SISTOLICA'], errors='coerce')
rhp_data_processed_normalized['PA DIASTOLICA'] = pd.to_numeric(rhp_data_processed_normalized['PA DIASTOLICA'], errors='coerce')
#rhp_data_processed_normalized['PPA'] = pd.to_numeric(rhp_data_processed_normalized['PPA'], errors='coerce')
#rhp_data_processed_normalized['B2'] = pd.to_numeric(rhp_data_processed_normalized['B2'], errors='coerce')
rhp_data_processed_normalized['FC'] = pd.to_numeric(rhp_data_processed_normalized['FC'], errors='coerce')
# rhp_data_processed_normalized['SEXO'] = pd.to_numeric(rhp_data_processed_normalized['SEXO'], errors='coerce')
# rhp_data_processed_normalized['MOTIVO1'] = pd.to_numeric(rhp_data_processed_normalized['MOTIVO1'], errors='coerce')
# rhp_data_processed_normalized['MOTIVO2'] = pd.to_numeric(rhp_data_processed_normalized['MOTIVO2'], errors='coerce')
# rhp_data_processed_normalized['SOPRO'] = pd.to_numeric(rhp_data_processed_normalized['SOPRO'], errors='coerce')
rhp_data_processed_normalized['SEXO_Feminino'] = pd.to_numeric(rhp_data_processed_normalized['SEXO_Feminino'], errors='coerce')
rhp_data_processed_normalized['SEXO_Indeterminado'] = pd.to_numeric(rhp_data_processed_normalized['SEXO_Indeterminado'], errors='coerce')
rhp_data_processed_normalized['SEXO_M'] = pd.to_numeric(rhp_data_processed_normalized['SEXO_M'], errors='coerce')
rhp_data_processed_normalized['SEXO_Masculino'] = pd.to_numeric(rhp_data_processed_normalized['SEXO_Masculino'], errors='coerce')
rhp_data_processed_normalized['SEXO_masculino'] = pd.to_numeric(rhp_data_processed_normalized['SEXO_masculino'], errors='coerce')
#rhp_data_processed_normalized['PULSOS_Amplos'] = pd.to_numeric(rhp_data_processed_normalized['PULSOS_Amplos'], errors='coerce')
rhp_data_processed_normalized['PULSOS_Diminuídos '] = pd.to_numeric(rhp_data_processed_normalized['PULSOS_Diminuídos '], errors='coerce')
rhp_data_processed_normalized['PULSOS_Femorais diminuidos'] = pd.to_numeric(rhp_data_processed_normalized['PULSOS_Femorais diminuidos'], errors='coerce')
#rhp_data_processed_normalized['PULSOS_NORMAIS'] = pd.to_numeric(rhp_data_processed_normalized['PULSOS_NORMAIS'], errors='coerce')
rhp_data_processed_normalized['PULSOS_Normais'] = pd.to_numeric(rhp_data_processed_normalized['PULSOS_Normais'], errors='coerce')
rhp_data_processed_normalized['PULSOS_Outro'] = pd.to_numeric(rhp_data_processed_normalized['PULSOS_Outro'], errors='coerce')
rhp_data_processed_normalized['PPA_HAS-1 PAS'] = pd.to_numeric(rhp_data_processed_normalized['PPA_HAS-1 PAS'], errors='coerce')
rhp_data_processed_normalized['PPA_HAS-2 PAD'] = pd.to_numeric(rhp_data_processed_normalized['PPA_HAS-2 PAD'], errors='coerce')
rhp_data_processed_normalized['PPA_HAS-2 PAS'] = pd.to_numeric(rhp_data_processed_normalized['PPA_HAS-2 PAS'], errors='coerce')
rhp_data_processed_normalized['PPA_Normal'] = pd.to_numeric(rhp_data_processed_normalized['PPA_Normal'], errors='coerce')
rhp_data_processed_normalized['PPA_Não Calculado'] = pd.to_numeric(rhp_data_processed_normalized['PPA_Não Calculado'], errors='coerce')
rhp_data_processed_normalized['PPA_Pre-Hipertensão PAD'] = pd.to_numeric(rhp_data_processed_normalized['PPA_Pre-Hipertensão PAD'], errors='coerce')
rhp_data_processed_normalized['PPA_Pre-Hipertensão PAS'] = pd.to_numeric(rhp_data_processed_normalized['PPA_Pre-Hipertensão PAS'], errors='coerce')
rhp_data_processed_normalized['B2_Hiperfonética'] = pd.to_numeric(rhp_data_processed_normalized['B2_Hiperfonética'], errors='coerce')
rhp_data_processed_normalized['B2_Normal'] = pd.to_numeric(rhp_data_processed_normalized['B2_Normal'], errors='coerce')
rhp_data_processed_normalized['B2_Outro'] = pd.to_numeric(rhp_data_processed_normalized['B2_Outro'], errors='coerce')
rhp_data_processed_normalized['B2_Única'] = pd.to_numeric(rhp_data_processed_normalized['B2_Única'], errors='coerce')
#rhp_data_processed_normalized['SOPRO_Sistolico e diastólico'] = pd.to_numeric(rhp_data_processed_normalized['SOPRO_Sistolico e diastólico'], errors='coerce')
rhp_data_processed_normalized['SOPRO_Sistólico'] = pd.to_numeric(rhp_data_processed_normalized['SOPRO_Sistólico'], errors='coerce')
rhp_data_processed_normalized['SOPRO_ausente'] = pd.to_numeric(rhp_data_processed_normalized['SOPRO_ausente'], errors='coerce')
rhp_data_processed_normalized['SOPRO_contínuo'] = pd.to_numeric(rhp_data_processed_normalized['SOPRO_contínuo'], errors='coerce')
rhp_data_processed_normalized['SOPRO_diastólico'] = pd.to_numeric(rhp_data_processed_normalized['SOPRO_diastólico'], errors='coerce')
rhp_data_processed_normalized['SOPRO_sistólico'] = pd.to_numeric(rhp_data_processed_normalized['SOPRO_sistólico'], errors='coerce')
rhp_data_processed_normalized['MOTIVO1_2 - Check-up'] = pd.to_numeric(rhp_data_processed_normalized['MOTIVO1_2 - Check-up'], errors='coerce')
rhp_data_processed_normalized['MOTIVO1_5 - Parecer cardiológico'] = pd.to_numeric(rhp_data_processed_normalized['MOTIVO1_5 - Parecer cardiológico'], errors='coerce')
rhp_data_processed_normalized['MOTIVO1_6 - Suspeita de cardiopatia'] = pd.to_numeric(rhp_data_processed_normalized['MOTIVO1_6 - Suspeita de cardiopatia'], errors='coerce')
rhp_data_processed_normalized['MOTIVO1_7 - Outro'] = pd.to_numeric(rhp_data_processed_normalized['MOTIVO1_7 - Outro'], errors='coerce')
rhp_data_processed_normalized['MOTIVO2_1 - Cardiopatia congenica'] = pd.to_numeric(rhp_data_processed_normalized['MOTIVO2_1 - Cardiopatia congenica'], errors='coerce')
rhp_data_processed_normalized['MOTIVO2_5 - Atividade física'] = pd.to_numeric(rhp_data_processed_normalized['MOTIVO2_5 - Atividade física'], errors='coerce')
rhp_data_processed_normalized['MOTIVO2_5 - Cirurgia'] = pd.to_numeric(rhp_data_processed_normalized['MOTIVO2_5 - Cirurgia'], errors='coerce')
rhp_data_processed_normalized['MOTIVO2_5 - Uso de cisaprida'] = pd.to_numeric(rhp_data_processed_normalized['MOTIVO2_5 - Uso de cisaprida'], errors='coerce')
rhp_data_processed_normalized['MOTIVO2_6 - Alterações de pulso/perfusão'] = pd.to_numeric(rhp_data_processed_normalized['MOTIVO2_6 - Alterações de pulso/perfusão'], errors='coerce')
rhp_data_processed_normalized['MOTIVO2_6 - Cansaço'] = pd.to_numeric(rhp_data_processed_normalized['MOTIVO2_6 - Cansaço'], errors='coerce')
rhp_data_processed_normalized['MOTIVO2_6 - Cardiopatia na familia'] = pd.to_numeric(rhp_data_processed_normalized['MOTIVO2_6 - Cardiopatia na familia'], errors='coerce')
rhp_data_processed_normalized['MOTIVO2_6 - Cianose'] = pd.to_numeric(rhp_data_processed_normalized['MOTIVO2_6 - Cianose'], errors='coerce')
#rhp_data_processed_normalized['MOTIVO2_6 - Cianose e dispnéia'] = pd.to_numeric(rhp_data_processed_normalized['MOTIVO2_6 - Cianose e dispnéia'], errors='coerce')
rhp_data_processed_normalized['MOTIVO2_6 - Dispnéia'] = pd.to_numeric(rhp_data_processed_normalized['MOTIVO2_6 - Dispnéia'], errors='coerce')
rhp_data_processed_normalized['MOTIVO2_6 - Dor precordial'] = pd.to_numeric(rhp_data_processed_normalized['MOTIVO2_6 - Dor precordial'], errors='coerce')
rhp_data_processed_normalized['MOTIVO2_6 - HAS/dislipidemia/obesidade'] = pd.to_numeric(rhp_data_processed_normalized['MOTIVO2_6 - HAS/dislipidemia/obesidade'], errors='coerce')
rhp_data_processed_normalized['MOTIVO2_6 - Palpitação/taquicardia/arritmia'] = pd.to_numeric(rhp_data_processed_normalized['MOTIVO2_6 - Palpitação/taquicardia/arritmia'], errors='coerce')
rhp_data_processed_normalized['MOTIVO2_6 - Sopro'] = pd.to_numeric(rhp_data_processed_normalized['MOTIVO2_6 - Sopro'], errors='coerce')
rhp_data_processed_normalized['MOTIVO2_Outro'] = pd.to_numeric(rhp_data_processed_normalized['MOTIVO2_Outro'], errors='coerce')
rhp_data_processed_normalized['CLASSE'] = pd.to_numeric(rhp_data_processed_normalized['CLASSE'], errors='coerce')


# Criar Pressão de Pulso Arterial (PPA numérica)
rhp_data_processed_normalized["PPA_num"] = rhp_data_processed_normalized["PA SISTOLICA"] - rhp_data_processed_normalized["PA DIASTOLICA"]


rhp_data_processed_normalized['Peso'].fillna(rhp_data_processed_normalized['Peso'].median(), inplace=True)
rhp_data_processed_normalized['Altura'].fillna(rhp_data_processed_normalized['Altura'].median(), inplace=True)
rhp_data_processed_normalized['IMC'].fillna(rhp_data_processed_normalized['IMC'].median(), inplace=True)
rhp_data_processed_normalized['IDADE'].fillna(rhp_data_processed_normalized['IDADE'].median(), inplace=True)
#rhp_data_processed_normalized['PULSOS'].fillna(rhp_data_processed_normalized['PULSOS'].mode()[0], inplace=True)
rhp_data_processed_normalized['PA SISTOLICA'].fillna(rhp_data_processed_normalized['PA SISTOLICA'].median(), inplace=True)
rhp_data_processed_normalized['PA DIASTOLICA'].fillna(rhp_data_processed_normalized['PA DIASTOLICA'].median(), inplace=True)
#rhp_data_processed_normalized['PPA'].fillna(rhp_data_processed_normalized['PPA'].mode()[0], inplace=True)
#rhp_data_processed_normalized['B2'].fillna(rhp_data_processed_normalized['B2'].mode()[0], inplace=True)
rhp_data_processed_normalized['FC'].fillna(rhp_data_processed_normalized['FC'].median(), inplace=True)
# rhp_data_processed_normalized['SEXO'].fillna(rhp_data_processed_normalized['SEXO'].mode()[0], inplace=True)
# rhp_data_processed_normalized['MOTIVO1'].fillna(rhp_data_processed_normalized['MOTIVO1'].mode()[0], inplace=True)
# rhp_data_processed_normalized['MOTIVO2'].fillna(rhp_data_processed_normalized['MOTIVO2'].mode()[0], inplace=True)
# rhp_data_processed_normalized['SOPRO'].fillna(rhp_data_processed_normalized['SOPRO'].mode()[0], inplace=True)

rhp_data_processed_normalized['SEXO_Feminino'].fillna(rhp_data_processed_normalized['SEXO_Feminino'].mode()[0], inplace=True)
rhp_data_processed_normalized['SEXO_Indeterminado'].fillna(rhp_data_processed_normalized['SEXO_Indeterminado'].mode()[0], inplace=True)
rhp_data_processed_normalized['SEXO_M'].fillna(rhp_data_processed_normalized['SEXO_M'].mode()[0], inplace=True)
rhp_data_processed_normalized['SEXO_Masculino'].fillna(rhp_data_processed_normalized['SEXO_Masculino'].mode()[0], inplace=True)
rhp_data_processed_normalized['SEXO_masculino'].fillna(rhp_data_processed_normalized['SEXO_masculino'].mode()[0], inplace=True)
#rhp_data_processed_normalized['PULSOS_Amplos'].fillna(rhp_data_processed_normalized['PULSOS_Amplos'].mode()[0], inplace=True)
rhp_data_processed_normalized['PULSOS_Diminuídos '].fillna(rhp_data_processed_normalized['PULSOS_Diminuídos '].mode()[0], inplace=True)
rhp_data_processed_normalized['PULSOS_Femorais diminuidos'].fillna(rhp_data_processed_normalized['PULSOS_Femorais diminuidos'].mode()[0], inplace=True)
#rhp_data_processed_normalized['PULSOS_NORMAIS'].fillna(rhp_data_processed_normalized['PULSOS_NORMAIS'].mode()[0], inplace=True)
rhp_data_processed_normalized['PULSOS_Normais'].fillna(rhp_data_processed_normalized['PULSOS_Normais'].mode()[0], inplace=True)
rhp_data_processed_normalized['PULSOS_Outro'].fillna(rhp_data_processed_normalized['PULSOS_Outro'].mode()[0], inplace=True)
rhp_data_processed_normalized['PPA_HAS-1 PAS'].fillna(rhp_data_processed_normalized['PPA_HAS-1 PAS'].mode()[0], inplace=True)
rhp_data_processed_normalized['PPA_HAS-2 PAD'].fillna(rhp_data_processed_normalized['PPA_HAS-2 PAD'].mode()[0], inplace=True)
rhp_data_processed_normalized['PPA_HAS-2 PAS'].fillna(rhp_data_processed_normalized['PPA_HAS-2 PAS'].mode()[0], inplace=True)
rhp_data_processed_normalized['PPA_Normal'].fillna(rhp_data_processed_normalized['PPA_Normal'].mode()[0], inplace=True)
rhp_data_processed_normalized['PPA_Não Calculado'].fillna(rhp_data_processed_normalized['PPA_Não Calculado'].mode()[0], inplace=True)
rhp_data_processed_normalized['PPA_Pre-Hipertensão PAD'].fillna(rhp_data_processed_normalized['PPA_Pre-Hipertensão PAD'].mode()[0], inplace=True)
rhp_data_processed_normalized['PPA_Pre-Hipertensão PAS'].fillna(rhp_data_processed_normalized['PPA_Pre-Hipertensão PAS'].mode()[0], inplace=True)
rhp_data_processed_normalized['B2_Hiperfonética'].fillna(rhp_data_processed_normalized['B2_Hiperfonética'].mode()[0], inplace=True)
rhp_data_processed_normalized['B2_Normal'].fillna(rhp_data_processed_normalized['B2_Normal'].mode()[0], inplace=True)
rhp_data_processed_normalized['B2_Outro'].fillna(rhp_data_processed_normalized['B2_Outro'].mode()[0], inplace=True)
rhp_data_processed_normalized['B2_Única'].fillna(rhp_data_processed_normalized['B2_Única'].mode()[0], inplace=True)
#rhp_data_processed_normalized['SOPRO_Sistolico e diastólico'].fillna(rhp_data_processed_normalized['SOPRO_Sistolico e diastólico'].mode()[0], inplace=True)
rhp_data_processed_normalized['SOPRO_Sistólico'].fillna(rhp_data_processed_normalized['SOPRO_Sistólico'].mode()[0], inplace=True)
rhp_data_processed_normalized['SOPRO_ausente'].fillna(rhp_data_processed_normalized['SOPRO_ausente'].mode()[0], inplace=True)
rhp_data_processed_normalized['SOPRO_contínuo'].fillna(rhp_data_processed_normalized['SOPRO_contínuo'].mode()[0], inplace=True)
rhp_data_processed_normalized['SOPRO_diastólico'].fillna(rhp_data_processed_normalized['SOPRO_diastólico'].mode()[0], inplace=True)
rhp_data_processed_normalized['SOPRO_sistólico'].fillna(rhp_data_processed_normalized['SOPRO_sistólico'].mode()[0], inplace=True)
rhp_data_processed_normalized['MOTIVO1_2 - Check-up'].fillna(rhp_data_processed_normalized['MOTIVO1_2 - Check-up'].mode()[0], inplace=True)
rhp_data_processed_normalized['MOTIVO1_5 - Parecer cardiológico'].fillna(rhp_data_processed_normalized['MOTIVO1_5 - Parecer cardiológico'].mode()[0], inplace=True)
rhp_data_processed_normalized['MOTIVO1_6 - Suspeita de cardiopatia'].fillna(rhp_data_processed_normalized['MOTIVO1_6 - Suspeita de cardiopatia'].mode()[0], inplace=True)
rhp_data_processed_normalized['MOTIVO1_7 - Outro'].fillna(rhp_data_processed_normalized['MOTIVO1_7 - Outro'].mode()[0], inplace=True)
rhp_data_processed_normalized['MOTIVO2_1 - Cardiopatia congenica'].fillna(rhp_data_processed_normalized['MOTIVO2_1 - Cardiopatia congenica'].mode()[0], inplace=True)
rhp_data_processed_normalized['MOTIVO2_5 - Atividade física'].fillna(rhp_data_processed_normalized['MOTIVO2_5 - Atividade física'].mode()[0], inplace=True)
rhp_data_processed_normalized['MOTIVO2_5 - Cirurgia'].fillna(rhp_data_processed_normalized['MOTIVO2_5 - Cirurgia'].mode()[0], inplace=True)
rhp_data_processed_normalized['MOTIVO2_5 - Uso de cisaprida'].fillna(rhp_data_processed_normalized['MOTIVO2_5 - Uso de cisaprida'].mode()[0], inplace=True)
rhp_data_processed_normalized['MOTIVO2_6 - Alterações de pulso/perfusão'].fillna(rhp_data_processed_normalized['MOTIVO2_6 - Alterações de pulso/perfusão'].mode()[0], inplace=True)
rhp_data_processed_normalized['MOTIVO2_6 - Cansaço'].fillna(rhp_data_processed_normalized['MOTIVO2_6 - Cansaço'].mode()[0], inplace=True)
rhp_data_processed_normalized['MOTIVO2_6 - Cardiopatia na familia'].fillna(rhp_data_processed_normalized['MOTIVO2_6 - Cardiopatia na familia'].mode()[0], inplace=True)
rhp_data_processed_normalized['MOTIVO2_6 - Cianose'].fillna(rhp_data_processed_normalized['MOTIVO2_6 - Cianose'].mode()[0], inplace=True)
#rhp_data_processed_normalized['MOTIVO2_6 - Cianose e dispnéia'].fillna(rhp_data_processed_normalized['MOTIVO2_6 - Cianose e dispnéia'].mode()[0], inplace=True)
rhp_data_processed_normalized['MOTIVO2_6 - Dispnéia'].fillna(rhp_data_processed_normalized['MOTIVO2_6 - Dispnéia'].mode()[0], inplace=True)
rhp_data_processed_normalized['MOTIVO2_6 - Dor precordial'].fillna(rhp_data_processed_normalized['MOTIVO2_6 - Dor precordial'].mode()[0], inplace=True)
rhp_data_processed_normalized['MOTIVO2_6 - HAS/dislipidemia/obesidade'].fillna(rhp_data_processed_normalized['MOTIVO2_6 - HAS/dislipidemia/obesidade'].mode()[0], inplace=True)
rhp_data_processed_normalized['MOTIVO2_6 - Palpitação/taquicardia/arritmia'].fillna(rhp_data_processed_normalized['MOTIVO2_6 - Palpitação/taquicardia/arritmia'].mode()[0], inplace=True)
rhp_data_processed_normalized['MOTIVO2_6 - Sopro'].fillna(rhp_data_processed_normalized['MOTIVO2_6 - Sopro'].mode()[0], inplace=True)
rhp_data_processed_normalized['MOTIVO2_Outro'].fillna(rhp_data_processed_normalized['MOTIVO2_Outro'].mode()[0], inplace=True)
#rhp_data_processed_normalized['Peso_Altura_Ratio'].fillna(rhp_data_processed_normalized['Peso_Altura_Ratio'].median(), inplace=True)
rhp_data_processed_normalized['PPA_num'].fillna(rhp_data_processed_normalized['PPA_num'].mode()[0], inplace=True)


#rhp_data_processed_normalized['CLASSE'].fillna(rhp_data_processed_normalized['CLASSE'].median(), inplace=True)
rhp_data_processed_normalized['CLASSE'].fillna(rhp_data_processed_normalized['CLASSE'].mode()[0], inplace=True)

# Aplicar a padronização (Z-score) aos atributos numéricos
numerical_columns = ["IDADE", "Peso", "Altura", "IMC", "PA SISTOLICA", "PA DIASTOLICA", "FC", "PPA_num"]

# Calcular a média e desvio padrão de cada atributo
means = rhp_data_processed_normalized[numerical_columns].mean()
stds = rhp_data_processed_normalized[numerical_columns].std()

# Aplicar a padronização
rhp_data_processed_normalized[numerical_columns] = (rhp_data_processed_normalized[numerical_columns] - means) / stds

rhp_data_processed_normalized.head()

# Salvar dados pre-processados
rhp_data_processed_normalized.to_csv('dados/rhp_data_processed_normalized.csv', index=False)
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

from scripts.experimentos import test_filtered, pred_knn, pred_nb, pred_svm, pred_mlp, pred_rf, pred_lr, pred_gb, \
    pred_dt

# Arquivo com todas as funcoes e codigos referentes a analise dos resultados


# Criar um DataFrame para armazenar os resultados
results = test_filtered[["Id"]].copy()
results["k-NN"] = pred_knn[:, 1]
results["Naive Bayes"] = pred_nb[:, 1]
results["SVM"] = pred_svm[:, 1]
results["Redes Neurais"] = pred_mlp[:, 1]
results["Random Forest"] = pred_rf[:, 1]
results["Logistic Regression"] = pred_lr[:, 1]
results["Gradient Boosting"] = pred_gb[:, 1]
results["Decision Tree"] = pred_dt[:, 1]


# Gera os arquivos de submissão

# Criar DataFrame para submissão
submission_naive_bayes = pd.DataFrame({
    'Id': results["Id"],
    'Predicted': results["Naive Bayes"]
})

submission_knn = pd.DataFrame({
    'Id': results["Id"],
    'Predicted': results["k-NN"]
})

submission_redes_neurais = pd.DataFrame({
    'Id': results["Id"],
    'Predicted': results["Redes Neurais"]
})

submission_random_forest = pd.DataFrame({
    'Id': results["Id"],
    'Predicted': results["Random Forest"]
})

submission_logistic_regression = pd.DataFrame({
    'Id': results["Id"],
    'Predicted': results["Logistic Regression"]
})

submission_gradient_boosting = pd.DataFrame({
    'Id': results["Id"],
    'Predicted': results["Gradient Boosting"]
})

submission_decision_tree = pd.DataFrame({
    'Id': results["Id"],
    'Predicted': results["Decision Tree"]
})

submission_svm = pd.DataFrame({
    'Id': results["Id"],
    'Predicted': results["SVM"]
})

# Salvar o arquivo
submission_naive_bayes.to_csv("dados/submission_naive_bayes.csv", index=False)
submission_redes_neurais.to_csv("dados/submission_redes_neurais.csv", index=False)
submission_knn.to_csv("dados/submission_knn.csv", index=False)
submission_random_forest.to_csv("dados/random_forest.csv", index=False)
submission_logistic_regression.to_csv("dados/logistic_regression.csv", index=False)
submission_gradient_boosting.to_csv("dados/gradient_boosting.csv", index=False)
submission_decision_tree.to_csv("dados/decision_tree.csv", index=False)
submission_svm.to_csv("dados/svm.csv", index=False)
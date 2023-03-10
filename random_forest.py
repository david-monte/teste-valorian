import pandas as pd
import numpy as np
import datetime as dt
# import pickle

from sklearn import model_selection
# from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, make_scorer

from sklearn.preprocessing import LabelEncoder

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# import xgboost
# from xgboost import XGBClassifier

SEED = 1275
N_CLASSES = 6

def read_dataframe():

    def date_converter(data):
        return dt.datetime.fromtimestamp(int(data)*(10**-9)).strftime('%Y-%m-%d %H:%M:%S,%f')

    df = pd.read_csv("coleta.txt", 
                 names = ["id","atividade","timestamp","ac_X","ac_Y","ac_Z"],
                 lineterminator = ";",
                 parse_dates=[2],
                 date_parser=date_converter,
                 nrows = 1098208)
    
    print("\nInformações sobre o Conjunto de Dados:\n")
    df.info()
    print("\n")

    # Remoção dos registros nos quais os três acelerômetros apresentaram leituras nulas simultaneamente
    ac_zeros_index = list(df[(df['ac_X'] == 0) & (df['ac_Y'] == 0) & (df['ac_Z'] == 0)]['ac_X'].index)
    df.drop(ac_zeros_index, axis=0, inplace=True)

    ### Remoção dos Registros Duplicados
    df.drop_duplicates(subset=["atividade","timestamp","ac_X","ac_Y","ac_Z"], keep='first', inplace=True)
    
    ### Após exclusão de registros, reordena os index
    df.reset_index(drop=True, inplace=True)

    return df

def df_split(df):
    
    X = df.drop(['id','timestamp','atividade'], axis=1)
    y = df['atividade']

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    ### Divisão do conjunto de dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.25, stratify=y_enc, random_state=SEED) # shuffle=False

    return X_train, X_test, y_train, y_test, label_encoder

def train_model_search(X_train, X_test, y_train, y_test, label_encoder):
    
    # Instanciando o modelo
    decisionTree = DecisionTreeClassifier(random_state=SEED, class_weight='balanced')

    # K-fold estratificado com k = 10
    cv = StratifiedKFold(n_splits=10)

    # definição dos parâmetros para a árvore de decisão
    param_distributions = { 'criterion': ['gini', 'entropy'],
                            'min_samples_split': [12,15,18,20,22,24],
                            'max_depth': [6,7,8,9,10],
                            'min_samples_leaf': [5,6,7,8,9,10]
                        }

    # Construindo métrica de avaliação
    f1_weighted_scorer = make_scorer(f1_score, average='weighted')

    # define random search for decision tree
    rnd_search_tree = RandomizedSearchCV( estimator=decisionTree, 
                                        param_distributions = param_distributions, 
                                        n_iter=25, scoring = f1_weighted_scorer, 
                                        n_jobs=-1, cv=cv, random_state=SEED
                                        )

    # execute search
    result_tree = rnd_search_tree.fit(X_train, y_train)
    # summarize result for decision tree
    print('=========Random Search Results for TREE==========')
    print('Melhor Score: %s' % result_tree.best_score_)
    print('Melhores Hiper-parâmetros: %s' % result_tree.best_params_)

    # Instanciando e avaliando o modelo
    decisionTree = DecisionTreeClassifier(**result_tree.best_params_, random_state=SEED, class_weight='balanced')

    model_dt = decisionTree.fit(X_train, y_train)
    y_predicted_tree = decisionTree.predict(X_test)

    print('\nDesempenho médio da Árvore de Decisão:')

    cv_results = model_selection.cross_val_score(decisionTree, X_train, y_train, cv=cv, scoring=f1_weighted_scorer)
    # cv_models_results['decisionTree'] = cv_results

    name = 'Arvore de Decisão'
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

    print("\nAcuracia Árvore de Decisão: Treinamento",  decisionTree.score(X_train, y_train)," Teste", decisionTree.score(X_test, y_test))
    print("\nClassification report:\n", classification_report(label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(y_predicted_tree)))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_predicted_tree))

    #return

def main():
    print("******** RANDOM FOREST *************")
    print("\nImportando o Conjunto de Dados")

    df = read_dataframe()

    print("\nSelecionando os conjuntos de treinamento e teste na proporção 75/25")
    print("\nObs: Será utilizada a validação cruzada. Portanto, não há conjunto de validação")

    X_train, X_test, y_train, y_test, label_encoder = df_split(df)

    print("\nBusca de hiper-parâmetros de forma randômica, Treinamento e Avaliação do melhor modelo")
    print("\nObs.: Técnicas para lidar com as classes desbalanceadas foram aplicadas durante o treinamento do modelo")
    print("\nAguarde a exibição das métricas de avaliação...")
        
    train_model_search(X_train, X_test, y_train, y_test, label_encoder)
        
    print("\n\nTAREFA FINALIZADA\n")

main()
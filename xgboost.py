import pandas as pd
import numpy as np
import datetime as dt
# import pickle

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, make_scorer

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

# from sklearn import tree
# from sklearn.tree import DecisionTreeClassifier

import xgboost
from xgboost import XGBClassifier

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

def train_model(X_train, X_test, y_train, y_test, label_encoder):
    # Instanciando o Modelo
    xgb = XGBClassifier(
                    booster='gbtree',
                    objective='multi:softmax', num_class=N_CLASSES, max_depth=15,
                    learning_rate=0.1, n_estimators=100,
                    random_state=SEED, n_jobs=-1 #, tree_method='gpu_hist', gpu_id=0
                    )
    
    # Calculando os pesos das classes
    SAMPLE_WEIGHT = compute_sample_weight(class_weight='balanced', y=y_train)

    model = xgb.fit(X_train, y_train, sample_weight = SAMPLE_WEIGHT)
    y_predicted_xgb = xgb.predict(X_test)

    print("\nClassification report:\n", classification_report(label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(y_predicted_xgb)))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_predicted_xgb))
    #return

def train_model_search(X_train, X_test, y_train, y_test, label_encoder):
    # Instanciando o Modelo
    xgb = XGBClassifier(
                    booster='gbtree',
                    objective='multi:softmax', num_class=N_CLASSES, max_depth=15,
                    learning_rate=0.1, n_estimators=100,
                    random_state=SEED, n_jobs=-1 #, tree_method='gpu_hist', gpu_id=0
                    )
    
    # Calculando os pesos das classes
    SAMPLE_WEIGHT = compute_sample_weight(class_weight='balanced', y=y_train)

    PARAM_DISTRIBUTIONS = [
        {
            'n_estimators': [85, 100, 125, 150],
            'learning_rate':[0.01, 0.1, 0.2, 0.3],
            'max_depth':[6, 7, 8, 9, 10],
            #'booster':['gbtree', 'gblinear'],
            'objective':['multi:softmax'],
            'gamma':[0.1, 0.5, 1.0],
            'min_child_weight':[2, 3, 4],
            #'subsample':[0.5, 0.7, 0.8, 0.9],
            #'colsample_bytree':[0.5, 0.7, 0.8, 0.9],
            #'colsample_bynode':[0.5, 0.7, 0.8, 0.9],
            #'colsample_bylevel':[0.5, 0.7, 0.8, 0.9]
        }
    ]

    # K-fold estratificado com k = 10
    cv = StratifiedKFold(n_splits=10)

    # Construindo métrica de avaliação
    f1_weighted_scorer = make_scorer(f1_score, average='weighted')

    rnd_search = RandomizedSearchCV(estimator=xgb, 
                                    param_distributions = PARAM_DISTRIBUTIONS, 
                                    n_iter=25, scoring=f1_weighted_scorer,
                                    n_jobs=-1, cv=cv, random_state=SEED
                                )
    rnd_search.fit(X_train, y_train, sample_weight = SAMPLE_WEIGHT)

    ### Verificando os valores dos parâmetros
    print("\nParâmetros Encontrados: ", rnd_search.best_params_)
    print("\nAvaliação do XGBoost com os parâmetros atualizados")

    hyper_xgb = XGBClassifier(**rnd_search.best_params_, num_class=N_CLASSES, random_state=SEED)
    model_hyper = hyper_xgb.fit(X_train, y_train, sample_weight=SAMPLE_WEIGHT)
    y_predicted_xgb_h = hyper_xgb.predict(X_test)

    print("\nClassification report:\n", classification_report(label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(y_predicted_xgb_h)))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_predicted_xgb_h))

def main():
    print("******** XGBOOST *************")
    print("\nImportando o Conjunto de Dados")

    df = read_dataframe()

    print("\nSelecionando os conjuntos de treinamento e teste na proporção 75/25")

    X_train, X_test, y_train, y_test, label_encoder = df_split(df)

    print("\nTreinamento do XGBoost com número de estimadores igual a 100, taxa de aprendizagem igual a 0,1 e os demais hiper-parâmetros mantidos em valores padrão.")

    print("\nObs.: Técnicas para lidar com as classes desbalanceadas foram aplicadas durante o treinamento do modelo")
    print("\nAguarde a exibição das métricas de avaliação...")
        
    train_model(X_train, X_test, y_train, y_test, label_encoder)

    flag_tuner = 0

    while flag_tuner < 1:
        tuner = input("\nDeseja refinar os valores dos hiper-parâmetros? (S/N)")

        if tuner == 'N' or tuner == 'n':
          flag_tuner = 1
        elif tuner == 'S' or tuner == 's':
          print("\nBusca de hiper-parâmetros de forma randômica, Treinamento e Avaliação do melhor modelo")
          print("\nObs: Será utilizada a validação cruzada. Portanto, não há conjunto de validação")
          print("\nPor favor aguarde...")

          train_model_search(X_train, X_test, y_train, y_test, label_encoder)
          flag_tuner = 1
        else:
          flag_tuner = 0
          print("Opção Inválida! Tente novamente")

    print("\n\nTAREFA FINALIZADA!")

main()
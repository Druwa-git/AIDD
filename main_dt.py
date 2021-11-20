import argparse
import os
import time
import re

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

import nsml
from nsml import DATASET_PATH


def bind_model(model, optimizer=None):
    def save(path, *args, **kwargs):
        with open(os.path.join(path, 'model_lgb.pkl'), 'wb') as fp:
            joblib.dump(model, fp)
        print('Model saved')

    def load(path, *args, **kwargs):
        with open(os.path.join(path, 'model_lgb.pkl'), 'rb') as fp:
            temp_class = joblib.load(fp)
        nsml.copy(temp_class, model)
        print('Model loaded')

    # 추론
    def infer(path, **kwargs):
        return inference(path, model)

    nsml.bind(save=save, load=load, infer=infer)  # 'nsml.bind' function must be called at the end.


# 추론
def inference(path, model, **kwargs):
    data = preproc_data(pd.read_csv(path), train=False)

    pred_proba = model.predict_proba(data)[:, 1]
    pred_labels = np.where(pred_proba >= .5, 1, 0).reshape(-1)

    results = [(proba, label) for proba, label in zip(pred_proba, pred_labels)]

    return results


# 데이터 전처리
# train 쪽 data preproc 과 test 쪽을 같이 바꿔주세요!
def preproc_data(data, label=None, train=True, val_ratio=0.2, seed=1234):
    if train:
        dataset = dict()

        # 성별 ['M', 'F'] -> [0, 1]로 변환
        data['gender_enc'] = np.where(data['gender'] == 'M', 0, 1)

        # 컬럼명에 특수 JSON 문자 포함시 발생하는 오류 방지
        data = data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

        # 날짜 datetime으로 변환
        # df.loc[:, 'date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        DROP_COLS = ['CDMID', 'gender', 'date', 'date_E']
        X = data.drop(columns=DROP_COLS).copy()
        X = X.fillna(X.median())
        y = label

        X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                          stratify=y,
                                                          test_size=val_ratio,
                                                          random_state=seed,
                                                          )

        # Min-max scaling
        scaler = MinMaxScaler()
        X_cols = X.columns

        # Only X, y export
        X = pd.DataFrame(scaler.fit_transform(X), columns=X_cols)
        X['gender_enc'] = X['gender_enc'].astype('category')

        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_cols)
        X_train['gender_enc'] = X_train['gender_enc'].astype('category')

        X_val = pd.DataFrame(scaler.transform(X_val), columns=X_cols)
        X_val['gender_enc'] = X_val['gender_enc'].astype('category')

        dataset['X'] = X
        dataset['y'] = y
        dataset['X_train'] = X_train
        dataset['y_train'] = y_train
        dataset['X_val'] = X_val
        dataset['y_val'] = y_val

        return dataset

    else:
        # 성별 ['M', 'F'] -> [0, 1]로 변환
        data['gender_enc'] = np.where(data['gender'] == 'M', 0, 1)

        # 컬럼명에 특수 JSON 문자 포함시 발생하는 오류 방지
        data = data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

        DROP_COLS = ['CDMID', 'gender', 'date', 'date_E']
        data = data.drop(columns=DROP_COLS).copy()
        data = data.fillna(data.median())
        scaler = MinMaxScaler()
        X_cols = data.columns
        X_test = pd.DataFrame(scaler.fit_transform(data), columns=X_cols)
        X_test['gender_enc'] = X_test['gender_enc'].astype('category')

        return X_test


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    args.add_argument('--seed', type=int, default=42)
    config = args.parse_args()

    time_init = time.time()

    np.random.seed(config.seed)

    params = {
        'random_state': 0,
        'max_depth': 4,
        'min_samples_leaf': 1,
    }

    # max_depth=range(1, 20)
    # training_accuracy = []
    # test_accuracy = []
    # training_f1 = []
    # test_f1 = []

    model = DecisionTreeClassifier(**params)


    # change
    # nsml.bind() should be called before nsml.paused()
    bind_model(model)

    # DONOTCHANGE: They are reserved for nsml
    # Warning: Do not load data before the following code!
    # test mode
    if config.pause:
        nsml.paused(scope=locals())

    # training mode
    if config.mode == 'train':
        data_path = DATASET_PATH + '/train/train_data'
        label_path = DATASET_PATH + '/train/train_label'

        raw_data = pd.read_csv(data_path)
        raw_labels = np.loadtxt(label_path, dtype=np.int16)
        dataset = preproc_data(raw_data, raw_labels, train=True, val_ratio=0.2, seed=1234)

        X_train = dataset['X_train']
        y_train = dataset['y_train']
        X_val = dataset['X_val']
        y_val = dataset['y_val']
        X = dataset['X']
        y = dataset['y']
        time_dl_init = time.time()
        print('Time to dataset initialization: ', time_dl_init - time_init)

        # K-Fold Validation
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kfold)
        print("Final Score : ", sum(scores)/len(scores))

        nsml.save(0)  # name of checkpoint; 'model_lgb.pkl' will be saved

        final_time = time.time()
        print("Time to dataset initialization: ", time_dl_init - time_init)
        print("Time spent on training :", final_time - time_dl_init)
        print("Total time: ", final_time - time_init)

        print("Done")
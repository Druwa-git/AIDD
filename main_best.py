import argparse
import os
import time
import re

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from imblearn import over_sampling
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.feature_selection import mutual_info_classif

import nsml
from nsml import DATASET_PATH

random_seed = 42


class Transformer:
    def __init__(self):
        pass

    def transform(self, data, kind):

        if kind == 'NoTransform':
            return data
        elif kind == 'MinMax':
            self.scaler = MinMaxScaler()
        elif kind == 'Normalize':
            self.scaler = Normalizer()
        elif kind == 'Standard':
            self.scaler = StandardScaler()

        transformed_data = pd.DataFrame(self.scaler.fit_transform(data))
        transformed_data.columns = data.columns

        return transformed_data


class Selector:
    def __init__(self):
        pass

    def select(self, data, label, kind, k, include_column=[]):
        if kind == 'NoSelection':
            return list(set(data.columns.tolist()[:k]).union(set(include_column)))

        elif kind == 'MI':
            MI_df = pd.DataFrame(mutual_info_classif(data, label, random_state=random_seed), index=data.columns)
            selected_feats = list(
                set(MI_df.sort_values(by=0, ascending=False).index[:k].tolist()).union(set(include_column)))

        return selected_feats


class OverSampler:
    def __init__(self):
        pass

    def oversample(self, data, label, kind):
        if kind == 'NoOverSample':
            return data, label
        elif kind == 'ROSE':
            oversampled_data, oversampled_label = over_sampling.RandomOverSampler(
                random_state=random_seed).fit_resample(data, label)
        elif kind == 'SMOTE':
            oversampled_data, oversampled_label = over_sampling.SMOTE(random_state=random_seed).fit_resample(data,
                                                                                                             label)

        return oversampled_data, oversampled_label

random_seed = 42
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

    # output format
    # [(proba, label), (proba, label), ..., (proba, label)]
    results = [(proba, label) for proba, label in zip(pred_proba, pred_labels)]

    return results


# 데이터 전처리
def preproc_data(data, label=None, train=True, val_ratio=0.2, seed=1234, k=None):
    scaler = Transformer()
    transform_algorithm = "Normalize"
    if train:
        dataset = dict()

        # 성별 ['M', 'F'] -> [0, 1]로 변환
        data['gender_enc'] = np.where(data['gender'] == 'M', 0, 1)
        # 컬럼명에 특수 JSON 문자 포함시 발생하는 오류 방지
        data = data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

        # 날짜 datetime으로 변환
        # df.loc[:, 'date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        # Drop and fill Non value.
        DROP_COLS = ['CDMID', 'gender', 'date', 'date_E']
        X = data.drop(columns=DROP_COLS).copy()
        X = X.fillna(X.median())
        y = label

        selector = Selector()
        selected_features = selector.select(X, y, "MI", k)
        X = X[selected_features]

        # Oversample
        oversampler = OverSampler()
        X, y = oversampler.oversample(X, y, "ROSE")

        X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                          stratify=y,
                                                          test_size=val_ratio,
                                                          random_state=seed,
                                                          )
        # Min-max scaling
        X_cols = X.columns

        # Only X, y export
        X = pd.DataFrame(scaler.transform(X, transform_algorithm), columns=X_cols)
        X_train = pd.DataFrame(scaler.transform(X_train, transform_algorithm), columns=X_cols)
        X_val = pd.DataFrame(scaler.transform(X_val, transform_algorithm), columns=X_cols)

        if 'gender_enc' in selected_features:
            X['gender_enc'] = X['gender_enc'].astype('category')
            X_train['gender_enc'] = X_train['gender_enc'].astype('category')
            X_val['gender_enc'] = X_val['gender_enc'].astype('category')

        dataset['X'] = X
        dataset['y'] = y
        dataset['X_train'] = X_train
        dataset['y_train'] = y_train
        dataset['X_val'] = X_val
        dataset['y_val'] = y_val

        return dataset, selected_features

    else:
        # 성별 ['M', 'F'] -> [0, 1]로 변환
        data['gender_enc'] = np.where(data['gender'] == 'M', 0, 1)

        # 컬럼명에 특수 JSON 문자 포함시 발생하는 오류 방지
        data = data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

        DROP_COLS = ['CDMID', 'gender', 'date', 'date_E']
        data = data.drop(columns=DROP_COLS).copy()
        data = data.fillna(data.median())
        selected_features = ['gender_enc', 'HDL', 'Cr', 'Wt', 'DBP', 'HbA1c', 'FBG', 'TG', 'BMI', 'CrCl', 'GGT', 'ALT']
        data = data[selected_features]
        X_cols = data.columns
        X_test = pd.DataFrame(scaler.transform(data, transform_algorithm), columns=X_cols)
        if 'gender_enc' in selected_features:
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


    model = AdaBoostClassifier(n_estimators=55, random_state=42)
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
        dataset, selected_features = preproc_data(raw_data, raw_labels, train=True, val_ratio=0.2, seed=1234, k=12)
        print(selected_features)

        X_train = dataset['X']
        y_train = dataset['y']
        X_val = dataset['X_val']
        y_val = dataset['y_val']
        time_dl_init = time.time()
        print('Time to dataset initialization: ', time_dl_init - time_init)

        model.fit(X_train, y_train)

        nsml.save(0)  # name of checkpoint; 'model_lgb.pkl' will be saved

        final_time = time.time()
        print("Time to dataset initialization: ", time_dl_init - time_init)
        print("Time spent on training :", final_time - time_dl_init)
        print("Total time: ", final_time - time_init)

        print("Done")
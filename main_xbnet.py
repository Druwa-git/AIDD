import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

import nsml
from nsml import DATASET_PATH

from xgboost import XGBClassifier
from collections import OrderedDict


def bind_model(model, optimizer=None):
    def save(path, *args, **kwargs):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(path, 'model.pt'))
        print('Model saved')

    def load(path, *args, **kwargs):
        state = torch.load(os.path.join(path, 'model.pt'))
        model.load_state_dict(state['model'])
        print(state['model'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('Model loaded')

    # 추론
    def infer(path, **kwargs):
        return inference(path, model)

    nsml.bind(save=save, load=load, infer=infer)  # 'nsml.bind' function must be called at the end.


# 추론
def inference(path, model, **kwargs):
    model.eval()

    data = Variable(preproc_data(pd.read_csv(path), train=False))
    output_pred_labels = torch.round(torch.sigmoid(model(data, train=False)))
    output_pred_labels = output_pred_labels.detach().numpy()
    output_pred_labels = output_pred_labels.astype('int').reshape(-1).tolist()

    # output format
    # [(step, label), (step, label), ..., (step, label)]
    results = [(step, label) for step, label in enumerate(output_pred_labels)]

    return results


# 데이터 전처리
def preproc_data(data, label=None, train=True, val_ratio=0.2, seed=1234):
    if train:
        dataset = dict()

        # NaN 값 0으로 채우기
        data = data.fillna(0)

        # 성별 ['M', 'F'] -> [0, 1]로 변환
        data['gender_enc'] = np.where(data['gender'] == 'M', 0, 1)

        # 날짜 datetime으로 변환
        # df.loc[:, 'date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        DROP_COLS = ['CDMID', 'gender', 'date', 'date_E']
        X = data.drop(columns=DROP_COLS).copy()
        y = label

        X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                          stratify=y,
                                                          test_size=val_ratio,
                                                          random_state=seed,
                                                          )

        X_train = torch.as_tensor(X_train.values).float()
        y_train = torch.as_tensor(y_train.reshape(-1, 1)).float()
        X_val = torch.as_tensor(X_val.values).float()
        y_val = torch.as_tensor(y_val.reshape(-1, 1)).float()

        dataset['train'] = TensorDataset(X_train, y_train)
        dataset['val'] = TensorDataset(X_val, y_val)

        return dataset, X_train, y_train

    else:
        # NaN 값 0으로 채우기
        # 이 부분은 평균 혹은 median value로 바꿀 것.
        data = data.fillna(0)

        # 성별 ['M', 'F'] -> [0, 1]로 변환
        data['gender_enc'] = np.where(data['gender'] == 'M', 0, 1)

        # 날짜 datetime으로 변환
        # df.loc[:, 'date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        DROP_COLS = ['CDMID', 'gender', 'date', 'date_E']
        data = data.drop(columns=DROP_COLS).copy()

        X_test = torch.as_tensor(data.values).float()

        return X_test

class Seq(torch.nn.Sequential):
    '''
     Seq uses sequential module to implement tree in the forward.
    '''
    def give(self, xg, num_layers_boosted, ep=0.001):
        '''
        Saves various information into the object for further usage in the training process
        :param xg(object of XGBoostClassifier): Object og XGBoostClassifier
        :param num_layers_boosted(int,optional): Number of layers to be boosted in the neural network.
        :param ep(int,optional): Epsilon for smoothing. Deafult: 0.001
        '''
        self.xg = xg
        self.epsilon = ep
        self.boosted_layers = OrderedDict()
        self.num_layers_boosted = num_layers_boosted

    def forward(self, input,train,l=torch.Tensor([1])):
        l,train = train,l
        for i, module in enumerate(self):
            input = module(input)
            x0 = input
            if train:
                self.l = l
                if i < self.num_layers_boosted:
                    try:
                        self.boosted_layers[i] = torch.from_numpy(np.array(
                            self.xg.fit(x0.detach().numpy(), (self.l).detach().numpy(),eval_metric="mlogloss").feature_importances_) + self.epsilon)
                    except:
                        pass
        return input

# 모델
class XBNETClassifier(nn.Module):
    '''
    XBNetClassifier is a model for classification tasks that tries to combine tree-based models with
    neural networks to create a robust architecture.
         :param X_values(numpy array): Features on which model has to be trained
         :param y_values(numpy array): Labels of the features i.e target variable
         :param num_layers(int): Number of layers in the neural network
         :param num_layers_boosted(int,optional): Number of layers to be boosted in the neural network. Default value: 1
         :param input_through_cmd(Boolean): Use to tell how you provide the inputs
         :param inputs_for_gui(list): Use only for providing inputs through list and when input_through_cmd is
                set to True
    '''
    def __init__(self, X_values=None, y_values=None, num_layers=2, num_layers_boosted=1,
                 input_through_cmd = False,inputs_for_gui=None, train=True):
        super(XBNETClassifier, self).__init__()
        self.name = "Classification"
        self.layers = OrderedDict()
        self.boosted_layers = {}
        self.num_layers = num_layers
        self.num_layers_boosted = num_layers_boosted
        if train:
            self.X = X_values
            self.y = y_values
        self.gui = input_through_cmd
        self.inputs_layers_gui = inputs_for_gui

        self.take_layers_dim()
        if train:
            self.base_tree()
            self.layers[str(0)].weight = torch.nn.Parameter(torch.from_numpy(self.temp.T))

        self.xg = XGBClassifier(n_estimators=100)

        self.sequential = Seq(self.layers)
        self.sequential.give(self.xg, self.num_layers_boosted)
        self.feature_importances_ = None

    def get(self, l):
        '''
        Gets the set of current actual outputs of the inputs
        :param l(tensor): Labels of the current set of inputs that are getting processed.
        '''
        self.l = l


    def take_layers_dim(self):
        '''
        Creates the neural network by taking input from the user
        :param gyi(Boolean): Is it being for GUI building purposes
        '''
        if self.gui == True:
            counter = 0
            for i in range(self.num_layers):
                inp = self.inputs_layers_gui[counter]
                counter += 1
                out = self.inputs_layers_gui[counter]
                counter += 1
                set_bias = True
                self.layers[str(i)] = torch.nn.Linear(inp, out, bias=set_bias)
                if i == 0:
                    self.input_out_dim = out
                self.labels = out
        else:
            for i in range(self.num_layers):
                inp = int(input("Enter input dimensions of layer " + str(i + 1) + ": "))
                out = int(input("Enter output dimensions of layer " + str(i + 1)+ ": "))
                set_bias = True
                self.layers[str(i)] = torch.nn.Linear(inp, out, bias=set_bias)
                if i == 0:
                    self.input_out_dim = out
                self.labels = out
            # "1. Sigmoid 2. Softmax 3. None"
            self.ch = 1
            if self.ch == 1:
                self.layers[str(self.num_layers)] = torch.nn.Sigmoid()
            elif self.ch == 2:
                dimension = int(input("Enter dimension for Softmax: "))
                self.layers[str(self.num_layers)] = torch.nn.Softmax(dim=dimension)
            else:
                pass

    def base_tree(self):
        '''
        Instantiates and trains a XGBRegressor on the first layer of the neural network to set its feature importances
         as the weights of the layer
        '''
        self.temp1 = XGBClassifier(n_estimators=100).fit(self.X, self.y,eval_metric="mlogloss").feature_importances_
        self.temp = self.temp1
        for i in range(1, self.input_out_dim):
            self.temp = np.column_stack((self.temp, self.temp1))

    def forward(self, x, train=True):
        if train:
            x = self.sequential(x, self.l, train)
        else:
            x = self.sequential(x, train)
        return x

    def save(self,path):
        '''
        Saves the entire model in the provided path
        :param path(string): Path where model should be saved
        '''
        torch.save(self,path)


if __name__ == '__main__':

    args = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--val_ratio', type=int, default=0.2)
    args.add_argument('--lr', type=float, default=0.01)
    args.add_argument('--input_size', type=int, default=22)
    args.add_argument('--epochs', type=int, default=30)
    config = args.parse_args()

    time_init = time.time()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    gui_list = [22, 4, 4, 1]
    # DONOTCHANGE: They are reserved for nsml
    # Warning: Do not load data before the following code!
    # test mode
    if config.pause:
        model = XBNETClassifier(num_layers=2, input_through_cmd=True,
                                inputs_for_gui=gui_list, train=False)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.lr)

        # nsml.bind() should be called before nsml.paused()
        bind_model(model, optimizer=optimizer)
        nsml.paused(scope=locals())

    # training mode
    if config.mode == 'train':
        data_path = DATASET_PATH + '/train/train_data'
        label_path = DATASET_PATH + '/train/train_label'

        raw_data = pd.read_csv(data_path)
        raw_labels = np.loadtxt(label_path, dtype=np.int16)
        dataset, x_train, y_train = preproc_data(raw_data, raw_labels, train=True, val_ratio=0.2, seed=1234)

        model = XBNETClassifier(X_values=x_train, y_values=y_train, num_layers=2,
                                input_through_cmd=True, inputs_for_gui=gui_list)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.lr)

        # nsml.bind() should be called before nsml.paused()
        bind_model(model, optimizer=optimizer)

        train_dl = DataLoader(dataset['train'], config.batch_size, shuffle=True)
        val_dl = DataLoader(dataset['val'], config.batch_size, shuffle=False)
        time_dl_init = time.time()
        print('Time to dataloader initialization: ', time_dl_init - time_init)

        min_val_loss = np.inf
        for epoch in range(config.epochs):
            # train model
            running_loss = 0.
            num_runs = 0
            model.train()
            total_length = len(train_dl)
            for iter_idx, (data, labels) in enumerate(train_dl):
                out = labels
                try:
                    if out.shape[0] >= 1:
                        out = torch.squeeze(out, 1)
                except:
                    pass
                model.get(out.float())
                data = Variable(data)
                labels = Variable(labels)

                output_pred = model(data)
                loss = loss_fn(output_pred, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_runs += 1

                # get current lr
                opt_params = optimizer.state_dict()['param_groups'][0]
                step = epoch * total_length + iter_idx

                nsml.report(
                    epoch=epoch + int(config.iteration),
                    epoch_total=config.epochs,
                    iter=iter_idx,
                    iter_total=total_length,
                    batch_size=config.batch_size,
                    train__loss=running_loss / num_runs,
                    step=step,
                    lr=opt_params['lr'],
                    scope=locals()
                )

            print(f"[Epoch {epoch}] Loss: {running_loss / num_runs}")

            # test model with validation data
            model.eval()
            running_loss = 0.
            num_runs = 0

            correct, total = 0, 0
            for data, labels in val_dl:
                data = Variable(data)
                labels = Variable(labels)

                output_pred = model(data)
                loss = loss_fn(output_pred, labels)

                running_loss += loss.item()
                num_runs += 1

                predicted = (output_pred.data >= 0.5)
                total += labels.size(0)
                correct += (predicted == labels).sum()


            print(f"[Validation] Loss: {running_loss / num_runs}")
            print('Accuracy: %f %%' % (100.0 * float(correct) / float(total)))

            nsml.report(
                summary=True,
                epoch=epoch,
                epoch_total=config.epochs,
                val__loss=running_loss / num_runs,
                step=(epoch + 1) * total_length,
                lr=opt_params['lr']
            )

            if (running_loss < min_val_loss) or (epoch % 10 == 0):
                nsml.save(epoch)

        final_time = time.time()
        print("Time to dataloader initialization: ", time_dl_init - time_init)
        print("Time spent on training :", final_time - time_dl_init)
        print("Total time: ", final_time - time_init)

        print("Done")
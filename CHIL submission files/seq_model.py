import config
import data_models as dm
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, f1_score
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture as GMM
import pandas as pd
import numpy as np
from numpy.random import choice
from collections import defaultdict
import pickle
import time

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal, OneHotCategorical, Categorical, Independent, MixtureSameFamily

from torch.autograd import Variable
import torch.nn.functional as F
torch.manual_seed(0)


# plotting
plt.rc('font', size=16, family='serif')
plt.grid(zorder=-100)
plt.switch_backend('agg')
plt.rcParams.update({'figure.max_open_warning': 0})

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

np.random.seed(config.rnd_seed)


class TARNET(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers, output_size=1, num_actions=8, drop_rate=0.2):
        super(TARNET, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        #self.n_components = n_components
        self.output_size = output_size
        # Defining the layers
        # Dropout layer
        self.dropout = nn.Dropout(drop_rate)
        self.num_actions = num_actions

        # RNN Layer
        self.rnn = nn.LSTM(input_size, hidden_dim, n_layers,
                           dropout=drop_rate, batch_first=True, bidirectional=True)
        self.elu = nn.ELU()

        # Fully connected layer
        self.final0 = nn.Linear(hidden_dim * 2, self.output_size)
        self.final1 = nn.Linear(hidden_dim * 2, self.output_size)

        self.final2 = nn.Linear(hidden_dim * 2, self.output_size)
        self.final3 = nn.Linear(hidden_dim * 2, self.output_size)

        self.final4 = nn.Linear(hidden_dim * 2, self.output_size)
        self.final5 = nn.Linear(hidden_dim * 2, self.output_size)

        self.final6 = nn.Linear(hidden_dim * 2, self.output_size)
        self.final7 = nn.Linear(hidden_dim * 2, self.output_size)

    def forward(self, x, a):
        batch_size = x.size(0)
        # print(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        rnn_out, hidden = self.rnn(x)

        # hidden shape: [num_layers, batch_size, rnn_size]
        rnn_out = self.dropout(rnn_out)
        #print("rnn_out.shape: ",rnn_out.shape)

        rnn_out = rnn_out[:, -1, :]

        if(a == 0):
            out = self.final0(rnn_out)
        elif(a == 1):
            out = self.final1(rnn_out)
        elif(a == 2):
            out = self.final2(rnn_out)
        elif(a == 3):
            out = self.final3(rnn_out)
        elif(a == 4):
            out = self.final4(rnn_out)
        elif(a == 5):
            out = self.final5(rnn_out)
        elif(a == 6):
            out = self.final6(rnn_out)
        elif(a == 7):
            out = self.final7(rnn_out)

        return out

    def init_state(self, device, batch_size=1):
        """
        initialises rnn states.
        """
        return (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))


class Model(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers, output_size=1, drop_rate=0.2):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_size = output_size

        # Dropout layer
        self.dropout = nn.Dropout(drop_rate)

        # Defining the layers

        # RNN Layer
        self.rnn = nn.LSTM(input_size, hidden_dim, n_layers,
                           dropout=drop_rate, batch_first=True, bidirectional=True)
        self.relu = nn.ReLU()

        # Fully connected layer
        self.final = nn.Linear(2 * hidden_dim, self.output_size)

    def forward(self, x):
        batch_size = x.size(0)
        # Passing in the input and hidden state into the model and obtaining outputs
        rnn_out, hidden = self.rnn(x)

        rnn_out = self.dropout(self.relu(rnn_out))
        # print("rnn_out.shape: ",rnn_out.shape)  # hidden shape: [num_layers, batch_size, rnn_size]

        #print("rnn_out.shape: ", rnn_out.shape)
        rnn_out = rnn_out[:, -1, :]
        #print("rnn_out.shape: ", rnn_out.shape)

        out = self.final(rnn_out)
        return out

    def init_state(self, device, batch_size=1):
        """
        initialises rnn states.
        """
        return (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))


class SeqDataset(Dataset):
    """A Dataset that stores a list of sequences and their corresponding category labels."""

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.float32)

    def __getitem__(self, idx):

        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)


def nll_loss(pi, normal, y):
    loglik = normal.log_prob(y.unsqueeze(1).expand_as(normal.loc))
    loglik = torch.sum(loglik, dim=2)
    # loss = -torch.logsumexp(torch.log(pi.probs) + loglik, dim=1)
    loss = -manual_logsumexp(torch.log(pi.probs) + loglik, dim=1)
    return torch.mean(loss)


def manual_logsumexp(x, dim=1):
    return torch.log(torch.sum(torch.exp(x), dim=dim) + 1e-10)


class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


def train_validate(train_loader, val_loader, A, params, model, optimizer, criterion, scheduler, model_name):
    valid_loss_min = np.Inf  # To track change in validation loss
    device = params.device
    print("device: ", device)
    pat = 0

    avg_train_losses = []
    avg_valid_losses = []

    for epoch in range(1, params.numEpochs + 1):
        print("Epoch: ", epoch)
        #print("valid_loss_min: ",valid_loss_min)

        # keep track of training loss
        running_train_loss = 0.0
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        start_time_train = time.time()
        model.train()
        ###################
        # train the model #
        ###################
        #hidden = decoder.module.init_hidden()
        hidden = model.init_state(device, params.batch_size)
        for batch_idx, (x, y) in enumerate(train_loader):

            x, y = x.float().to(device), y.view(-1, 1).float().to(device)  # , a.to(device)

            # Create a new variable for the hidden state, necessary to calculate the gradients
            hidden = tuple(([Variable(var.data) for var in hidden]))

            optimizer.zero_grad()  # Clears existing gradients from previous epoch

            out = model(x)

            loss = criterion(out, y)

            loss.backward()  # Does backpropagation and calculates gradients

            with torch.no_grad():
                model.final.bias.fill_(0.)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()  # Updates the weights accordingly

            # update training loss
            running_train_loss += loss.item() * x.size(0)

            # print statistics
            running_loss += loss.item()
            if batch_idx % params.minibatch_print_freq == params.minibatch_print_freq - 1:    # print every 100 mini-batches
                print('[Epoch: %d, Minibatch: %5d] Loss: %.3f' %
                      (epoch, batch_idx + 1, running_loss / params.minibatch_print_freq))
                running_loss = 0.0

            total_train += y.size(0)

        end_time_train = time.time()

        start_time_val = time.time()
        # calculate average losses
        avg_train_loss = running_train_loss / total_train

        ######################
        # validate the model #
        ######################

        # keep track of validation loss
        total_val = 0
        correct_val = 0
        running_valid_loss = 0.0

        end_time_train = time.time()
        model.eval()
        val_hidden = model.init_state(device, params.batch_size)
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x, y = x.float().to(device), y.view(-1, 1).float().to(device)  # , a.to(device)
                # Create a new variable for the hidden state, necessary to calculate the gradients
                hidden = tuple(([Variable(var.data) for var in val_hidden]))

                out = model(x)
                loss = criterion(out, y)

                # update training loss
                running_valid_loss += loss.item() * x.size(0)

                total_val += y.size(0)

        # calculate average losses
        avg_valid_loss = running_valid_loss / total_val
        end_time_val = time.time()

        # calculate average losses
        avg_train_losses.append(avg_train_loss)
        avg_valid_losses.append(avg_valid_loss)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(list(range(1, len(avg_train_losses) + 1)),
                 avg_train_losses, color='g')
        plt.plot(list(range(1, len(avg_valid_losses) + 1)),
                 avg_valid_losses, color='r')

        plt.title('Plot of Training and validation losses')
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.grid(True)
        plt.savefig(params.plots_path + 'train_val_loss_' + model_name +
                    '_' + str(int(A)) + '.png', format='png', dpi=1000)
        plt.close(fig)

        scheduler.step()
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTraining Time: {:.6f} s \tValidation Time: {:.6f} s'.format(
            epoch, avg_train_loss, avg_valid_loss, end_time_train - start_time_train, end_time_val - start_time_val))

        torch.save(model.state_dict(), params.models_path +
                   'latest_' + model_name + '_' + str(int(A)) + '.pth')

        # save model if validation loss has decreased
        if avg_valid_loss < valid_loss_min:
            model_lowest_val_loss = model
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                avg_valid_loss))
            torch.save(model_lowest_val_loss.state_dict(), params.models_path +
                       'minvalloss_' + model_name + '_' + str(int(A)) + '.pth')
            valid_loss_min = avg_valid_loss
            pat = 0
        else:
            pat += 1
            if pat >= params.patience:
                print("Early stopping !")
                break

    return model_lowest_val_loss


def predict_on_val(params, model, val_loader):

    pred_label = []
    true_label = []

    device = params.device

    model.eval()

    val_hidden = model.init_state(device, params.batch_size)
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            x, y = x.float().to(device),  y.view(-1, 1).float().to(device)  # , a.to(device)
            # Create a new variable for the hidden state, necessary to calculate the gradients
            hidden = tuple(([Variable(var.data) for var in val_hidden]))

            out = model(x)
            pred_label.append(out.cpu().numpy())
            true_label.append(y.cpu().numpy())

            del out
            del y

            torch.cuda.empty_cache()

    return np.concatenate(pred_label).flatten().astype(np.float32), np.concatenate(true_label).flatten().astype(np.float32)


def generate_training_data_TARNet(RIDs, history, A, adj_cols, prev_cols, gen_autoreg_df, month):
    X = []
    Y = []
    A = []

    cols = ['RID', 'VISCODE', 'Y_0', 'Y_1', 'Y_2', 'Y_3',
            'Y_4', 'Y_5', 'Y_6', 'Y_7', 'Y_hat', 'DX', 'A_Cat']
    cols = adj_cols + cols

    for rid in RIDs:
        df = gen_autoreg_df.loc[(gen_autoreg_df['RID'] == rid) & (gen_autoreg_df['VISCODE'] >= (
            month - (history * 12))) & (gen_autoreg_df['VISCODE'] <= month)][cols]
        df_target = gen_autoreg_df.loc[(gen_autoreg_df['RID'] == rid) & (
            gen_autoreg_df['VISCODE'] == month)][['Y_hat']]
        df_a = gen_autoreg_df.loc[(gen_autoreg_df['RID'] == rid) & (
            gen_autoreg_df['VISCODE'] == month)][['A_Cat']]
        df['DX'] = df['DX'].shift(1)
        df['DX'] = df['DX'].fillna(-1)
        #df.loc[df['VISCODE']==month]['Y_hat'] = 1000
        df.loc[df['VISCODE'] == month, ['Y_hat']] = 86
        # print(df)
        #min_max_scaler = MinMaxScaler()
        #df[['Y_hat']] = df[['Y_hat']].pow(3)
        #df[['Y_hat']] = min_max_scaler.fit_transform(df[['Y_hat']])

        df = pd.concat([df, pd.get_dummies(df['DX'].astype(pd.CategoricalDtype(
            categories=list(range(3)))), prefix='DX', drop_first=True)], axis=1)
        df.drop(['DX'], axis=1, inplace=True)

        df = dm.check_categorical(
            df, list(df.columns), '', categorical_cols=config.Categorical_cols)

        df.drop(['RID'], axis=1, inplace=True)
        df.drop(['VISCODE'], axis=1, inplace=True)
        df.drop(['A_Cat'], axis=1, inplace=True)

        X.append(df.values)
        Y.append(df_target.values[0])
        A.append(df_a.values[0])
    X = np.array(X)
    Y = np.array(Y).flatten()
    A = np.array(A).flatten()

    return X, Y, A


def generate_training_data_TLearner(RIDs, history, A, adj_cols, prev_cols, gen_autoreg_df, month):
    X = []
    Y = []
    Y_A = []
    Y_0 = []

    cols = ['RID', 'VISCODE', 'Y_0', 'Y_1', 'Y_2', 'Y_3',
            'Y_4', 'Y_5', 'Y_6', 'Y_7', 'Y_hat', 'DX', 'A_Cat']
    cols = adj_cols + cols

    for rid in RIDs:
        df = gen_autoreg_df.loc[(gen_autoreg_df['RID'] == rid) & (gen_autoreg_df['VISCODE'] >= (
            month - (history * 12))) & (gen_autoreg_df['VISCODE'] <= month)][cols]
        if(int(df.query('VISCODE ==' + str(month))['A_Cat']) == A):
            df_target = df.loc[df['VISCODE'] == month][['Y_hat']]
            df_Y_A = df.loc[df['VISCODE'] == month][['Y_' + str(int(A))]]
            df_Y_0 = df.loc[df['VISCODE'] == month][['Y_0']]

            for prev_col in prev_cols:
                df[prev_col + '_prev'] = df[prev_col].shift(1)
                if(prev_col == 'DX'):
                    df[prev_col + '_prev'] = df[prev_col + '_prev'].fillna(-1)
                elif(prev_col == 'Y_hat'):
                    df[prev_col + '_prev'] = df[prev_col + '_prev'].fillna(170)

            df['A_Cat_prev'] = df['A_Cat'].shift(1)
            df['A_Cat_prev'] = df['A_Cat_prev'].fillna(-1)

            df.loc[df['VISCODE'] == month, ['Y_hat']] = 170  # Invalid value

            for i in range(0, 8):
                df.drop(['Y_' + str(int(i))], axis=1, inplace=True)

            # print(df.head())
            #print(float(df_Y_A.values[0]), float(df_target.values[0]))
            df.drop(['RID'], axis=1, inplace=True)
            df.drop(['VISCODE'], axis=1, inplace=True)
            df.drop(['A_Cat'], axis=1, inplace=True)
            df.drop(['Y_hat'], axis=1, inplace=True)
            if config.unconfounded:
                df.drop(['DX'], axis=1, inplace=True)

            df = dm.check_categorical(
                df, list(df.columns), '', categorical_cols=config.Categorical_cols)

            X.append(df.values)
            Y.append(float(df_target.values[0]))
            Y_A.append(float(df_Y_A.values[0]))
            Y_0.append(float(df_Y_0.values[0]))

    X = np.array(X)
    Y = np.array(Y).flatten()
    Y_A = np.array(Y_A).flatten()
    Y_0 = np.array(Y_0).flatten()

    #print(X.shape, Y.shape, Y_A.shape, Y_0.shape)

    return zip(X, Y, Y_A, Y_0)


def generate_ATE_testing_data(RIDs, history, gen_autoreg_df, month):
    X = []
    Y = []

    cols = ['RID', 'VISCODE', 'PTETHCAT', 'PTRACCAT', 'PTEDUCAT',
            'PTGENDER', 'PTMARRY', 'TAU', 'PTAU', 'APOE4', 'FDG', 'AV45', 'Y_hat']
    for rid in RIDs:
        # print(rid)
        df = gen_autoreg_df.loc[(gen_autoreg_df['RID'] == rid) & (
            gen_autoreg_df['VISCODE'] <= month - (history * 12))][cols]
        df_target = gen_autoreg_df.loc[(gen_autoreg_df['RID'] == rid) & (
            gen_autoreg_df['VISCODE'] == month)][['Y_hat']]
        df['DX'] = df['DX'].shift(1)
        df['DX'] = df['DX'].fillna(-1)
        df.loc[df['VISCODE'] == month, ['Y_hat']] = 86
        print(df)

        df = pd.concat([df, pd.get_dummies(df['DX'].astype(pd.CategoricalDtype(
            categories=list(range(3)))), prefix='DX', drop_first=True)], axis=1)
        df.drop(['DX'], axis=1, inplace=True)

        df = dm.check_categorical(
            df, list(df.columns), '', categorical_cols=config.Categorical_cols)

        df.drop(['RID'], axis=1, inplace=True)
        df.drop(['VISCODE'], axis=1, inplace=True)

        X.append(df.values)
        Y.append(df_target.values[0])
    #df_ground = pDataFrameaFrame([Y_A, Y_0], columns=['Y_A', 'Y_0'])

    X = np.array(X)
    Y = np.array(Y)

    return X, Y


def generate_CATE_testing_data(RIDs, history, A, adj_cols, prev_cols, gen_autoreg_df, month):
    X = []
    Y = []
    Y_A = []
    Y_0 = []

    cols = ['RID', 'VISCODE', 'Y_0', 'Y_1', 'Y_2', 'Y_3',
            'Y_4', 'Y_5', 'Y_6', 'Y_7', 'Y_hat', 'DX', 'A_Cat']
    cols = adj_cols + cols

    for rid in RIDs:
        # print(rid)
        df = gen_autoreg_df.loc[(gen_autoreg_df['RID'] == rid) & (gen_autoreg_df['VISCODE'] >= (
            month - (history * 12))) & (gen_autoreg_df['VISCODE'] <= month)][cols]

        df_target = df.loc[df['VISCODE'] == month][['Y_hat']]
        df_Y_A = df.loc[df['VISCODE'] == month][['Y_' + str(int(A))]]
        df_Y_0 = df.loc[df['VISCODE'] == month][['Y_0']]

        # print(df.loc[(df['RID']==RIDs[0])][[]])

        for prev_col in prev_cols:

            df[prev_col + '_prev'] = df[prev_col].shift(1)

            if(prev_col == 'DX'):
                df[prev_col + '_prev'] = df[prev_col + '_prev'].fillna(-1)
            elif(prev_col == 'Y_hat'):
                df[prev_col + '_prev'] = df[prev_col + '_prev'].fillna(170)

        # print(df.loc[(df['RID']==RIDs[0])])

        df.loc[df['VISCODE'] == month, ['Y_hat']] = 170  # Invalid value

        df['A_Cat_prev'] = df['A_Cat'].shift(1)
        df['A_Cat_prev'] = df['A_Cat_prev'].fillna(-1)

        # df = dm.check_categorical(
        #    df, list(df.columns), '', categorical_cols=config.Categorical_cols)

        df.drop(['RID'], axis=1, inplace=True)
        df.drop(['VISCODE'], axis=1, inplace=True)
        df.drop(['A_Cat'], axis=1, inplace=True)
        df.drop(['Y_hat'], axis=1, inplace=True)
        if config.unconfounded:
            df.drop(['DX'], axis=1, inplace=True)

        df = dm.check_categorical(
            df, list(df.columns), '', categorical_cols=config.Categorical_cols)

        for i in range(0, 8):
            df.drop(['Y_' + str(int(i))], axis=1, inplace=True)

        X.append(df.values)
        Y.append(df_target.values[0])
        Y_A.append(df_Y_A.values[0])
        Y_0.append(df_Y_0.values[0])

    X = np.array(X)
    Y = np.array(Y).flatten()
    Y_A = np.array(Y_A).flatten()
    Y_0 = np.array(Y_0).flatten()

    #print(X.shape, Y.shape, Y_A.shape, Y_0.shape)

    return X, Y, Y_A, Y_0


def train(month, history, gen_autoreg_df_train, gen_autoreg_df_test, RIDs_train,
          RIDs_test, policy, N, epsilon, gamma_0, gamma_1, train=True):
    params = config.SeqParameters()
    adj_cols = []
    prev_cols = []

    print("Policy: ", policy)

    model_name = params.model_name + '_month_' + str(month) + '_history_' + str(history) + '_policy_' + str(
        policy) + '_N_' + str(N) + '_epsilon_' + str(epsilon) + '_gamma0_' + str(gamma_0) + '_gamma1_' + str(gamma_1)

    if (policy == 'Santiago'):
        adj_cols = config.cols_Santiago_Based_seq
        prev_cols = config.prev_cols_Santiago_Based
        params.input_size = 44
    elif(policy == 'DX'):
        adj_cols = config.cols_DX_Based_seq
        prev_cols = config.prev_cols_DX_Based
        params.input_size = 43

    if(train):
        train_sets = {}
        test_sets = {}
        models = {}

        for i in range(8):
            #generate_training_data_TLearner(RIDs, history, A, adj_cols, prev_cols, gen_autoreg_df, month)
            train_sets[i] = generate_training_data_TLearner(
                RIDs_train, history, i, adj_cols, prev_cols, gen_autoreg_df_train, month)
            test_sets[i] = generate_training_data_TLearner(
                RIDs_test, history, i, adj_cols, prev_cols, gen_autoreg_df_test, month)

        for a in range(8):
            # DataLoaders
            X_train, Y_train, _, _ = zip(*train_sets[a])
            X_train = np.array(X_train)
            Y_train = np.array(Y_train)
            print(X_train.shape, Y_train.shape)

            X_test, Y_test, _, _ = zip(*test_sets[a])
            X_test = np.array(X_test)
            Y_test = np.array(Y_test)

            print("A: ", a, len(X_train), len(X_test))
            dataset_train = SeqDataset(X_train, Y_train)
            dataset_val = SeqDataset(X_test, Y_test)

            train_dataloader = DataLoader(
                dataset_train, batch_size=params.batch_size, drop_last=True, shuffle=True)
            val_dataloader = DataLoader(
                dataset_val, batch_size=params.batch_size, drop_last=True, shuffle=True)

            # Model
            model = Model(params.input_size,
                          params.hidden_dim, params.n_layers)
            #model = torch.nn.DataParallel(model)
            model = model.cuda()

            # print(model)

            # Model parameters
            optimizer = torch.optim.SGD(
                model.parameters(), lr=params.lr, weight_decay=params.decay, momentum=0.9)
            criterion = params.criterion
            steps = params.steps
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, params.steps)

            # Train
            models[a] = train_validate(train_dataloader, val_dataloader, a,
                                       params, model, optimizer, criterion, scheduler, model_name)
    return models


def seq_ATE_PEHE(RIDs_test, history, gen_autoreg_df, month, policy, N, epsilon, gamma_0, gamma_1):
    params = config.SeqParameters()
    ATEs = {}
    pehes = {}
    m_hats = {}

    adj_cols = []
    prev_cols = []
    input_size = 0

    print('policy: ', policy)

    if (policy == 'Santiago'):
        adj_cols = config.cols_Santiago_Based_seq
        prev_cols = config.prev_cols_Santiago_Based
        params.input_size = 44  # 45
    elif(policy == 'DX'):
        adj_cols = config.cols_DX_Based_seq
        prev_cols = config.prev_cols_DX_Based
        params.input_size = 43  # 44

    model_name = params.model_name + '_month_' + str(month) + '_history_' + str(history) + '_policy_' + str(
        policy) + '_N_' + str(N) + '_epsilon_' + str(epsilon) + '_gamma0_' + str(gamma_0) + '_gamma1_' + str(gamma_1)
    #print(len(RIDs_test), len(gen_autoreg_df))

    for a in range(8):
        X, Y, Y_A, Y_0 = generate_CATE_testing_data(
            RIDs_test, history, a, adj_cols, prev_cols, gen_autoreg_df, month)
        X = np.array(X)
        Y = np.array(Y)
        Y_A = np.array(Y_A)
        Y_0 = np.array(Y_0)

        dataset_val = SeqDataset(X, Y)
        CATE_val_dataloader = DataLoader(
            dataset_val, batch_size=1, drop_last=True, shuffle=False)

        model = Model(params.input_size, params.hidden_dim, params.n_layers)

        model.load_state_dict(torch.load(
            params.models_path + 'minvalloss_' + model_name + '_' + str(a) + '.pth'))
        model.cuda()
        pred_Y_A, true = predict_on_val(params, model, CATE_val_dataloader)

        m_hats[a] = np.mean(pred_Y_A)

        model0 = Model(params.input_size, params.hidden_dim, params.n_layers)
        model0.load_state_dict(torch.load(
            params.models_path + 'minvalloss_' + model_name + '_' + str(0) + '.pth'))
        model0.cuda()
        pred_Y_0, true = predict_on_val(params, model0, CATE_val_dataloader)

        data = pd.DataFrame({'true': Y, 'Y_A': Y_A, 'pred_Y_A': pred_Y_A, 'Y_0': Y_0,
                            'pred_Y_0': pred_Y_0}, columns=['true', 'Y_A', 'pred_Y_A', 'Y_0', 'pred_Y_0'])

        data['Y_A_Y_0'] = data['Y_A'] - data['Y_0']
        data['pred_Y_A_pred_Y_0'] = data['pred_Y_A'] - data['pred_Y_0']

        pehe = mean_squared_error(data['Y_A_Y_0'], data['pred_Y_A_pred_Y_0'])
        pehes[a] = pehe

    for i in range(8):
        ATEs[i] = m_hats[i] - m_hats[0]

        return pehes, ATEs

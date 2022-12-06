import config
import data_models as dm
import warnings
import pandas as pd
import numpy as np
import random
from numpy.random import choice
from collections import defaultdict
import copy

from sklearn.mixture import GaussianMixture as GMM
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.preprocessing import LabelEncoder

from scipy.stats import norm

from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# plotting
import matplotlib.pyplot as plt
from matplotlib import ticker

plt.rc('font', size=16, family='serif')
plt.grid(zorder=-100)
plt.switch_backend('agg')
plt.rcParams.update({'figure.max_open_warning': 0})

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

np.random.seed(config.rnd_seed)


def pre_processing(df1, df2, train_size):
    # Columns to be standardized

    # , 'Y_0', 'Y_1', 'Y_2', 'Y_3', 'Y_4', 'Y_5', 'Y_6', 'Y_7','Y_hat']
    standardize_x_cols = ['TAU', 'PTAU', 'FDG', 'AV45', 'AGE', 'CDRSB', 'MMSE']

    gen_autoreg_df = df1.copy()
    gen_autoreg_df_random = df2.copy()

    sc = StandardScaler()
    gen_autoreg_df[standardize_x_cols] = sc.fit_transform(
        gen_autoreg_df[standardize_x_cols])
    gen_autoreg_df_random[standardize_x_cols] = sc.fit_transform(
        gen_autoreg_df_random[standardize_x_cols])

    # creating instance of labelencoder
    labelencoder = LabelEncoder()
    gen_autoreg_df['A_Cat'] = labelencoder.fit_transform(gen_autoreg_df['A'])
    gen_autoreg_df_random['A_Cat'] = labelencoder.fit_transform(
        gen_autoreg_df_random['A'])

    gen_autoreg_df = pd.concat([gen_autoreg_df, pd.get_dummies(
        gen_autoreg_df['A'], prefix='A', drop_first=True)], axis=1)
    gen_autoreg_df.drop(['A'], axis=1, inplace=True)

    gen_autoreg_df_random = pd.concat([gen_autoreg_df_random, pd.get_dummies(
        gen_autoreg_df_random['A'], prefix='A', drop_first=True)], axis=1)
    gen_autoreg_df_random.drop(['A'], axis=1, inplace=True)

    # Train-Test Data split
    RIDs = np.array(pd.unique(gen_autoreg_df.RID)).astype(int)
    random.Random(0).shuffle(RIDs)
    train_size = 0.8

    RIDs_train = np.array(RIDs[:int(train_size * len(RIDs))]).astype(int)
    RIDs_test = np.setdiff1d(RIDs, RIDs_train)

    gen_autoreg_df_train = gen_autoreg_df.loc[(
        gen_autoreg_df['RID'].isin(RIDs_train))]
    gen_autoreg_df_test = gen_autoreg_df.loc[(
        gen_autoreg_df['RID'].isin(RIDs_test))]

    gen_autoreg_df_random_train = gen_autoreg_df_random.loc[(
        gen_autoreg_df_random['RID'].isin(RIDs_train))]
    gen_autoreg_df_random_test = gen_autoreg_df_random.loc[(
        gen_autoreg_df_random['RID'].isin(RIDs_test))]

    return gen_autoreg_df, gen_autoreg_df_random, gen_autoreg_df_train, gen_autoreg_df_test, gen_autoreg_df_random_train, gen_autoreg_df_random_test, RIDs_train, RIDs_test


# Ground Truth Mean Treatment effects Random Assignment
def a_GtruthEffects(A, df, month):
    a = 'A_' + str(int(A))
    data = df.loc[df['VISCODE'] == month]
    tauMean = np.mean(data.loc[data[a] == 1]['Delta'].values)
    return tauMean

# T-Learner


def outcome_Regression_Adjustment_TLearner(df, month, A, cols, prev_cols, Y_col, test_size=0.2):
    a = 'A_' + str(int(A))
    # print(a)
    data = df.loc[df['VISCODE'] == int(month)][cols]
    for prev_col in prev_cols:
        data[prev_col + '_prev'] = df.loc[df['VISCODE'] ==
                                          int(month - 12)][prev_col].values  # Get from previous month

    if(A == 0):
        data = data.loc[((data['A_1'] == 0) & (data['A_2'] == 0) & (data['A_3'] == 0) & (
            data['A_4'] == 0) & (data['A_5'] == 0) & (data['A_6'] == 0) & (data['A_7'] == 0))]
    else:
        data = data.loc[data[a] == 1]

    data = dm.check_categorical(
        data, cols, Y_col, categorical_cols=config.Categorical_cols)
    data = data.dropna()
    # print(data)

    # Shuffle the dataset.
    data_shuffled = data.sample(frac=1.0, random_state=0)
    # Split into input part X and output part Y.
    for i in range(1, 8):
        data_shuffled = data_shuffled.drop('A_' + str(int(i)), axis=1)

    data_X = data_shuffled.drop(Y_col, axis=1)
    data_Y = data_shuffled[Y_col]

    # Partition the data into training and test sets.
    data_Xtrain, data_Xtest, data_Ytrain, data_Ytest = train_test_split(
        data_X, data_Y, test_size=test_size, random_state=0)
    # print(len(data_Xtrain))
    Effect_estimators = copy.deepcopy(config.Effect_estimators)
    regr = Effect_estimators[config.regrr]  # Lasso(alpha=0.1)
    regr.fit(data_Xtrain, data_Ytrain)

    # print(data_Xtest.columns)
    data_Yguess = regr.predict(data_Xtest)
    rmse = np.sqrt(mean_squared_error(data_Yguess, data_Ytest))
    print("ATE TLearner " + Y_col + " RMSE: ", rmse,
          " R2  :", r2_score(data_Yguess, data_Ytest))

    return regr, rmse


def predict_Regression_Adjustment_TLearner(model, A, df, month, cols, prev_cols, Y_col):
    #a = 'A_' + str(int(A))
    data = df.loc[df['VISCODE'] == int(month)][cols]
    for prev_col in prev_cols:
        data[prev_col + '_prev'] = df.loc[df['VISCODE'] ==
                                          int(month - 12)][prev_col].values  # Get from previous month

    data = dm.check_categorical(
        data, cols, Y_col, categorical_cols=config.Categorical_cols)
    data = data.dropna()

    for i in range(1, 8):
        data = data.drop('A_' + str(int(i)), axis=1)

    # print(len(data))
    true_Y_hat = data['Y_hat'].values

    data = data.drop(Y_col, axis=1)
    Y_hat = model.predict(data)

    Y_hat_mean = np.mean(Y_hat)

    data['true_Y_hat'] = true_Y_hat
    data['pred_Y_hat'] = Y_hat

    return data, Y_hat_mean


def outcome_Regression_Adjustment_SLearner(df, month, cols, prev_cols, Y_col, test_size=0.2, categorical_cols=config.Categorical_cols):

    data = df.loc[df['VISCODE'] == int(month)][cols]

    for prev_col in prev_cols:
        data[prev_col + '_prev'] = df.loc[df['VISCODE'] ==
                                          int(month - 12)][prev_col].values  # Get from previous month

    data = dm.check_categorical(
        data, cols, Y_col, categorical_cols=config.Categorical_cols)
    data = data.dropna()
    # print(data)

    # Shuffle the dataset.
    data_shuffled = data.sample(frac=1.0, random_state=0)
    # Split into input part X and output part Y.
    data_X = data_shuffled.drop(Y_col, axis=1)
    data_Y = data_shuffled[Y_col]

    # Partition the data into training and test sets.
    data_Xtrain, data_Xtest, data_Ytrain, data_Ytest = train_test_split(
        data_X, data_Y, test_size=test_size, random_state=0)

    Effect_estimators = copy.deepcopy(config.Effect_estimators)
    regr = Effect_estimators[config.regrr]  # Lasso(alpha=0.1)
    regr.fit(data_Xtrain, data_Ytrain)

    # print(data_Xtest.columns)
    data_Yguess = regr.predict(data_Xtest)
    rmse = np.sqrt(mean_squared_error(data_Yguess, data_Ytest))
    print("ATE SLearner " + Y_col + " RMSE: ", rmse,
          " R2  :", r2_score(data_Yguess, data_Ytest))

    return regr, rmse


def predict_Regression_Adjustment_SLearner(model, A, df, month, cols, prev_cols, Y_col):
    #a = 'A_' + str(int(A))
    data = df.loc[df['VISCODE'] == int(month)][cols]

    for prev_col in prev_cols:
        data[prev_col + '_prev'] = df.loc[df['VISCODE'] ==
                                          int(month - 12)][prev_col].values  # Get from previous month

    for i in range(1, 8):
        data['A_' + str(int(i))] = 0

    if(A != 0):
        data['A_' + str(int(A))] = 1
    # print(data.head())

    data = dm.check_categorical(
        data, cols, Y_col, categorical_cols=config.Categorical_cols)
    data = data.dropna()
    # print(len(data))
    true_Y_hat = data['Y_hat'].values

    data = data.drop(Y_col, axis=1)
    Y_hat = model.predict(data)

    Y_hat_mean = np.mean(Y_hat)

    data['true_Y_hat'] = true_Y_hat
    data['pred_Y_hat'] = Y_hat

    return data, Y_hat_mean


def plot_ATE(title, val1, val2, val3, a):
    df = pd.DataFrame(val1, columns=['Ground Truth ATE'], index=a)
    df['Model with Diagnosis-informed policy'] = val2
    df['Model with Random Policy'] = val3

    plt.rc('font', size=16, family='serif')
    plt.grid(zorder=-100)
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.style.use('tableau-colorblind10')

    df.plot(kind='bar')
    plt.xlabel("Treatment")
    plt.ylabel("Treatment Effect")
    plt.grid(color='k', linestyle='dotted')

    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.gca().xaxis.set_tick_params(rotation=0)
    plt.savefig('plots/ATE_plot.png', format='png', dpi=500)

# S-Learner


def S_Learner_ATE(month, gen_autoreg_df_train, gen_autoreg_df_test, gen_autoreg_df_random_train, gen_autoreg_df_random_test):

    cols = ['Y_hat', 'A_1', 'A_2', 'A_3', 'A_4', 'A_5', 'A_6', 'A_7']

    print("DX Policy")

    Y_col = 'Y_hat'
    RegAdjustment_model, RegAdjustment_rmse = outcome_Regression_Adjustment_SLearner(
        gen_autoreg_df_train, month, config.cols_DX_Based + cols, config.prev_cols_DX_Based, Y_col, test_size=0.2)
    #print(" model.coef_ ", RegAdjustment_model.coef_)

    print("Santiago Hernandez Policy")
    RegAdjustment_model_random, RegAdjustment_rmse_random = outcome_Regression_Adjustment_SLearner(
        gen_autoreg_df_random_train, month,  config.cols_Santiago_Based + cols, config.prev_cols_Santiago_Based, Y_col, test_size=0.2)
    #print(" model.coef_ ", RegAdjustment_model_random.coef_)

    m_hats = {}
    for i in range(8):
        _, Y_hat_mean = predict_Regression_Adjustment_SLearner(
            RegAdjustment_model, i, gen_autoreg_df_test, month, config.cols_DX_Based + cols, config.prev_cols_DX_Based, Y_col)
        # print(data.head(10))
        m_hats[i] = Y_hat_mean

    m_hats_random = {}
    for i in range(8):
        _, Y_hat_mean_random = predict_Regression_Adjustment_SLearner(
            RegAdjustment_model_random, i, gen_autoreg_df_random_test, month,  config.cols_Santiago_Based + cols, config.prev_cols_Santiago_Based, Y_col)
        # print(data.head(10))
        m_hats_random[i] = Y_hat_mean_random

    ATEs = {}
    for i in range(8):
        ATEs[i] = m_hats[i] - m_hats[0]
    ATEs_random = {}
    for i in range(8):
        ATEs_random[i] = m_hats_random[i] - m_hats_random[0]

    return ATEs, ATEs_random

# T-Learner


def T_Learner_ATE(month, gen_autoreg_df_train, gen_autoreg_df_test, gen_autoreg_df_random_train, gen_autoreg_df_random_test):
    cols = ['Y_hat', 'A_1', 'A_2', 'A_3', 'A_4', 'A_5', 'A_6', 'A_7']
    Y_col = 'Y_hat'

    print("DX Policy")
    A_models = {}
    for i in range(8):
        RegAdjustment_model, RegAdjustment_rmse = outcome_Regression_Adjustment_TLearner(
            gen_autoreg_df_train, month, i,  config.cols_DX_Based + cols, config.prev_cols_DX_Based, Y_col, test_size=0.2)
        A_models[i] = RegAdjustment_model

    print("Santiago Hernandez Policy")
    A_models_random = {}
    for i in range(8):
        RegAdjustment_model_random, RegAdjustment_rmse_random = outcome_Regression_Adjustment_TLearner(
            gen_autoreg_df_random_train, month, i,  config.cols_Santiago_Based + cols, config.prev_cols_Santiago_Based, Y_col, test_size=0.2)
        A_models_random[i] = RegAdjustment_model_random

    m_hats = {}
    m_hats_random = {}

    for i in list(range(8)):
        _, Y_hat_mean = predict_Regression_Adjustment_TLearner(
            A_models[i], i, gen_autoreg_df_test, month,  config.cols_DX_Based + cols, config.prev_cols_DX_Based, Y_col)
        m_hats[i] = Y_hat_mean

        # Random
        _, Y_hat_mean_random = predict_Regression_Adjustment_TLearner(
            A_models_random[i], i, gen_autoreg_df_random_test, month,   config.cols_Santiago_Based + cols, config.prev_cols_Santiago_Based, Y_col)
        m_hats_random[i] = Y_hat_mean_random

    ATEs = {}
    ATEs_random = {}
    for i in range(8):
        ATEs[i] = m_hats[i] - m_hats[0]
        ATEs_random[i] = m_hats_random[i] - m_hats_random[0]

    return ATEs, ATEs_random

# IPW


def IPW_Adjustment_Learner(df, month, cols, prev_cols, Y_col, test_size=0.2):
    data = df.loc[df['VISCODE'] == int(month)][cols]

    for prev_col in prev_cols:
        data[prev_col + '_prev'] = df.loc[df['VISCODE'] ==
                                          int(month - 12)][prev_col].values  # Get from previous month

    data = data.drop('Y_hat', axis=1)
    data = dm.check_categorical(
        data, cols, Y_col, categorical_cols=config.Categorical_cols)
    data = data.dropna()

    # Shuffle the dataset.
    data_shuffled = data.sample(frac=1.0, random_state=0)
    # Split into input part X and output part Y.
    data_X = data_shuffled.drop(Y_col, axis=1)
    data_Y = data_shuffled[Y_col]

    # Partition the data into training and test sets.
    data_Xtrain, data_Xtest, data_Ytrain, data_Ytest = train_test_split(
        data_X, data_Y, test_size=test_size, random_state=0)

    lr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    #lr = RandomForestClassifier(max_depth=10, random_state=0)

    lr.fit(data_Xtrain, data_Ytrain)
    data_Yguess = lr.predict(data_Xtest)
    print("IPW " + Y_col + " Accuracy  :", accuracy_score(data_Yguess, data_Ytest),
          " F1 score: ", f1_score(data_Yguess, data_Ytest, average='weighted'))

    return lr


def weight_trimming(epsilon, x):
    #print(x, epsilon)
    if (x < epsilon):
        x = epsilon
    elif (x > (1 - epsilon)):
        x = 1 - epsilon
    # print(x)
    return x


def predictIPW_Adjustment(model, epsilon, A, df, month, cols, prev_cols, Y_col):

    data = df.loc[df['VISCODE'] == int(month)][cols]
    for prev_col in prev_cols:
        data[prev_col + '_prev'] = df.loc[df['VISCODE'] ==
                                          int(month - 12)][prev_col].values  # Get from previous month

    data = dm.check_categorical(
        data, cols, Y_col, categorical_cols=config.Categorical_cols)
    # print(len(data))
    data = data.dropna()

    N = len(data)

    # print(data.head())
    data0 = data.loc[data['A_Cat'] == 0]
    n0 = len(data0)

    # print(data.head())
    data = data.loc[data['A_Cat'] == A]
    na = len(data)

    pred_data = data.drop('Y_hat', axis=1)
    pred_data = pred_data.drop('A_Cat', axis=1)
    # print(len(pred_data))

    a_probs = model.predict_proba(pred_data)
    # print(len(a_probs))

    data['a_probs'] = [x[A] for x in a_probs]  # a_probs
    # print(data['a_probs'])
    # print(len(data.a_probs))

    data['Y_hat_div_a_prob'] = data['Y_hat'] / data['a_probs']
    # data['Y_hat_div_a_prob'] = data['Y_hat_div_a_prob'].map(lambda x: weight_trimming(epsilon, x))#a_probs

    # print(len(data['Y_hat_div_a_prob']))
    Y_hat_mean = np.sum(data['Y_hat_div_a_prob'].values) / N

    return data, Y_hat_mean


def IPW_ATE(month, gen_autoreg_df_train, gen_autoreg_df_test, gen_autoreg_df_random_train, gen_autoreg_df_random_test):

    cols = ['A_Cat', 'Y_hat']

    print("DX Policy")
    Y_col = 'A_Cat'

    IPW_model = IPW_Adjustment_Learner(
        gen_autoreg_df_train, month, config.cols_DX_Based + cols, config.prev_cols_DX_Based, Y_col, test_size=0.2)

    print("Santiago Policy")
    IPW_model_random = IPW_Adjustment_Learner(
        gen_autoreg_df_random_train, month, config.cols_Santiago_Based + cols, config.prev_cols_Santiago_Based, Y_col, test_size=0.2)

    epsilon = 1e1
    m_hats = {}
    m_hats_random = {}
    for i in range(8):
        data, Y_hat_mean = predictIPW_Adjustment(
            IPW_model, epsilon, i, gen_autoreg_df_test, month, config.cols_DX_Based + cols, config.prev_cols_DX_Based, Y_col)
        m_hats[i] = Y_hat_mean

        # Random
        _, Y_hat_mean_random = predictIPW_Adjustment(
            IPW_model_random, epsilon, i, gen_autoreg_df_random_test, month, config.cols_Santiago_Based + cols, config.prev_cols_Santiago_Based, Y_col)
        m_hats_random[i] = Y_hat_mean_random

    ATEs = {}
    ATEs_random = {}
    for i in range(8):
        ATEs[i] = m_hats[i] - m_hats[0]
        ATEs_random[i] = m_hats_random[i] - m_hats_random[0]

    return ATEs, ATEs_random

# CATE Estimation
# T-Learner


def CATE_TLearner(df, month, A, cols, prev_cols, Y_col, test_size=0.2):
    a = 'A_' + str(int(A))
    data = df.loc[df['VISCODE'] == int(month)][cols]

    for prev_col in prev_cols:
        data[prev_col + '_prev'] = df.loc[df['VISCODE'] ==
                                          int(month - 12)][prev_col].values  # Get DX from previous month

    data = dm.check_categorical(
        data, cols, Y_col, categorical_cols=config.Categorical_cols)
    data = data.dropna()

    if(A == 0):
        data = data.loc[((data['A_1'] == 0) & (data['A_2'] == 0) & (data['A_3'] == 0) & (
            data['A_4'] == 0) & (data['A_5'] == 0) & (data['A_6'] == 0) & (data['A_7'] == 0))]
    else:
        data = data.loc[data[a] == 1]

    # print(data)
    # Shuffle the dataset.
    data_shuffled = data.sample(frac=1.0, random_state=0)
    # Split into input part X and output part Y.
    for i in range(1, 8):
        data_shuffled = data_shuffled.drop('A_' + str(int(i)), axis=1)

    for i in range(0, 8):
        data_shuffled = data_shuffled.drop('Y_' + str(int(i)), axis=1)

    data_shuffled = data_shuffled.drop('A_Cat', axis=1)

    data_X = data_shuffled.drop(Y_col, axis=1)
    data_Y = data_shuffled[Y_col]

    # Partition the data into training and test sets.
    data_Xtrain, data_Xtest, data_Ytrain, data_Ytest = train_test_split(
        data_X, data_Y, test_size=test_size, random_state=0)
    # print(len(data_Xtrain))

    Effect_estimators = copy.deepcopy(config.Effect_estimators)
    regr = Effect_estimators[config.regrr]
    regr.fit(data_Xtrain, data_Ytrain)

    # print(data_Xtest.columns)
    data_Yguess = regr.predict(data_Xtest)
    rmse = np.sqrt(mean_squared_error(data_Yguess, data_Ytest))
    print("CATE TLearner " + Y_col + " RMSE: ", rmse,
          " R2  :", r2_score(data_Yguess, data_Ytest))

    return regr, rmse


def predict_CATE_TLearner(model_0, model, A, df, month, cols, prev_cols, Y_col):
    #a = 'A_' + str(int(A))
    data = df.loc[df['VISCODE'] == int(month)][cols]
    for prev_col in prev_cols:
        data[prev_col + '_prev'] = df.loc[df['VISCODE'] ==
                                          int(month - 12)][prev_col].values  # Get from previous month

    data = dm.check_categorical(
        data, cols, Y_col, categorical_cols=config.Categorical_cols)
    data = data.dropna()

    for i in range(1, 8):
        data = data.drop('A_' + str(int(i)), axis=1)

    # print(len(data))
    true_Y_hat = data['Y_hat'].values
    true_A = data['A_Cat'].values

    data = data.drop('A_Cat', axis=1)

    Y_A = data['Y_' + str(int(A))].values
    Y_0 = data['Y_0'].values

    for i in range(0, 8):
        data = data.drop('Y_' + str(int(i)), axis=1)

    data = data.drop(Y_col, axis=1)
    Y_hat_0 = model_0.predict(data)
    Y_hat_pred = model.predict(data)

    #Y_hat_mean = np.mean(Y_hat)

    data['true_Y_hat'] = true_Y_hat
    #data['pred_Y_'+ str(int(A))] = Y_hat_pred
    data['pred_Y_A'] = Y_hat_pred
    data['pred_Y_0'] = Y_hat_0
    data['true_A'] = true_A

    data['Y_A'] = Y_A
    data['Y_0'] = Y_0

    data['Y_A_Y_0'] = data['Y_A'] - data['Y_0']
    data['pred_Y_A_pred_Y_0'] = data['pred_Y_A'] - data['pred_Y_0']

    return data[['true_A', 'true_Y_hat', 'pred_Y_A', 'pred_Y_0', 'Y_A', 'Y_0', 'Y_A_Y_0', 'pred_Y_A_pred_Y_0']]


def T_Learner_CATE(month, gen_autoreg_df_train, gen_autoreg_df_test, gen_autoreg_df_random_train, gen_autoreg_df_random_test):
    cols = ['Y_0', 'Y_1', 'Y_2', 'Y_3', 'Y_4', 'Y_5', 'Y_6', 'Y_7',
            'Y_hat', 'A_1', 'A_2', 'A_3', 'A_4', 'A_5', 'A_6', 'A_7', 'A_Cat']

    Y_col = 'Y_hat'

    print("DX Policy")
    A_models = {}
    for i in range(8):
        CATE_TLearner_model, CATE_TLearner_rmse = CATE_TLearner(
            gen_autoreg_df_train, month, i, config.cols_DX_Based + cols, config.prev_cols_DX_Based, Y_col, test_size=0.2)
        A_models[i] = CATE_TLearner_model
        # print("DX Policy CATE_TLearner_model.coeff_ action: ",
        #      i, " ", CATE_TLearner_model.coef_)

    print("Santiago Policy")
    A_models_random = {}
    for i in range(8):
        CATE_TLearner_model_random, CATE_TLearner_rmse_random = CATE_TLearner(
            gen_autoreg_df_random_train, month, i, config.cols_Santiago_Based + cols, config.prev_cols_Santiago_Based, Y_col, test_size=0.2)
        A_models_random[i] = CATE_TLearner_model_random
        # print("Santiago Policy CATE_TLearner_model.coeff_ action: ",
        #      i, " ", CATE_TLearner_model.coef_)

    pehes = {}
    pehes_random = {}

    for i in range(8):
        data = predict_CATE_TLearner(A_models[0], A_models[i], i, gen_autoreg_df_test,
                                     month, config.cols_DX_Based + cols, config.prev_cols_DX_Based, Y_col)
        print("DX: CATE TLearner (Real, Predicted) " + str(i) +
              " ", np.mean(data['Y_A']), np.mean(data['pred_Y_A']))
        pehe = mean_squared_error(data['Y_A_Y_0'], data['pred_Y_A_pred_Y_0'])

        pehes[i] = pehe

        # Santiago
        data_random = predict_CATE_TLearner(A_models_random[0], A_models_random[i], i, gen_autoreg_df_random_test,
                                            month, config.cols_Santiago_Based + cols, config.prev_cols_Santiago_Based, Y_col)
        print("Santiago: CATE TLearner (Real, Predicted) " + str(i) + " ",
              np.mean(data_random['Y_A']), np.mean(data_random['pred_Y_A']))
        pehe_random = mean_squared_error(
            data_random['Y_A_Y_0'], data_random['pred_Y_A_pred_Y_0'])
        pehes_random[i] = pehe_random

    return pehes, pehes_random


def CATE_SLearner(df, month, cols, prev_cols, Y_col, test_size=0.2):

    data = df.loc[df['VISCODE'] == int(month)][cols]

    for prev_col in prev_cols:
        data[prev_col + '_prev'] = df.loc[df['VISCODE'] ==
                                          int(month - 12)][prev_col].values  # Get from previous month

    data = dm.check_categorical(
        data, cols, Y_col, categorical_cols=config.Categorical_cols)
    data = data.dropna()

    # Shuffle the dataset.
    data_shuffled = data.sample(frac=1.0, random_state=0)

    for i in range(0, 8):
        data_shuffled = data_shuffled.drop('Y_' + str(int(i)), axis=1)

    # Split into input part X and output part Y.
    data_X = data_shuffled.drop(Y_col, axis=1)
    data_Y = data_shuffled[Y_col]

    # Partition the data into training and test sets.
    data_Xtrain, data_Xtest, data_Ytrain, data_Ytest = train_test_split(
        data_X, data_Y, test_size=test_size, random_state=0)

    Effect_estimators = copy.deepcopy(config.Effect_estimators)
    regr = Effect_estimators[config.regrr]
    regr.fit(data_Xtrain, data_Ytrain)

    data_Yguess = regr.predict(data_Xtest)
    rmse = np.sqrt(mean_squared_error(data_Yguess, data_Ytest))
    print("CATE SLearner " + Y_col + " RMSE: ", rmse,
          " R2  :", r2_score(data_Yguess, data_Ytest))

    return regr, rmse


def predict_CATE_SLearner(model, A, df, month, cols, prev_cols, Y_col):
    #a = 'A_' + str(int(A))
    data = df.loc[df['VISCODE'] == int(month)][cols]

    for prev_col in prev_cols:
        data[prev_col + '_prev'] = df.loc[df['VISCODE'] ==
                                          int(month - 12)][prev_col].values  # Get from previous month

    data = dm.check_categorical(
        data, cols, Y_col, categorical_cols=config.Categorical_cols)
    data = data.dropna()

    for i in range(1, 8):
        data['A_' + str(int(i))] = 0

    data0 = data.copy()

    if(A != 0):
        data['A_' + str(int(A))] = 1
    # print(data.head())

    Y_A = data['Y_' + str(int(A))].values
    Y_0 = data['Y_0'].values

    for i in range(0, 8):
        data = data.drop('Y_' + str(int(i)), axis=1)
        data0 = data0.drop('Y_' + str(int(i)), axis=1)

    #data = data.dropna()
    #data0= data0.dropna()
    # print(len(data))
    true_Y_hat = data['Y_hat'].values

    data = data.drop(Y_col, axis=1)
    data0 = data0.drop(Y_col, axis=1)

    Y_hat_0 = model.predict(data0)
    Y_hat_pred = model.predict(data)

    #Y_hat_mean = np.mean(Y_hat)
    data['true_Y_hat'] = true_Y_hat
    #data['pred_Y_'+ str(int(A))] = Y_hat_pred
    data['pred_Y_A'] = Y_hat_pred
    data['pred_Y_0'] = Y_hat_0

    data['Y_A'] = Y_A
    data['Y_0'] = Y_0

    data['Y_A_Y_0'] = data['Y_A'] - data['Y_0']
    data['pred_Y_A_pred_Y_0'] = data['pred_Y_A'] - data['pred_Y_0']

    return data[['true_Y_hat', 'pred_Y_A', 'pred_Y_0', 'Y_A', 'Y_0', 'Y_A_Y_0', 'pred_Y_A_pred_Y_0']]


def S_Learner_CATE(month, gen_autoreg_df_train, gen_autoreg_df_test, gen_autoreg_df_random_train, gen_autoreg_df_random_test):

    cols = ['Y_0', 'Y_1', 'Y_2', 'Y_3', 'Y_4', 'Y_5', 'Y_6', 'Y_7',
            'Y_hat', 'A_1', 'A_2', 'A_3', 'A_4', 'A_5', 'A_6', 'A_7', 'A_Cat']

    Y_col = 'Y_hat'

    print("DX Policy")
    CATESL_model, CATESL_rmse = CATE_SLearner(
        gen_autoreg_df_train, month, config.cols_DX_Based + cols, config.prev_cols_DX_Based, Y_col, test_size=0.2)

    print("Santiago Hernandez Policy")
    CATESL_model_random, CATESL_rmse_random = CATE_SLearner(
        gen_autoreg_df_random_train, month, config.cols_Santiago_Based + cols, config.prev_cols_Santiago_Based, Y_col, test_size=0.2)

    pehes = {}
    pehes_random = {}

    for i in range(8):
        data = predict_CATE_SLearner(CATESL_model, i, gen_autoreg_df_test,
                                     month, config.cols_DX_Based + cols, config.prev_cols_DX_Based, Y_col)

        pehe = mean_squared_error(data['Y_A_Y_0'], data['pred_Y_A_pred_Y_0'])
        pehes[i] = pehe

        # SH policy
        data_random = predict_CATE_SLearner(CATESL_model_random, i, gen_autoreg_df_random_test,
                                            month, config.cols_Santiago_Based + cols, config.prev_cols_Santiago_Based, Y_col)

        pehe_random = mean_squared_error(
            data_random['Y_A_Y_0'], data_random['pred_Y_A_pred_Y_0'])
        pehes_random[i] = pehe_random

    return pehes, pehes_random


def plot_cate(title, val1, val2, a):
    from matplotlib import ticker
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        val1, columns=['Model with Diagnosis-informed policy'], index=a)
    df['Model with Random Policy'] = val2

    plt.rcParams["figure.figsize"] = (12, 8)
    plt.style.use('tableau-colorblind10')

    df.plot(kind='bar')
    plt.title(title)
    plt.xlabel("Treatment")
    plt.ylabel("PEHE")
    plt.grid(color='k', linestyle='dotted')

    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.gca().xaxis.set_tick_params(rotation=0)


def plot_A_probs(gen_autoreg_df_test):
    gen_autoreg_df_test['proA_cat'] = gen_autoreg_df_test.groupby(
        'A_Cat')['A_Cat'].transform(lambda x: x.count() / len(gen_autoreg_df_test))
    group2 = gen_autoreg_df_test.groupby('A_Cat')
    A_cat_probs = pd.DataFrame(group2.apply(
        lambda x: x['proA_cat'].unique()[0]), columns=["prob"])
    A_cat_probs = A_cat_probs.reset_index(drop=True)
    gen_autoreg_df_test = gen_autoreg_df_test.drop('proA_cat', axis=1)
    A_cat_probs.plot(kind='bar')

    plt.savefig('plots/test_A_probs.png', format='png', dpi=500)


def plot_histogram(data):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1)
    nbins = 50

    plt.hist(data, nbins)

    plt.title("")
    plt.xlabel("p(T=t|X=x)")
    plt.grid()
    # plt.ylabel("Frequency")

    plt.show()

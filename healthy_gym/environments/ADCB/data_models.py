import warnings
import pandas as pd
import numpy as np
from numpy.random import choice
from collections import defaultdict
import copy
import pickle

from sklearn.mixture import GaussianMixture as GMM
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from scipy.stats import norm

from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV


from . import config

# plotting
import matplotlib.pyplot as plt
plt.rc('font', size=16, family='serif')
plt.grid(zorder=-100)
plt.switch_backend('agg')
plt.rcParams.update({'figure.max_open_warning': 0})

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

np.random.seed(config.rnd_seed)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')


class BLModel:
    def __init__(self):
        pass

    def fit(self, df, cols, Y_col, class_or_reg, test_size=0.2):
        data = df[cols]
        data = data.dropna()
        # Shuffle the dataset.
        data_shuffled = data.sample(frac=1.0, random_state=0)
        # Split into input part X and output part Y.
        data_X = data_shuffled.drop(Y_col, axis=1)
        data_Y = data_shuffled[Y_col]

        # Partition the data into training and test sets.
        data_Xtrain, data_Xtest, data_Ytrain, data_Ytest = train_test_split(
            data_X, data_Y, test_size=test_size, random_state=0)

        if(class_or_reg == 'CLASSIFICATION'):
            lr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
            lr.fit(data_Xtrain, data_Ytrain)
            # #print(data_Xtest.columns)
            data_Yguess = lr.predict(data_Xtest)
            # print(Y_col + " Accuracy  :", accuracy_score(data_Yguess, data_Ytest),
            # " F1 score: ", f1_score(data_Yguess, data_Ytest, average='weighted'))
            return lr
        elif(class_or_reg == 'REGRESSION'):
            lr = LinearRegression()
            # lr = GradientBoostingRegressor(random_state=0)
            lr.fit(data_Xtrain, data_Ytrain)
            # #print(data_Xtest.columns)
            data_Yguess = lr.predict(data_Xtest)
            rmse = np.sqrt(mean_squared_error(data_Yguess, data_Ytest))
            # print(Y_col + " RMSE: ", rmse, " R2  :",
            # r2_score(data_Yguess, data_Ytest))

            return lr, rmse


def fit_X1_model(df, Y_col, class_or_reg, test_size=0.2, return_metrics=False, grid_search=config.grid_search):
    data_Ytest_residuals = None
    all_pred_cols = copy.deepcopy(config.all_pred_cols)

    cols = all_pred_cols[Y_col]
    cols.append(Y_col)

    data = df[cols]
    data = check_categorical(
        data, cols, Y_col, categorical_cols=config.Categorical_cols)
    data = data.dropna()

    # print("\ncols: ",cols, "Y_col: ",Y_col)
    # print(Y_col, "\n", data.head())

    # Shuffle the dataset.
    data_shuffled = data.sample(frac=1.0, random_state=0)

    # Split into input part X and output part Y.
    data_X = data_shuffled.drop(Y_col, axis=1)
    data_Y = data_shuffled[Y_col]

    # print(data_X.head(),"\n", data_Y.head(), "\n", data_Y.shape)
    # print(data_X.columns,"\n", data_Y.shape)
    # data_y = data_Y.iloc[:, 0] if data_Y.shape[1]>0 else data_Y

    # Partition the data into training and test sets.
    data_Xtrain, data_Xtest, data_Ytrain, data_Ytest = train_test_split(
        data_X, data_Y, test_size=test_size, random_state=0)

    if(class_or_reg == 'CLASSIFICATION'):
        # print(data_Xtrain.shape, data_Ytrain.shape)
        # print(data_Xtrain.head(), data_Ytrain.head())

        if grid_search:
            clf_list = {}

            for k, v in config.clf_estimators.items():
                clf_list[k] = GridSearchCV(
                    v, config.clf_parameters_all[k], cv=3, scoring='accuracy', n_jobs=-1)

            # print("\n**" + Y_col)
            for k, clf in clf_list.items():
                clf.fit(data_Xtrain, data_Ytrain)

                data_Yguess_te = clf.predict(data_Xtest)
                data_Yguess_tr = clf.predict(data_Xtrain)

                n_classes = len(clf.classes_)

                acc_tr = accuracy_score(data_Ytrain, data_Yguess_tr)
                bacc_tr = balanced_accuracy_score(data_Ytrain, data_Yguess_tr)
                f1_tr = f1_score(data_Ytrain, data_Yguess_tr,
                                 average='weighted')

                corr_tr = 1 * (data_Ytrain == data_Yguess_tr)
                acc_std_tr = corr_tr.std()
                corr_te = 1 * (data_Ytest == data_Yguess_te)
                acc_std_te = corr_te.std()

                acc_te = accuracy_score(data_Ytest, data_Yguess_te)
                bacc_te = balanced_accuracy_score(data_Ytest, data_Yguess_te)
                f1_te = f1_score(data_Ytest, data_Yguess_te,
                                 average='weighted')

                # print("Classifier: " + str(k) + "\n" + Y_col + " Accuracy: %.2f" % acc_te,
                # " F1 score: %.2f" % f1_te,
                # " Balanced accuracy: %.2f" % bacc_te,
                # " Classes: %d" % n_classes)

                lr_results = {'clf': k, 'target': Y_col, 'acc_tr': acc_tr, 'acc_te': acc_te, 'acc_std_tr': acc_std_tr,
                              'acc_std_te': acc_std_te, 'bacc_tr': bacc_tr, 'bacc_te': bacc_te,
                              'f1_tr': f1_tr, 'f1_te': f1_te, 'n_classes': n_classes, 'n_train': data_Xtrain.shape[0],
                              'n_test': data_Xtest.shape[0]}

                # print("clf.best_params_", clf.best_params_)

            if not return_metrics:
                return clf_list['lr']

            else:
                return clf_list['lr'], lr_results
        else:
            # LogisticRegression(solver='lbfgs', multi_class='multinomial')
            lr = config.DGPcol_estimators[Y_col]
            lr.fit(data_Xtrain, data_Ytrain)
            # #print(data_Xtest.columns)
            data_Yguess_te = lr.predict(data_Xtest)
            data_Yguess_tr = lr.predict(data_Xtrain)

            n_classes = len(lr.classes_)

            acc_tr = accuracy_score(data_Ytrain, data_Yguess_tr)
            bacc_tr = balanced_accuracy_score(data_Ytrain, data_Yguess_tr)
            f1_tr = f1_score(data_Ytrain, data_Yguess_tr, average='weighted')

            corr_tr = 1 * (data_Ytrain == data_Yguess_tr)
            acc_std_tr = corr_tr.std()
            corr_te = 1 * (data_Ytest == data_Yguess_te)
            acc_std_te = corr_te.std()

            acc_te = accuracy_score(data_Ytest, data_Yguess_te)
            bacc_te = balanced_accuracy_score(data_Ytest, data_Yguess_te)
            f1_te = f1_score(data_Ytest, data_Yguess_te, average='weighted')

            # print(Y_col + " Accuracy: %.2f" % acc_te,
            # " F1 score: %.2f" % f1_te,
            # " Balanced accuracy: %.2f" % bacc_te,
            # " Classes: %d" % n_classes)

            lr_results = {'target': Y_col, 'acc_tr': acc_tr, 'acc_te': acc_te, 'acc_std_tr': acc_std_tr,
                          'acc_std_te': acc_std_te, 'bacc_tr': bacc_tr, 'bacc_te': bacc_te,
                          'f1_tr': f1_tr, 'f1_te': f1_te, 'n_classes': n_classes, 'n_train': data_Xtrain.shape[0],
                          'n_test': data_Xtest.shape[0]}
            with open(config.data_path + 'models/' + Y_col + '_model.pkl', 'wb') as f:
                pickle.dump(lr, f)

            if not return_metrics:
                return lr
            else:
                return lr, lr_results

    elif(class_or_reg == 'REGRESSION'):
        if grid_search:
            reg_list = {}

            for k, v in config.reg_estimators.items():
                reg_list[k] = GridSearchCV(
                    v, config.reg_parameters_all[k], cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

            # print("\n**" + Y_col)
            for k, reg in reg_list.items():
                reg.fit(data_Xtrain, data_Ytrain)

                data_Yguess_te = reg.predict(data_Xtest)
                data_Yguess_tr = reg.predict(data_Xtrain)

                data_Ytest_residuals = (data_Yguess_te - data_Ytest)
                # data_Ytest_residuals = (data_Yguess_tr - data_Ytrain)

                rmse_tr = np.sqrt(mean_squared_error(
                    data_Yguess_tr, data_Ytrain))
                r2_tr = r2_score(data_Yguess_tr, data_Ytrain)

                rmse_te = np.sqrt(mean_squared_error(
                    data_Yguess_te, data_Ytest))
                r2_te = r2_score(data_Yguess_te, data_Ytest)

                y_std = data_Ytest.std()
                res_std_tr = (data_Yguess_tr - data_Ytrain).std()
                res_std_te = (data_Yguess_te - data_Ytest).std()

                # print("Regressor: " + str(k) + "\n" + Y_col + " RMSE: %.2f" % rmse_te,
                # " R2: %.2f" % r2_te)

                lr_results = {'target': Y_col, 'rmse_tr': rmse_tr, 'rmse_te': rmse_te,
                              'r2_tr': r2_tr, 'r2_te': r2_te, 'y_std': y_std,
                              'res_std_tr': res_std_tr, 'res_std_te': res_std_te,
                              'n_train': data_Xtrain.shape[0], 'n_test': data_Xtest.shape[0]}

                # print("reg.best_params_", reg.best_params_)

            if not return_metrics:
                # ToDo: Return correct RMSE
                return reg_list['lr'], rmse_te

            else:
                return reg_list['lr'], rmse_te, data_Ytest_residuals, lr_results
        else:
            lr = config.DGPcol_estimators[Y_col]  # LinearRegression()
            lr.fit(data_Xtrain, data_Ytrain)

            data_Yguess_tr = lr.predict(data_Xtrain)
            data_Yguess_te = lr.predict(data_Xtest)

            data_Ytest_residuals = (data_Yguess_te - data_Ytest)
            # data_Ytest_residuals = (data_Yguess_tr - data_Ytrain)

            rmse_tr = np.sqrt(mean_squared_error(data_Yguess_tr, data_Ytrain))
            r2_tr = r2_score(data_Yguess_tr, data_Ytrain)

            rmse_te = np.sqrt(mean_squared_error(data_Yguess_te, data_Ytest))
            r2_te = r2_score(data_Yguess_te, data_Ytest)

            y_std = data_Ytest.std()
            res_std_tr = (data_Yguess_tr - data_Ytrain).std()
            res_std_te = (data_Yguess_te - data_Ytest).std()

            # print(Y_col + " RMSE: %.2f" % rmse_te,
            # " R2: %.2f" % r2_te)

            lr_results = {'target': Y_col, 'rmse_tr': rmse_tr, 'rmse_te': rmse_te,
                          'r2_tr': r2_tr, 'r2_te': r2_te, 'y_std': y_std,
                          'res_std_tr': res_std_tr, 'res_std_te': res_std_te,
                          'n_train': data_Xtrain.shape[0], 'n_test': data_Xtest.shape[0]}

            with open(config.data_path + 'models/' + Y_col + '_model.pkl', 'wb') as f:
                pickle.dump(lr, f)

            if not return_metrics:
                return lr, rmse_te
            else:
                return lr, rmse_te, data_Ytest_residuals, lr_results


def get_autoreg_keys(df, ids, autoreg_steps=config.autoreg_steps):
    # autoreg_steps = {12:(0, 12), 24:(12, 24), 36:(24, 36), 48:(36, 48),
    #             60:(48, 60), 72:(60, 72), 84:(72, 84), 96:(84, 96),
    #             108:(96, 108), 120:(108, 120)}

    vs_list = []
    for rid in ids:
        vs_list.append({rid: list(df.loc[df['RID'] == rid].VISCODE)})

    # #print(vs_list)
    autoreg_key_dict = {}
    for keyaut, step in autoreg_steps.items():
        # #print(keyaut)
        # #print(step)
        vs_keys = []
        for vs in vs_list:
            for keyvs, valuevs in vs.items():
                if step[0] in valuevs and step[1] in valuevs:
                    vs_keys.append(keyvs)

        autoreg_key_dict[keyaut] = vs_keys
    return autoreg_key_dict


def check_range(x):
    return 0 if x < 0 else x


def fit_for_NaNs(ADNI_DGP_NoNaNs_df, imputed_idxs_dict, month, col, predcols, model, class_or_reg):
    # print("******\n\n", col, "\n", predcols, model)
    if(len(imputed_idxs_dict[month]) > 0):
        if(class_or_reg == 'REGRESSION'):
            df = ADNI_DGP_NoNaNs_df[ADNI_DGP_NoNaNs_df['RID'].isin(
                imputed_idxs_dict[month])]
            df = df.loc[df['VISCODE'] == month]
            row_ids = list(df.row_id)

            # #print(model["model"])
            # #print(model["RMSE"])
            # print("df[predcols].head() \n", df[predcols].head())

            df_pred = check_categorical(df[predcols], predcols, col)
            # print("df_pred.head() \n", df_pred.head())

            pred = model["model"].predict(
                df_pred) + np.random.normal(0, model["RMSE"], size=(len(df))).reshape(-1)
            pred = [check_range(x) for x in pred]

            # #print(pred)
            # print(len(ADNI_DGP_NoNaNs_df.loc[ADNI_DGP_NoNaNs_df['row_id'].isin(row_ids), col]))
            ADNI_DGP_NoNaNs_df.loc[ADNI_DGP_NoNaNs_df['row_id'].isin(
                row_ids), col] = pred

        elif(class_or_reg == 'CLASSIFICATION'):
            df = ADNI_DGP_NoNaNs_df[ADNI_DGP_NoNaNs_df['RID'].isin(
                imputed_idxs_dict[month])]
            df = df.loc[df['VISCODE'] == month]
            row_ids = list(df.row_id)

            df_pred = check_categorical(df[predcols], predcols, col)
            pred = [int(choice(len(x), 1, p=x))
                    for x in model["model"].predict_proba(df_pred)]
            # #print(pred)
            # print(len(ADNI_DGP_NoNaNs_df.loc[ADNI_DGP_NoNaNs_df['row_id'].isin(row_ids), col]))
            ADNI_DGP_NoNaNs_df.loc[ADNI_DGP_NoNaNs_df['row_id'].isin(
                row_ids), col] = pred

    return ADNI_DGP_NoNaNs_df


def ADAS13_cleanup(x):
    if x > 85:
        x = 85
    if x < 0:
        x = 0
    return x


def CDRSB_cleanup(x):
    if x > 18:
        x = 18
    if x < 0:
        x = 0
    return round(x * 2) / 2


def MMSE_cleanup(x):
    if x > 30:
        x = 30
    if x < 0:
        x = 0
    return round(x)


def standardize_x(df1, cols, Y_col, continuous_cols=config.continuous_cols):

    standardize_cols = [c for c in cols if(
        ((c in continuous_cols)) and (c != Y_col))]

    df = df1.copy()
    sc = StandardScaler()
    df[standardize_cols] = sc.fit_transform(df[standardize_cols])

    return df


def check_categorical(df, cols, Y_col, categorical_cols=config.Categorical_cols):
    for col in cols:
        # print("\ncheck_categorical: ", col)
        # OR any(s in col for s in categorical_cols)
        if(((col in categorical_cols)) and (col != Y_col)):
            # print("\ncheck_categorical,  True: ", col)
            df = df.astype({col: int})
            # #print(df.head())
            df = pd.concat([df, pd.get_dummies(df[col].astype(pd.CategoricalDtype(
                categories=list(range(categorical_cols[col])))), prefix=col, drop_first=True)], axis=1)
            # df = pd.concat([df, pd.get_dummies(df[col], prefix=col, drop_first=True)], axis=1)
            df.drop([col], axis=1, inplace=True)
    return df

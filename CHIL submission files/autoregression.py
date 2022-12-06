import config
import data_models as dm
import treatments as tr
import warnings
import sys
import pandas as pd
import numpy as np
import pickle
from numpy.random import choice
from collections import defaultdict
import copy
from scipy.stats import skewnorm


from sklearn.utils import shuffle
from sklearn.mixture import GaussianMixture as GMM
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.preprocessing import LabelEncoder

from scipy.stats import norm

from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV


# plotting
import matplotlib.pyplot as plt
plt.rc('font', size=16, family='serif')
plt.grid(zorder=-100)
plt.switch_backend('agg')
plt.rcParams.update({'figure.max_open_warning': 0})

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

np.random.seed(config.rnd_seed)


# Fitting $X_1$
def gen_potential_Outcomes(observed_a, Y_hat, ADAS13, Z, gamma_0, gamma_1, P1):
    Y_s = {}
    for a in range(8):
        if(a == observed_a):
            Y = Y_hat
        else:
            delta = tr.assign_treatment_effect(a, Z, gamma_0, gamma_1, P1)
            Y = ADAS13 + delta
        Y_s[a] = Y
    return (Y_s[0], Y_s[1], Y_s[2], Y_s[3], Y_s[4], Y_s[5], Y_s[6], Y_s[7])


def sample_noise(ADNI_bl, col, n):
    # a is the skewness parameter
    a, loc, scale = skewnorm.fit(ADNI_bl[col].values)
    rv = skewnorm(a)
    mean = (rv.mean() * scale) + loc
    noise = ((rv.rvs(size=n) * scale) + loc) - mean

    return noise


def sample_asymetric_AGE(N):
    # These have been fixed for convenience. Fitted from original ADNI data
    a, loc, scale = -0.6783052361536166, 76.18409213099177, 7.897017603971423
    rv = skewnorm(a)
    val = ((rv.rvs(size=N) * scale) + loc)  # - mean
    return val


def sample_asymetric(covariate_vals, n, log=False):
    loc, scale = 0, 0
    if(log):
        s, loc, scale = lognorm.fit(covariate_vals)
        rv = lognorm(s)
    else:
        a, loc, scale = skewnorm.fit(covariate_vals)
        rv = skewnorm(a)

    #mean= (rv.mean()*scale) + loc
    val = ((rv.rvs(size=n) * scale) + loc)  # - mean

    return val

# Autoregression


def fit_auto_regressor(df, autoreg_key_lists, autoreg_steps, Y_col, class_or_reg, month=12, test_size=0.2, return_metrics=True, grid_search=config.grid_search):

    all_pred_cols = copy.deepcopy(config.all_pred_cols)
    cols = all_pred_cols[Y_col]
    cols.append(Y_col)

    data = pd.DataFrame({})
    cols.append('RID')

    # Iterate over months
    for month in [12, 24, 36, 48]:
        autoreg_month_values = autoreg_key_lists[month]

        df1 = df[(df['RID'].isin(autoreg_month_values)) & (
            df['VISCODE'] == autoreg_steps[month][0])][['RID', Y_col]]
        df1 = df1.rename({Y_col: Y_col + '_prev'}, axis=1)
        df2 = df[(df['RID'].isin(autoreg_month_values)) & (
            df['VISCODE'] == autoreg_steps[month][1])][cols]
        df2 = df2.rename({Y_col: Y_col + '_curr'}, axis=1)

        df3 = pd.merge(df1, df2, on='RID')
        data = pd.concat([data, df3], axis=0)

    data = data.drop('RID', axis=1)
    data = data.dropna()

    # Shuffle the dataset.
    data_shuffled = data.sample(frac=1.0, random_state=0)

    # Split into input part X and output part Y.
    data_X = data_shuffled.drop(Y_col + '_curr', axis=1)

    # dummyfy
    data_Y = data_shuffled[Y_col + '_curr']

    data_X = dm.check_categorical(data_X, list(
        data_X.columns), '_', categorical_cols=config.Categorical_cols)
    #print(Y_col, " :", list(data_X.columns))
    # Partition the data into training and test sets.
    data_Xtrain, data_Xtest, data_Ytrain, data_Ytest = train_test_split(
        data_X, data_Y, test_size=test_size, random_state=0)
    #print(len(data_Xtrain), len(data_Xtest))

    if(class_or_reg == 'CLASSIFICATION'):

        if grid_search:
            clf_list = {}

            for k, v in config.clf_estimators.items():
                clf_list[k] = GridSearchCV(
                    v, config.clf_parameters_all[k], cv=3, scoring='accuracy', n_jobs=-1)

            print("\n**" + Y_col)
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

                print("Classifier: " + str(k) + "\n" + Y_col + " Autoregression Accuracy: %.2f" % acc_te,
                      " F1 score: %.2f" % f1_te,
                      " Balanced accuracy: %.2f" % bacc_te,
                      " Classes: %d" % n_classes)

                lr_results = {'clf': k, 'target': Y_col, 'acc_tr': acc_tr, 'acc_te': acc_te, 'acc_std_tr': acc_std_tr,
                              'acc_std_te': acc_std_te, 'bacc_tr': bacc_tr, 'bacc_te': bacc_te,
                              'f1_tr': f1_tr, 'f1_te': f1_te, 'n_classes': n_classes, 'n_train': data_Xtrain.shape[0],
                              'n_test': data_Xtest.shape[0]}

                print("clf.best_params_", clf.best_params_)

            if not return_metrics:
                return clf_list['lr']

            else:
                return clf_list['lr'], lr_results
        else:
            # LogisticRegression(solver='lbfgs', multi_class='multinomial')
            lr = config.DGPcol_estimators[Y_col]
            lr.fit(data_Xtrain, data_Ytrain)
            # print(data_Xtest.columns)

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

            print(Y_col + " Month ", month, " Autoregression Accuracy  :",
                  acc_te, " F1 score: ", f1_te)

            lr_results = {'target': Y_col, 'acc_tr': acc_tr, 'acc_te': acc_te, 'acc_std_tr': acc_std_tr,
                          'acc_std_te': acc_std_te, 'bacc_tr': bacc_tr, 'bacc_te': bacc_te,
                          'f1_tr': f1_tr, 'f1_te': f1_te, 'n_classes': n_classes, 'n_train': data_Xtrain.shape[0],
                          'n_test': data_Xtest.shape[0]}

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

            print("\n**" + Y_col)
            for k, reg in reg_list.items():
                reg.fit(data_Xtrain, data_Ytrain)

                data_Yguess_te = reg.predict(data_Xtest)
                data_Yguess_tr = reg.predict(data_Xtrain)

                rmse_tr = np.sqrt(mean_squared_error(
                    data_Yguess_tr, data_Ytrain))
                r2_tr = r2_score(data_Yguess_tr, data_Ytrain)

                rmse_te = np.sqrt(mean_squared_error(
                    data_Yguess_te, data_Ytest))
                r2_te = r2_score(data_Yguess_te, data_Ytest)

                y_std = data_Ytest.std()
                res_std_tr = (data_Yguess_tr - data_Ytrain).std()
                res_std_te = (data_Yguess_te - data_Ytest).std()

                print("Regressor: " + str(k) + "\n" + Y_col + "Autoregression RMSE: %.2f" % rmse_te,
                      " R2: %.2f" % r2_te)

                lr_results = {'target': Y_col, 'rmse_tr': rmse_tr, 'rmse_te': rmse_te,
                              'r2_tr': r2_tr, 'r2_te': r2_te, 'y_std': y_std,
                              'res_std_tr': res_std_tr, 'res_std_te': res_std_te,
                              'n_train': data_Xtrain.shape[0], 'n_test': data_Xtest.shape[0]}

                print("reg.best_params_", reg.best_params_)

            if not return_metrics:
                # ToDo: Return correct RMSE
                return reg_list['lr'], rmse_te

            else:
                return reg_list['lr'], rmse_te, lr_results
        else:
            lr = config.DGPcol_estimators[Y_col]
            lr.fit(data_Xtrain, data_Ytrain)
            # print(data_Xtest.columns)

            data_Yguess_tr = lr.predict(data_Xtrain)
            data_Yguess_te = lr.predict(data_Xtest)

            rmse_tr = np.sqrt(mean_squared_error(data_Yguess_tr, data_Ytrain))
            r2_tr = r2_score(data_Yguess_tr, data_Ytrain)

            rmse_te = np.sqrt(mean_squared_error(data_Yguess_te, data_Ytest))
            r2_te = r2_score(data_Yguess_te, data_Ytest)

            y_std = data_Ytest.std()
            res_std_tr = (data_Yguess_tr - data_Ytrain).std()
            res_std_te = (data_Yguess_te - data_Ytest).std()

            print(Y_col + " Month ", month, " Autoregression RMSE: ",
                  rmse_te, " R2  :",  r2_te)

            lr_results = {'target': Y_col, 'rmse_tr': rmse_tr, 'rmse_te': rmse_te,
                          'r2_tr': r2_tr, 'r2_te': r2_te, 'y_std': y_std,
                          'res_std_tr': res_std_tr, 'res_std_te': res_std_te,
                          'n_train': data_Xtrain.shape[0], 'n_test': data_Xtest.shape[0]}

            if not return_metrics:
                return lr, rmse_te
            else:
                return lr, rmse_te, lr_results


# Generating samples of X1
def generate_N_Samples_bl(num_repetition, N, P1, PTETHCAT_probs, PTRACCAT_probs, PTGENDER_probs, AB_ratio_bl_df, gamma_0, gamma_1, epsilon=0.1, return_metrics=config.return_metrics):
    np.random.seed(num_repetition)

    gen_data = []
    residuals = {}

    lrAPOE4, resAPOE4 = dm.fit_X1_model(
        AB_ratio_bl_df, 'APOE4', 'CLASSIFICATION', test_size=0.2, return_metrics=True)
    lrEDUCAT, resEDUCAT = dm.fit_X1_model(
        AB_ratio_bl_df, 'PTEDUCAT', 'CLASSIFICATION', test_size=0.2, return_metrics=True)
    lrMARRY, resMARRY = dm.fit_X1_model(
        AB_ratio_bl_df, 'PTMARRY', 'CLASSIFICATION', test_size=0.2, return_metrics=True)
    lrTAU, TAURMSE, residuals['TAU'], resTAU = dm.fit_X1_model(
        AB_ratio_bl_df, 'TAU', 'REGRESSION', test_size=0.2, return_metrics=True)
    lrPTAU, PTAURMSE, residuals['PTAU'], resPTAU = dm.fit_X1_model(
        AB_ratio_bl_df,  'PTAU', 'REGRESSION', test_size=0.2, return_metrics=True)
    lrFDG, FDGRMSE, residuals['FDG'], resFDG = dm.fit_X1_model(
        AB_ratio_bl_df,  'FDG', 'REGRESSION', test_size=0.2, return_metrics=True)
    lrAV45, AV45RMSE, residuals['AV45'], resAV45 = dm.fit_X1_model(
        AB_ratio_bl_df, 'AV45', 'REGRESSION', test_size=0.2, return_metrics=True)
    lrADAS13, ADAS13RMSE, residuals['ADAS13'], resADAS13 = dm.fit_X1_model(
        AB_ratio_bl_df, 'ADAS13', 'REGRESSION', test_size=0.2, return_metrics=True)
    lrMMSE, MMSERMSE, residuals['MMSE'], resMMSE = dm.fit_X1_model(
        AB_ratio_bl_df, 'MMSE', 'REGRESSION', test_size=0.2, return_metrics=True)
    lrCDRSB, CDRSBRMSE, residuals['CDRSB'], resCDRSB = dm.fit_X1_model(
        AB_ratio_bl_df, 'CDRSB', 'REGRESSION', test_size=0.2, return_metrics=True)
    lrDX, resDX = dm.fit_X1_model(
        AB_ratio_bl_df, 'DX', 'CLASSIFICATION', test_size=0.2, return_metrics=True)

    with open(config.residuals_file, 'wb') as f:
        pickle.dump(residuals, f)

    # Store results of baseline model fitting
    if return_metrics:
        resClassifiers = pd.DataFrame([resAPOE4, resEDUCAT, resMARRY, resDX])
        resRegressors = pd.DataFrame(
            [resTAU, resPTAU, resFDG, resAV45, resADAS13, resCDRSB, resMMSE])
        resClassifiers.to_csv('data/resClassifiers_bl.csv',
                              encoding='utf-8', index=False)
        resRegressors.to_csv('data/resRegressors_bl.csv',
                             encoding='utf-8', index=False)
        baseline_results = [resClassifiers, resRegressors]

    # load gmm model for sampling Z
    with open('data/fitted_gmm_Z.pickle', 'rb') as f:
        fitted_gmm_Z = pickle.load(f)

    AGES = sample_asymetric_AGE(N)

    ABETARatio_samples, Z_samples = fitted_gmm_Z.sample(N)
    ABETARatio_samples, Z_samples = shuffle(ABETARatio_samples, Z_samples)

    # Sample noises and save them
    noises = {}
    noise_cols = ['TAU', 'PTAU', 'FDG', 'AV45', 'MMSE', 'ADAS13', 'CDRSB']
    for col in noise_cols:
        # sample_noise(AB_ratio_bl_df, col, N)
        noises[col] = sample_asymetric(residuals[col].values, N)

    # Generate baseline samples
    for i in range(N):
        AGE = '{0:.1f}'.format(AGES[i])

        # Sample AbetaRatio and Z from fitted GMM
        ABETARatio = ABETARatio_samples[i][0]
        Z = Z_samples[i]  # int(np.random.binomial(size=1, n=1, p=P1))

        PTETHCAT = choice(list(PTETHCAT_probs.PTETHCAT),
                          1, p=list(PTETHCAT_probs.prob))
        PTRACCAT = choice(list(PTRACCAT_probs.PTRACCAT),
                          1, p=list(PTRACCAT_probs.prob))
        PTGENDER = choice(list(PTGENDER_probs.PTGENDER),
                          1, p=list(PTGENDER_probs.prob))

        ddict = {'PTETHCAT': PTETHCAT,
                 'PTRACCAT': PTRACCAT, 'PTGENDER': PTGENDER}

        APOE4 = lrAPOE4.predict_proba(dm.check_categorical(
            pd.DataFrame(ddict), config.all_pred_cols['APOE4'], 'APOE4'))
        APOE4 = choice(len(APOE4[0]), 1, p=list(APOE4[0]))[0]

        #ddict = {'PTETHCAT': PTETHCAT, 'PTRACCAT': PTRACCAT, 'PTGENDER': PTGENDER}

        PTEDUCAT = lrEDUCAT.predict_proba(dm.check_categorical(
            pd.DataFrame(ddict), config.all_pred_cols['PTEDUCAT'], 'PTEDUCAT'))
        PTEDUCAT = choice(len(PTEDUCAT[0]), 1, p=list(PTEDUCAT[0]))[0]
        #PTEDUCAT = choice(np.unique(AB_ratio_bl_df.PTEDUCAT), 1, p=list(PTEDUCAT[0]))[0]

        PTMARRY = lrMARRY.predict_proba(dm.check_categorical(pd.DataFrame(
            {'PTGENDER': PTGENDER}), config.all_pred_cols['PTMARRY'], 'PTMARRY'))
        PTMARRY = choice(len(PTMARRY[0]), 1, p=list(PTMARRY[0]))[0]

        ddict = {'PTETHCAT': PTETHCAT, 'PTRACCAT': PTRACCAT, 'PTGENDER': PTGENDER,
                 'Z': Z, 'APOE4': APOE4, 'AGE': AGE, 'ABETARatio': ABETARatio}
        TAU = lrTAU.predict(dm.check_categorical(pd.DataFrame(
            ddict), config.all_pred_cols['TAU'], 'TAU'))[0] + noises['TAU'][i]
        while TAU < 0:
            TAU = lrTAU.predict(dm.check_categorical(pd.DataFrame(ddict), config.all_pred_cols['TAU'], 'TAU'))[
                0] + sample_asymetric(residuals['TAU'].values, 1)[0]  # sample_noise(AB_ratio_bl_df, 'TAU', 1)[0]

        PTAU = lrPTAU.predict(dm.check_categorical(pd.DataFrame(
            ddict), config.all_pred_cols['PTAU'], 'PTAU'))[0] + noises['PTAU'][i]
        while PTAU < 0:
            PTAU = lrPTAU.predict(dm.check_categorical(pd.DataFrame(ddict), config.all_pred_cols['PTAU'], 'PTAU'))[
                0] + sample_asymetric(residuals['PTAU'].values, 1)[0]  # sample_noise(AB_ratio_bl_df, 'PTAU', 1)[0]

        ddict = {'PTETHCAT': PTETHCAT, 'PTRACCAT': PTRACCAT, 'Z': Z,
                 'TAU': TAU, 'PTAU': PTAU, 'APOE4': APOE4, 'ABETARatio': ABETARatio}
        FDG = lrFDG.predict(dm.check_categorical(pd.DataFrame(
            ddict), config.all_pred_cols['FDG'], 'FDG'))[0] + noises['FDG'][i]
        while FDG < 0:
            FDG = lrFDG.predict(dm.check_categorical(pd.DataFrame(ddict), config.all_pred_cols['FDG'], 'FDG'))[
                0] + sample_asymetric(residuals['FDG'].values, 1)[0]  # sample_noise(AB_ratio_bl_df, 'FDG', 1)[0]

        AV45 = lrAV45.predict(dm.check_categorical(pd.DataFrame(
            ddict), config.all_pred_cols['AV45'], 'AV45'))[0] + noises['AV45'][i]
        while AV45 < 0:
            AV45 = lrAV45.predict(dm.check_categorical(pd.DataFrame(ddict), config.all_pred_cols['AV45'], 'AV45'))[
                0] + sample_asymetric(residuals['AV45'].values, 1)[0]  # sample_noise(AB_ratio_bl_df, 'AV45', 1)[0]

        # @ TODO FJ: Adapt to asymmetric noise ^ #Newton Done
        ddict = {'PTETHCAT': PTETHCAT, 'PTRACCAT': PTRACCAT,  'PTEDUCAT': PTEDUCAT, 'PTGENDER': PTGENDER, 'PTMARRY': PTMARRY,
                 'Z': Z, 'TAU': TAU, 'PTAU': PTAU, 'APOE4': APOE4, 'FDG': FDG, 'AV45': AV45, 'ABETARatio': ABETARatio}
        ADAS13 = lrADAS13.predict(dm.check_categorical(pd.DataFrame(
            ddict), config.all_pred_cols['ADAS13'], 'ADAS13'))[0] + noises['ADAS13'][i]
        while ADAS13 < 0:
            ADAS13 = lrADAS13.predict(dm.check_categorical(pd.DataFrame(ddict), config.all_pred_cols['ADAS13'], 'ADAS13'))[
                0] + sample_asymetric(residuals['ADAS13'].values, 1)[0]  # sample_noise(AB_ratio_bl_df, 'ADAS13', 1)[0]
        if ADAS13 > 85:
            ADAS13 = 85

        ddict = {'PTETHCAT': PTETHCAT, 'PTRACCAT': PTRACCAT,  'PTEDUCAT': PTEDUCAT, 'PTGENDER': PTGENDER, 'PTMARRY': PTMARRY,
                 'Z': Z, 'TAU': TAU, 'PTAU': PTAU, 'APOE4': APOE4, 'FDG': FDG, 'AV45': AV45, 'ADAS13': ADAS13, 'ABETARatio': ABETARatio}
        MMSE = lrMMSE.predict(dm.check_categorical(pd.DataFrame(
            ddict), config.all_pred_cols['MMSE'], 'MMSE'))[0] + noises['MMSE'][i]
        while MMSE < 0:
            MMSE = lrADAS13.predict(dm.check_categorical(pd.DataFrame(ddict), config.all_pred_cols['MMSE'], 'MMSE'))[
                0] + sample_asymetric(residuals['MMSE'].values, 1)[0]  # sample_noise(AB_ratio_bl_df, 'MMSE', 1)[0]
        if MMSE > 30:
            MMSE = 30

        CDRSB = lrCDRSB.predict(dm.check_categorical(pd.DataFrame(
            ddict), config.all_pred_cols['CDRSB'], 'CDRSB'))[0] + noises['CDRSB'][i]
        while CDRSB < 0:
            CDRSB = lrCDRSB.predict(dm.check_categorical(pd.DataFrame(ddict), config.all_pred_cols['CDRSB'], 'CDRSB'))[
                0] + sample_asymetric(residuals['CDRSB'].values, 1)[0]  # sample_noise(AB_ratio_bl_df, 'CDRSB', 1)[0]
        if CDRSB > 18:
            CDRSB = 18

        A = 0
        delta = tr.assign_treatment_effect(A, Z, gamma_0, gamma_1, P1)
        #['PTGENDER', 'PTEDUCAT', 'PTETHCAT', 'PTRACCAT', 'PTMARRY', 'APOE4', 'FDG', 'AV45', 'TAU', 'PTAU', 'DX', 'MMSE']
        Y_hat = ADAS13 + delta

        Y_0, Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, Y_7 = gen_potential_Outcomes(
            A, Y_hat, ADAS13, Z, gamma_0, gamma_1, P1)

        ddict = {'PTETHCAT': PTETHCAT, 'PTRACCAT': PTRACCAT, 'PTGENDER': PTGENDER, 'Z': Z,
                 'TAU': TAU, 'PTAU': PTAU, 'APOE4': APOE4, 'FDG': FDG, 'AV45': AV45, 'ADAS13': ADAS13}
        #DX = lrDX.predict(pd.DataFrame(ddict))[0]
        DX = lrDX.predict_proba(dm.check_categorical(
            pd.DataFrame(ddict), config.all_pred_cols['DX'], 'DX'))
        DX = choice(len(DX[0]), 1, p=list(DX[0]))[0]

        ddict = {'RID': i, 'AGE': AGE, 'VISCODE': 0, 'PTETHCAT': PTETHCAT[0], 'PTRACCAT': PTRACCAT[0],
                 'PTMARRY': PTMARRY, 'PTEDUCAT': PTEDUCAT, 'PTGENDER': PTGENDER[0], 'Z': Z, 'ABETARatio': ABETARatio,
                 'TAU': TAU, 'PTAU': PTAU, 'APOE4': APOE4, 'FDG': FDG, 'AV45': AV45, 'ADAS13': ADAS13, 'MMSE': MMSE,
                 'CDRSB': CDRSB, 'DX': DX, 'A': A, 'Delta': delta, 'Y_hat': Y_hat,
                 'Y_0': Y_0, 'Y_1': Y_1, 'Y_2': Y_2, 'Y_3': Y_3, 'Y_4': Y_4, 'Y_5': Y_5, 'Y_6': Y_6, 'Y_7': Y_7}
        gen_data.append(ddict)

    gen_data_df = pd.DataFrame(gen_data)

    if not return_metrics:
        return gen_data_df
    else:
        return gen_data_df, baseline_results


# Generating autoregression samples
def gen_autoregression_samples(N, num_repetition, AB_ratio_bl_df, gen_data, df, autoreg_key_lists,
                               autoreg_steps, num_steps, gamma_0, gamma_1, P1, epsilon=0.1, policy='DX', return_metrics=config.return_metrics):
    np.random.seed(num_repetition)
    last_month = num_steps * 12  # + 12

    df2 = pd.DataFrame()
    df2 = df2.append(gen_data[['RID', 'AGE', 'PTETHCAT', 'PTRACCAT', 'PTGENDER', 'APOE4', 'PTEDUCAT', 'PTMARRY', 'TAU', 'PTAU', 'FDG', 'AV45', 'Z', 'ABETARatio',
                     'VISCODE', 'ADAS13', 'MMSE', 'CDRSB', 'DX', 'A', 'Delta', 'Y_hat', 'Y_0', 'Y_1', 'Y_2', 'Y_3', 'Y_4', 'Y_5', 'Y_6', 'Y_7']], ignore_index=True)

    # Sample noises and save them
    residuals = None
    # load gresiduals for sampling noises
    with open(config.residuals_file, 'rb') as f:
        residuals = pickle.load(f)
    noises = {}
    noise_cols = ['TAU', 'PTAU', 'FDG', 'AV45', 'MMSE', 'ADAS13', 'CDRSB']
    AGE = gen_data.AGE
    df_age = pd.DataFrame(AGE, columns={"AGE"})

    Z = gen_data.Z
    ABETARatio = gen_data.ABETARatio.values
    #print('\nABETARatio', ABETARatio)
    PTETHCAT = gen_data.PTETHCAT
    PTRACCAT = gen_data.PTRACCAT
    PTGENDER = gen_data.PTGENDER
    RID = gen_data.RID.values

    lrAPOE4_auto, resAPOE4 = fit_auto_regressor(
        df, autoreg_key_lists, autoreg_steps, 'APOE4', 'CLASSIFICATION', return_metrics=True)
    lrEDUCAT_auto, resEDUCAT = fit_auto_regressor(
        df, autoreg_key_lists, autoreg_steps, 'PTEDUCAT', 'CLASSIFICATION', return_metrics=True)
    lrMARRY_auto, resMARRY = fit_auto_regressor(
        df, autoreg_key_lists, autoreg_steps, 'PTMARRY', 'CLASSIFICATION')
    lrTAU_auto, TAURMSE, resTAU = fit_auto_regressor(
        df, autoreg_key_lists, autoreg_steps, 'TAU', 'REGRESSION', return_metrics=True)
    lrPTAU_auto, PTAURMSE, resPTAU = fit_auto_regressor(
        df, autoreg_key_lists, autoreg_steps, 'PTAU', 'REGRESSION', return_metrics=True)
    lrFDG_auto, FDGRMSE, resFDG = fit_auto_regressor(
        df, autoreg_key_lists, autoreg_steps, 'FDG', 'REGRESSION', return_metrics=True)
    lrAV45_auto, AV45RMSE, resAV45 = fit_auto_regressor(
        df, autoreg_key_lists, autoreg_steps, 'AV45', 'REGRESSION', return_metrics=True)
    lrADAS13_auto, ADAS13RMSE, resADAS13 = fit_auto_regressor(
        df, autoreg_key_lists, autoreg_steps, 'ADAS13', 'REGRESSION', return_metrics=True)
    lrCDRSB_auto, CDRSBRMSE, resCDRSB = fit_auto_regressor(
        df, autoreg_key_lists, autoreg_steps, 'CDRSB', 'REGRESSION', return_metrics=True)
    lrMMSE_auto, MMSERMSE, resMMSE = fit_auto_regressor(
        df, autoreg_key_lists, autoreg_steps, 'MMSE', 'REGRESSION', return_metrics=True)
    lrDX_auto, resDX = fit_auto_regressor(
        df, autoreg_key_lists, autoreg_steps, 'DX', 'CLASSIFICATION', return_metrics=True)

    # Store results of autoregression model fitting
    if return_metrics:
        resClassifiers = pd.DataFrame([resAPOE4, resEDUCAT, resMARRY, resDX])
        resRegressors = pd.DataFrame(
            [resTAU, resPTAU, resFDG, resAV45, resADAS13, resMMSE, resCDRSB])
        resClassifiers.to_csv('data/resClassifiers_auto_' + str(policy) + '.csv',
                              encoding='utf-8', index=False)
        resRegressors.to_csv('data/resRegressors_auto_' + str(policy) + '.csv',
                             encoding='utf-8', index=False)
        autoreg_results = [resClassifiers, resRegressors]

    for month in range(12, last_month, 12):
        #print("month: ", month)
        noises = {}
        for col in noise_cols:
            # sample_noise(AB_ratio_bl_df, col, N)
            noises[col] = sample_asymetric(residuals[col].values, N)

        df_age['AGE'] = df_age['AGE'].map(lambda x: float(x) + 1)
        #print("AGE:", df_age.head())
        AGE = df_age.AGE.values

        pred_df = df2.loc[df2['VISCODE'] == (
            month - 12)][['APOE4', 'PTETHCAT', 'PTRACCAT', 'PTGENDER']]
        APOE4 = lrAPOE4_auto.predict_proba(
            dm.check_categorical(pred_df, list(pred_df.columns), '_'))
        APOE4 = [int(choice(len(x), 1, p=x)) for x in APOE4]

        pred_df = df2.loc[df2['VISCODE'] == (
            month - 12)][['PTEDUCAT', 'PTETHCAT', 'PTRACCAT', 'PTGENDER']]
        PTEDUCAT = lrEDUCAT_auto.predict_proba(
            dm.check_categorical(pred_df, list(pred_df.columns), '_'))
        PTEDUCAT = [int(choice(len(x), 1, p=x)) for x in PTEDUCAT]

        pred_df = df2.loc[df2['VISCODE'] == (
            month - 12)][['PTMARRY', 'PTGENDER']]
        PTMARRY = lrMARRY_auto.predict_proba(
            dm.check_categorical(pred_df, list(pred_df.columns), '_'))
        PTMARRY = [int(choice(len(x), 1, p=x)) for x in PTMARRY]

        df_past = df2.loc[df2['VISCODE'] == (
            month - 12)][['TAU', 'PTETHCAT', 'PTRACCAT', 'PTGENDER', 'Z']]
        df_past['APOE4'] = APOE4
        df_past['AGE'] = AGE
        df_past['ABETARatio'] = ABETARatio
        # + np.random.normal(0, TAURMSE, size=(len(df2.loc[df2['VISCODE'] == (month-12)]))).reshape(-1)
        TAU = lrTAU_auto.predict(dm.check_categorical(
            df_past, list(df_past.columns), '_')) + noises['TAU']
        TAU = [check_range(x) for x in TAU]

        df_past = df2.loc[df2['VISCODE'] == (
            month - 12)][['PTAU', 'PTETHCAT', 'PTRACCAT', 'PTGENDER', 'Z']]
        df_past['APOE4'] = APOE4
        df_past['AGE'] = AGE
        df_past['ABETARatio'] = ABETARatio
        # + np.random.normal(0, PTAURMSE, size=(len(df2.loc[df2['VISCODE'] == (month-12)]))).reshape(-1)
        PTAU = np.array(lrPTAU_auto.predict(dm.check_categorical(
            df_past, list(df_past.columns), '_'))) + noises['PTAU']
        PTAU = [check_range(x) for x in PTAU]

        df_past = df2.loc[df2['VISCODE'] == (
            month - 12)][['FDG', 'PTETHCAT', 'PTRACCAT', 'Z']]
        df_past['TAU'] = TAU
        df_past['PTAU'] = PTAU
        df_past['APOE4'] = APOE4
        df_past['ABETARatio'] = ABETARatio
        # + np.random.normal(0, FDGRMSE, size=(len(df2.loc[df2['VISCODE'] == (month-12)]))).reshape(-1)
        FDG = lrFDG_auto.predict(dm.check_categorical(
            df_past, list(df_past.columns), '_')) + noises['FDG']
        FDG = [check_range(x) for x in FDG]

        df_past = df2.loc[df2['VISCODE'] == (
            month - 12)][['AV45', 'PTETHCAT', 'PTRACCAT', 'Z']]
        df_past['TAU'] = TAU
        df_past['PTAU'] = PTAU
        df_past['APOE4'] = APOE4
        df_past['ABETARatio'] = ABETARatio
        # + np.random.normal(0, AV45RMSE, size=(len(df2.loc[df2['VISCODE'] == (month-12)]))).reshape(-1)
        AV45 = lrAV45_auto.predict(dm.check_categorical(
            df_past, list(df_past.columns), '_')) + noises['AV45']
        AV45 = [check_range(x) for x in AV45]
        # @ TODO FJ: Adapt to asymmetric noise ^

        df_past = df2.loc[df2['VISCODE'] == (
            month - 12)][['ADAS13', 'PTETHCAT', 'PTRACCAT', 'PTEDUCAT', 'PTGENDER', 'Z']]
        df_past['PTMARRY'] = PTMARRY
        df_past['TAU'] = TAU
        df_past['PTAU'] = PTAU
        df_past['APOE4'] = APOE4
        df_past['FDG'] = FDG
        df_past['AV45'] = AV45
        df_past['ABETARatio'] = ABETARatio

        # + np.random.normal(0, ADAS13RMSE/2, size=(len(df2.loc[df2['VISCODE'] == (month-12)]))).reshape(-1) #+ noises['ADAS13']
        ADAS13 = lrADAS13_auto.predict(dm.check_categorical(
            df_past, list(df_past.columns), '_')) + noises['ADAS13']

        df_past = df2.loc[df2['VISCODE'] == (
            month - 12)][['MMSE', 'PTETHCAT', 'PTRACCAT', 'PTEDUCAT', 'PTGENDER', 'Z']]
        df_past['PTMARRY'] = PTMARRY
        df_past['TAU'] = TAU
        df_past['PTAU'] = PTAU
        df_past['APOE4'] = APOE4
        df_past['FDG'] = FDG
        df_past['AV45'] = AV45
        df_past['ADAS13'] = ADAS13
        df_past['ABETARatio'] = ABETARatio
        # + np.random.normal(0, MMSERMSE, size=(len(df2.loc[df2['VISCODE'] == (month-12)]))).reshape(-1)
        MMSE = lrMMSE_auto.predict(dm.check_categorical(
            df_past, list(df_past.columns), '_')) + noises['MMSE']

        df_past = df2.loc[df2['VISCODE'] == (
            month - 12)][['CDRSB', 'PTETHCAT', 'PTRACCAT', 'PTEDUCAT', 'PTGENDER', 'Z']]
        df_past['PTMARRY'] = PTMARRY
        df_past['TAU'] = TAU
        df_past['PTAU'] = PTAU
        df_past['APOE4'] = APOE4
        df_past['FDG'] = FDG
        df_past['AV45'] = AV45
        df_past['ADAS13'] = ADAS13
        df_past['ABETARatio'] = ABETARatio
        # + np.random.normal(0, CDRSBRMSE, size=(len(df2.loc[df2['VISCODE'] == (month-12)]))).reshape(-1)
        CDRSB = lrCDRSB_auto.predict(dm.check_categorical(
            df_past, list(df_past.columns), '_')) + noises['CDRSB']

        ddict = {'RID': RID, 'AGE': AGE, 'PTETHCAT': PTETHCAT, 'PTRACCAT': PTRACCAT, 'PTGENDER': PTGENDER, 'APOE4': APOE4, 'PTEDUCAT': PTEDUCAT,
                 'PTMARRY': PTMARRY, 'TAU': TAU, 'PTAU': PTAU, 'FDG': FDG, 'AV45': AV45, 'Z': Z, 'ABETARatio': ABETARatio, 'ADAS13': ADAS13, 'MMSE': MMSE, 'CDRSB': CDRSB}

        #length_dict = {key: len(value) for key, value in ddict.items()}
        gen_data_auto = pd.DataFrame(ddict)
        gen_data_auto['ADAS13'] = gen_data_auto['ADAS13'].map(
            lambda x: dm.ADAS13_cleanup(x))
        gen_data_auto['MMSE'] = gen_data_auto['MMSE'].map(
            lambda x: dm.MMSE_cleanup(x))
        gen_data_auto['CDRSB'] = gen_data_auto['CDRSB'].map(
            lambda x: dm.CDRSB_cleanup(x))

        # print(length_dict)
        prev_dx = list(df2.loc[df2['VISCODE'] == (month - 12)]['DX'])
        prev_A = list(df2.loc[df2['VISCODE'] == (month - 12)]['A'])

        gen_data_auto_copy = gen_data_auto.copy()
        gen_data_auto_copy['prev_A'] = prev_A
        gen_data_auto_copy['prev_DX'] = prev_dx
        gen_data_auto_copy['RACE'] = gen_data_auto_copy.apply(
            lambda x: race(PTRACCAT=x['PTRACCAT'], PTETHCAT=x['PTETHCAT']), axis=1)
        standardscaler = MinMaxScaler()  # StandardScaler()

        standard_df = standardscaler.fit_transform(
            gen_data_auto_copy[["AGE", "PTEDUCAT", "MMSE", "CDRSB", 'ADAS13']])
        standard_df = pd.DataFrame(standard_df, columns=[
                                   "AGE", "PTEDUCAT", "MMSE", "CDRSB", 'ADAS13'])

        gen_data_auto_copy[["AGE", "PTEDUCAT", "MMSE", "CDRSB", 'ADAS13']] = standard_df[[
            "AGE", "PTEDUCAT", "MMSE", "CDRSB", 'ADAS13']]
        if (policy == 'DX_Based'):
            gen_data_auto['A'] = [tr.assign_treatment_DX(
                x, epsilon, policy) for x in prev_dx]
        else:
            #assign_treatment(DX, AGE, GENDER, MARRIED, EDUCATION, MMSE, CDRSB, prev_A, epsilon, policy)
            gen_data_auto['A'] = gen_data_auto_copy.apply(lambda x: tr.assign_treatment_Santiago(RACE=x['RACE'], AGE=x['AGE'], GENDER=x['PTGENDER'],
                                                          MARRIED=x['PTMARRY'], EDUCATION=x['PTEDUCAT'], MMSE=x['MMSE'], CDRSB=x['CDRSB'],
                                                          prev_A=x['prev_A'], epsilon=epsilon, policy=policy), axis=1)

        gen_data_auto['Delta'] = gen_data_auto.apply(lambda x: tr.assign_treatment_effect(
            a=x['A'], Z=x['Z'], gamma_0=gamma_0, gamma_1=gamma_1, P_1=P1), axis=1)
        gen_data_auto['Y_hat'] = gen_data_auto.apply(
            lambda x: x['ADAS13'] + x['Delta'], axis=1)

        YY = pd.DataFrame(np.array(list(gen_data_auto.apply(lambda x: gen_potential_Outcomes(
            observed_a=x['A'], Y_hat=x['Y_hat'], ADAS13=x['ADAS13'], Z=x['Z'], gamma_0=gamma_0, gamma_1=gamma_1, P1=P1), axis=1))),
            columns=['Y_0', 'Y_1', 'Y_2', 'Y_3', 'Y_4', 'Y_5', 'Y_6', 'Y_7'])

        gen_data_auto['Y_0'] = YY['Y_0'].values
        gen_data_auto['Y_1'] = YY['Y_1'].values
        gen_data_auto['Y_2'] = YY['Y_2'].values
        gen_data_auto['Y_3'] = YY['Y_3'].values
        gen_data_auto['Y_4'] = YY['Y_4'].values
        gen_data_auto['Y_5'] = YY['Y_5'].values
        gen_data_auto['Y_6'] = YY['Y_6'].values
        gen_data_auto['Y_7'] = YY['Y_7'].values

        dx_df = df2.loc[df2['VISCODE'] == (
            month - 12)][['DX', 'PTETHCAT', 'PTRACCAT',  'PTGENDER', 'Z', 'TAU', 'PTAU', 'APOE4', 'FDG', 'AV45']]
        dx_df['ADAS13'] = gen_data_auto['Y_hat'].values

        DX = lrDX_auto.predict_proba(
            dm.check_categorical(dx_df, list(dx_df.columns), '_'))
        DX = [int(choice(len(x), 1, p=x)) for x in DX]

        gen_data_auto['DX'] = DX

        gen_data_auto['VISCODE'] = month

        df2 = df2.append(gen_data_auto, ignore_index=True)

    if not return_metrics:
        return df2
    else:
        return df2, autoreg_results


def check_range(x):
    return 0 if x < 0 else x


def plot_statistics(ADNI_DGP_NoNaNs_df, gen_autoreg_df):
    data_cols = ['VISCODE', 'PTGENDER', 'PTEDUCAT', 'PTETHCAT', 'PTRACCAT',
                 'PTMARRY', 'APOE4', 'FDG', 'AV45', 'TAU', 'PTAU', 'Z', 'DX', 'ADAS13', 'MMSE', 'CDRSB']

    orig_data = ADNI_DGP_NoNaNs_df[data_cols]

    gen_data = gen_autoreg_df[data_cols]

    try:
        for month in range(0, 132, 12):
            for col in data_cols:
                fig = plt.figure(figsize=(15, 4))
                #x = fig.add_subplot(1,1,1)
                nbins = 100

                plt.subplot(1, 2, 1)
                plt.hist(
                    np.array(orig_data.loc[orig_data['VISCODE'] == month][col]), nbins)
                plt.title("" + col + " original data, month " + str(month))
                plt.xlabel(col)
                plt.grid()

                plt.subplot(1, 2, 2)
                plt.hist(
                    np.array(gen_data.loc[gen_data['VISCODE'] == month][col]), nbins)
                plt.title("" + col + " generated data, month " + str(month))
                plt.xlabel(col)
                plt.grid()

                plt.savefig('plots/generated_' + col + '_' +
                            str(month) + '.png', format='png', dpi=500)
                # plt.show()

        data_colsg = ['A']  # 'Delta', 'Y_hat']
        for month in range(12, 132, 12):
            for col in data_colsg:
                fig = plt.figure(figsize=(15, 4))
                #x = fig.add_subplot(1,1,1)
                nbins = 100

                plt.subplot(1, 2, 1)
                plt.hist(
                    np.array(gen_autoreg_df.loc[gen_autoreg_df['VISCODE'] == month][col]), nbins)
                plt.title(
                    "" + col + " generated data with covariate treatment assignment policy, month " + str(month))
                plt.xlabel(col)
                plt.grid()

                plt.savefig('plots/generated_' + col + '_' +
                            str(month) + '.png', format='png', dpi=500)
                # @TODO: Change to .pdf

                # plt.show()

        data_colsg = ['A']  # 'Delta', 'Y_hat']
        for month in range(12, 132, 12):
            for col in data_colsg:
                fig = plt.figure(figsize=(15, 4))
                #x = fig.add_subplot(1,1,1)
                nbins = 100

                plt.subplot(1, 2, 1)
                plt.hist(np.array(
                    gen_autoreg_df_random.loc[gen_autoreg_df_random['VISCODE'] == month][col]), nbins)
                plt.title(
                    "" + col + " generated data with random treatment assignment policy, month " + str(month))
                plt.xlabel(col)
                plt.grid()

                plt.savefig('plots/generated_' + col + '_' +
                            str(month) + '.png', format='png', dpi=500)
                # plt.show()
    except:
        pass


def race(PTRACCAT, PTETHCAT):
    if (int(PTRACCAT) == 0):
        race = 'White'
    elif (int(PTRACCAT) == 1 and int(PTETHCAT) != 0):
        race = 'Black'
    elif (int(PTETHCAT) == 0):
        race = 'Non-Black Hispanic'
    else:
        race = 'Other'
    return race

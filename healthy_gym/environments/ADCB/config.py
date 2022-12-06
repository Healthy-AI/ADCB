import os
import warnings
import pandas as pd
import numpy as np
from numpy.random import choice
from collections import defaultdict

from sklearn.mixture import GaussianMixture as GMM
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso

from scipy.stats import norm

from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

# plotting
import matplotlib.pyplot as plt
plt.rc('font', size=16, family='serif')
plt.grid(zorder=-100)
plt.switch_backend('agg')
plt.rcParams.update({'figure.max_open_warning': 0})

rnd_seed = 0
np.random.seed(rnd_seed)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

here = os.path.dirname(os.path.abspath(__file__))

data_path = ABETA_filespath = os.path.join(here, './data/')
# Filepaths
ADNIFilepath = os.path.join(here, './data/ADNIMERGE.csv')

ABETA_filespath = os.path.join(here, './data/ABETA_files/')

# gen_file_prefix_policy1 = 'data/gen_DX_Based_'
# gen_file_prefix_policy2 = 'data/gen_Santiago_Based_'

ADAS13RMSE_file = os.path.join(here, 'data/ADAS13RMSE_file.pickle')
residuals_file = os.path.join(here, 'data/residuals_file.pickle')
metrics_results_file = os.path.join(here, 'data/metrics_results_file.pickle')
fitted_gmm_Z_file = os.path.join(here, 'data/fitted_gmm_Z.pickle')

adas_noise_bl = 6.316
adas_noise_ar = 4.423

TAU_NOISE = 127.09
PTAU_NOISE = 14.14
FDG_NOISE = 0.15
AV45_NOISE = 0.22
MMSE_NOISE = 2.60
ADAS_NOISE = 9.58

Autoreg_TAU_NOISE = 129.76
Autoreg_PTAU_NOISE = 13.90
Autoreg_FDG_NOISE = 0.14
Autoreg_AV45_NOISE = 0.21
Autoreg_MMSE_NOISE = 2.47
Autoreg_ADAS_NOISE = 9.56

DGPcols = ['RID', 'AGE', 'VISCODE', 'PTGENDER', 'PTEDUCAT', 'PTETHCAT', 'PTRACCAT', 'PTMARRY', 'APOE4',
           'FDG', 'AV45', 'ABETA', 'ABETA40', 'ABETA42', 'TAU', 'PTAU', 'DX', 'MMSE', 'ADAS13', 'CDRSB'
           ]
#
Categorical_cols = {'Z': 6, 'PTGENDER': 2, 'PTEDUCAT': 13, 'PTETHCAT': 3, 'PTRACCAT': 7, 'PTMARRY': 5, 'APOE4': 3, 'DX': 3,  'PTGENDER_prev': 2,
                    'PTEDUCAT_prev': 13, 'PTETHCAT_prev': 3, 'PTRACCAT_prev': 7, 'PTMARRY_prev': 5, 'APOE4_prev': 3, 'DX_prev': 3, 'A_Cat_prev': 8
                    }

continuous_cols = ['AGE', 'FDG', 'AV45',  'TAU',
                   'PTAU', 'MMSE', 'ADAS13', 'CDRSB', 'ABETARatio']


latent_dim = 6

# autoreg_steps = {12:(0, 12), 24:(12, 24), 36:(24, 36), 48:(36, 48),
#                 60:(48, 60), 72:(60, 72), 84:(72, 84), 96:(84, 96),
#                 108:(96, 108), 120:(108, 120)}

autoreg_steps = {12: (0, 12), 24: (12, 24), 36: (24, 36), 48: (36, 48),
                 60: (36, 48), 72: (36, 48), 84: (36, 48), 96: (36, 48),
                 108: (36, 48), 120: (36, 48)
                 }

months = {'bl': 0, 'm12': 12, 'm24': 24, 'm36': 36, 'm48': 48, 'm60': 60, 'm72': 72, 'm84': 84, 'm96': 96,
          'm108': 108, 'm120': 120
          }

EDUCAT_YEAR_map = {8: 0, 9: 1, 10: 2, 11: 3, 12: 4, 13: 5,
                   14: 6, 15: 7, 16: 8, 17: 9, 18: 10, 19: 11, 20: 12}

DX_Codes = {
    'CN': int(0),
    'MCI': int(1),
    'Dementia': int(2)
}

PTRACCAT_Codes = {
    'White': 0,
    'Black': 1,
    'More than one': 2,
    'Am Indian/Alaskan': 3,
    'Asian': 4,
    'Hawaiian/Other PI': 5,
    'Unknown': 6
}

PTETHCAT_Codes = {
    'Hisp/Latino': 0,
    'Not Hisp/Latino': 1,
    'Unknown': 2
}

PTMARRY_Codes = {
    'Married': 0,
    'Never married': 1,
    'Widowed': 2,
    'Divorced': 3,
    'Unknown': 4
}

N = 10000  # Number of patient trajectories
epsilon = 0.1
gamma = 2.0
month = 60
history = 3
num_steps = 11
bool_train = False
unconfounded = True
num_repetitions = 10
return_metrics = False

grid_search = False

# Look into expanding age and education according to Santiago et. al 2010 in future iteration
OR_AchEI = {
    'intercept': 6.0,  # 6
    'gender': 1.30,
    'race_W': 1,
    'race_B': 0.59,
    'race_NBH': 0.8,
    'age': 0.98,
    'education_l4': 1,
    'education_4_8': 0.97,
    'education_g8': 1,
    'marriage': 1.21,
    'MMSE': 0.99,
    'CDR': 1.21,
}

OR_Memantine = {
    'intercept': 0.22,  # 0.22
    'gender': 1,
    'race_W': 1,
    'race_B': 0.43,
    'race_NBH': 0.69,
    'age': 0.99,
    'education_l4': 1,
    'education_4_8': 1.19,
    'education_g8': 1.38,
    'marriage': 1.41,
    'MMSE': 0.97,
    'CDR': 1.45,
    'prev_AchEI': 4.58,
}

A_Delta = {
    0: 0,
    1: -1.95,
    2: -2.48,
    3: -3.03,
    4: -3.20,
    5: -2.01,
    6: -1.29,
    7: -2.69
}

effect_coeffs = [2, 3, 4, 2.5, 5, 6]
"""beta = np.array([[0, -1, -1,  1,  0,  1],
                 [1,  0, -1, -1,  1,  0],
                 [-1,  1,  0,  1,  0, -1],
                 [-1,  0,  1,  0,  1, -1],
                 [1, -1,  0,  0,  1, -1],
                 [1, -1, -1, -1,  1,  1],
                 [0,  1,  0,  1, -1, -1],
                 [1,  0,  0,  0, -1,  0]])
"""

beta = np.array([[0., -1.,  0.5, -2.,  1.,  1.5],
                 [-2., -1.,  2.,  1.5,  1.5, -2.],
                 [-1.,  0.5, -2.,  2., -1.,  1.5],
                 [-1.,  0.,  0.5,  0.5,  2., -2.],
                 [1.,  0.5, -1., -2.,  1.5,  0.],
                 [-2., -1., -1.,  1.,  2.,  1.],
                 [-2., -2.,  0.5,  0.,  1.5,  2.],
                 [0.5,  1., -2.,  0., -1.,  1.5]])

imputation_cols = ['TAU', 'PTAU', 'FDG',
                   'AV45', 'ADAS13', 'MMSE', 'CDRSB', 'DX']

if(latent_dim == 6):
    all_pred_cols = {
        'PTMARRY': ['PTGENDER'],
        'PTEDUCAT': ['PTETHCAT', 'PTRACCAT', 'PTGENDER'],
        'TAU':  ['PTETHCAT', 'PTRACCAT', 'PTGENDER', 'Z', 'AGE'],
        'PTAU':  ['PTETHCAT', 'PTRACCAT', 'PTGENDER', 'Z', 'AGE'],
        'FDG':  ['PTETHCAT', 'PTRACCAT', 'Z', 'TAU', 'PTAU'],
        'AV45':  ['PTETHCAT', 'PTRACCAT', 'Z', 'TAU', 'PTAU'],
        'ADAS13': ['PTETHCAT', 'PTRACCAT', 'PTEDUCAT', 'PTGENDER', 'PTMARRY', 'Z', 'TAU', 'PTAU',  'FDG', 'AV45'],
        'MMSE': ['PTETHCAT', 'PTRACCAT', 'PTEDUCAT', 'PTGENDER', 'PTMARRY', 'Z', 'TAU', 'PTAU',  'FDG', 'AV45', 'ADAS13'],
        'CDRSB': ['PTETHCAT', 'PTRACCAT', 'PTEDUCAT', 'PTGENDER', 'PTMARRY', 'Z', 'TAU', 'PTAU',  'FDG', 'AV45', 'ADAS13'],
        'DX': ['PTETHCAT', 'PTRACCAT',  'PTGENDER', 'Z', 'TAU', 'PTAU',  'FDG', 'AV45', 'ADAS13']
    }
else:
    all_pred_cols = {
        'PTMARRY': ['PTGENDER'],
        'PTEDUCAT': ['PTETHCAT', 'PTRACCAT', 'PTGENDER'],
        'APOE4': ['PTETHCAT', 'PTRACCAT', 'PTGENDER'],
        'TAU':  ['PTETHCAT', 'PTRACCAT', 'PTGENDER', 'Z', 'APOE4', 'AGE', 'ABETARatio'],
        'PTAU':  ['PTETHCAT', 'PTRACCAT', 'PTGENDER', 'Z', 'APOE4', 'AGE', 'ABETARatio'],
        'FDG':  ['PTETHCAT', 'PTRACCAT', 'Z', 'TAU', 'PTAU', 'APOE4', 'ABETARatio'],
        'AV45':  ['PTETHCAT', 'PTRACCAT', 'Z', 'TAU', 'PTAU', 'APOE4', 'ABETARatio'],
        'ADAS13': ['PTETHCAT', 'PTRACCAT', 'PTEDUCAT', 'PTGENDER', 'PTMARRY', 'Z', 'TAU', 'PTAU', 'APOE4', 'FDG', 'AV45', 'ABETARatio'],
        'MMSE': ['PTETHCAT', 'PTRACCAT', 'PTEDUCAT', 'PTGENDER', 'PTMARRY', 'Z', 'TAU', 'PTAU', 'APOE4', 'FDG', 'AV45', 'ADAS13', 'ABETARatio'],
        'CDRSB': ['PTETHCAT', 'PTRACCAT', 'PTEDUCAT', 'PTGENDER', 'PTMARRY', 'Z', 'TAU', 'PTAU', 'APOE4', 'FDG', 'AV45', 'ADAS13', 'ABETARatio'],
        'DX': ['PTETHCAT', 'PTRACCAT',  'PTGENDER', 'Z', 'TAU', 'PTAU', 'APOE4', 'FDG', 'AV45', 'ADAS13']
    }
# Adjustment sets for CATE Estimators
cols_DX_Based = ['PTETHCAT', 'PTRACCAT', 'PTEDUCAT',
                 'PTGENDER', 'PTMARRY', 'TAU', 'PTAU', 'APOE4', 'FDG', 'AV45']  # , 'DX']
cols_Santiago_Based = ['AGE', 'PTETHCAT', 'PTRACCAT', 'PTEDUCAT', 'PTGENDER',
                       'PTMARRY', 'TAU', 'PTAU', 'APOE4', 'FDG', 'AV45', 'MMSE', 'CDRSB']  # , 'DX']

# Adjustment sets for CATE Estimators
cols_DX_Based_seq = ['PTETHCAT', 'PTRACCAT', 'PTEDUCAT',
                     'PTGENDER', 'PTMARRY', 'TAU', 'PTAU', 'APOE4', 'FDG', 'AV45']
cols_Santiago_Based_seq = ['AGE', 'PTETHCAT', 'PTRACCAT', 'PTEDUCAT', 'PTGENDER',
                           'PTMARRY', 'TAU', 'PTAU', 'APOE4', 'FDG', 'AV45', 'MMSE', 'CDRSB']

prev_cols_DX_Based = ['DX', 'Y_hat']
prev_cols_Santiago_Based = ['Y_hat']

# list of classifiers
clf_lr = LogisticRegression()
clf_rfc = RandomForestClassifier()
clf_gxb = GradientBoostingClassifier()
clf_knn = KNeighborsClassifier()
clf_svm = SVC()

clf_estimators = {
    'lr': clf_lr,
    'rfc': clf_rfc,
    'gxb': clf_gxb,
    'knn': clf_knn,
    # 'svm': clf_svm
}


# parameter options for all calssifiers
clf_parameters_all = {

    'lr': [{
        'solver': ['lbfgs', 'newton-cg', 'sag', 'saga']
    }],

    'rfc': [{'n_estimators': list(range(1, 310, 50)),
             # 'min_samples_leaf': list(range(1, 60, 20)),
             'max_depth': list(range(2, 20, 4))
             # 'max_features': ['auto', 'sqrt'],
             # 'bootstrap': [True, False],
             # 'criterion':('entropy', 'gini')
             }],

    'gxb': [{
            # "loss":["deviance"],
            # "learning_rate": [0.01, 0.05, 0.1],
            # "min_samples_split": np.linspace(0.1, 0.5, 5),
            # "min_samples_leaf": np.linspace(0.1, 0.5, 5),
            "max_depth": list(range(2, 20, 4)),
            # "max_features":["log2", "sqrt"],
            # "criterion": ["friedman_mse",  "mae"],
            # "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
            "n_estimators": list(range(1, 310, 50))
            }],

    'knn': [{
        'n_neighbors': [5, 10, 15],
        'weights': ["uniform", "distance"]
    }],

    # 'svm' :[{'decision_function_shape':'ovo'}],
}

# list of regressors
reg_lr = LinearRegression()
reg_rfr = RandomForestRegressor()
reg_gxb = GradientBoostingRegressor()
reg_knn = KNeighborsRegressor()
reg_svm = SVR()

reg_estimators = {
    'lr': reg_lr,
    'rfr': reg_rfr,
    'gxb': reg_gxb,
    'knn': reg_knn,
    # 'svm': reg_svm
}

# parameter options for all regressors
reg_parameters_all = {
    'lr': [{
        "fit_intercept": [True, False]
    }],

    'rfr': [{'n_estimators': list(range(1, 310, 50)),
             # 'min_samples_leaf': list(range(1, 60, 20)),
             'max_depth': list(range(2, 20, 4))
             # 'max_features': ['auto', 'sqrt'],
             # 'bootstrap': [True, False],
             # 'criterion':('entropy', 'gini')
             }],

    'gxb': [{
            # "loss":["deviance"],
            # "learning_rate": [0.01, 0.05, 0.1],
            # "min_samples_split": np.linspace(0.1, 0.5, 5),
            # "min_samples_leaf": np.linspace(0.1, 0.5, 5),
            "max_depth": list(range(2, 20, 4)),
            # "max_features":["log2", "sqrt"],
            # "criterion": ["friedman_mse",  "mae"],
            # "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
            "n_estimators": list(range(1, 310, 50))
            }],
    'knn': [{
        'n_neighbors': [5, 10, 15],
        'weights': ["uniform", "distance"]
    }],
    # 'svm' :[{'decision_function_shape':'ovo'}],
}

DGPcol_estimators = {
    'PTMARRY': LogisticRegression(solver='lbfgs'),
    'PTEDUCAT': LogisticRegression(solver='lbfgs'),
    'APOE4': KNeighborsClassifier(n_neighbors=10, weights='uniform'),
    'TAU':  RandomForestRegressor(max_depth=2, n_estimators=201),
    'PTAU':  RandomForestRegressor(max_depth=2, n_estimators=51),
    'FDG':  GradientBoostingRegressor(max_depth=2, n_estimators=51),
    'AV45':  RandomForestRegressor(max_depth=2, n_estimators=151),
    'ADAS13': RandomForestRegressor(max_depth=6, n_estimators=251),
    'MMSE': GradientBoostingRegressor(max_depth=2, n_estimators=51),
    'CDRSB': GradientBoostingRegressor(max_depth=2, n_estimators=51),
    'DX': LogisticRegression(solver='lbfgs')
}

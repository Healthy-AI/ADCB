import config
import pandas as pd
import numpy as np
#from sklearn.utils.extmath import softmax
from numpy import exp

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')

np.random.seed(config.rnd_seed)


def treatment_effects(tau, P_1, gamma_0, gamma_1, level):
    if level == 'L0':
        L = tau / (P_1 + (gamma_1 * (1 - P_1)))
    elif level == 'H0':
        L = (gamma_0 * tau) / (P_1 + (gamma_0 * (1 - P_1)))
    elif level == 'L1':
        L = tau / (P_1 + (gamma_0 * (1 - P_1)))
    elif level == 'H1':
        L = (gamma_1 * tau) / ((1 - P_1) + (gamma_1 * (P_1)))
    return L


# Modelling treatments
def assign_treatment_effect(a, Z, gamma_0, gamma_1, P_1):
    A_Delta = [
        {
            0: 0,
            1: treatment_effects(-1.95, P_1, gamma_0, gamma_1, 'L0'),
            2: treatment_effects(-2.48, P_1, gamma_0, gamma_1, 'L0'),
            3: treatment_effects(-3.03, P_1, gamma_0, gamma_1, 'H0'),
            4: treatment_effects(-3.20, P_1, gamma_0, gamma_1, 'H0'),
            5: treatment_effects(-2.01, P_1, gamma_0, gamma_1, 'L0'),
            6: treatment_effects(-1.29, P_1, gamma_0, gamma_1, 'H0'),
            7: treatment_effects(-2.69, P_1, gamma_0, gamma_1, 'L0')

        },  # Z=0

        {
            0: 0,
            1: treatment_effects(-1.95, P_1, gamma_0, gamma_1, 'H1'),
            2: treatment_effects(-2.48, P_1, gamma_0, gamma_1, 'H1'),
            3: treatment_effects(-3.03, P_1, gamma_0, gamma_1, 'L1'),
            4: treatment_effects(-3.20, P_1, gamma_0, gamma_1, 'L1'),
            5: treatment_effects(-2.01, P_1, gamma_0, gamma_1, 'H1'),
            6: treatment_effects(-1.29, P_1, gamma_0, gamma_1, 'L1'),
            7: treatment_effects(-2.69, P_1, gamma_0, gamma_1, 'H1')
        }  # Z=1
    ]
    delta = A_Delta[int(Z)][int(a)]
    return delta


def assign_treatment_DX(DX, epsilon, policy):
    treatments_by_effects = [[0], [1, 2, 5, 6], [3, 4, 7]]
    p = np.random.random()
    if p > epsilon:
        a = np.random.choice(treatments_by_effects[DX])
    else:
        a = np.random.choice(8)
    return a


def assign_treatment_Santiago(RACE, AGE, GENDER, MARRIED, EDUCATION, MMSE, CDRSB, prev_A, epsilon, policy):
    treatments_by_type = [[0], [1, 2, 3, 4, 5], [6], [7]]
    p = np.random.random()
    c_AchEI, c_Memantine = compute_Cs(
        RACE, AGE, GENDER, MARRIED, EDUCATION, MMSE, CDRSB, prev_A, epsilon)

    # if p > epsilon:
    if(c_AchEI == 0 and c_Memantine == 0):
        a = 0
    elif(c_AchEI == 1 and c_Memantine == 0):
        a = np.random.choice(treatments_by_type[1])
    elif(c_AchEI == 0 and c_Memantine == 1):
        a = 6
    elif(c_AchEI == 1 and c_Memantine == 1):
        a = 7
    # else:
    #    a = np.random.choice(8)

    return a


def compute_Cs(RACE, AGE, GENDER, MARRIED, EDUCATION, MMSE, CDRSB, prev_A, epsilon=0.1):
    #print('RACE, AGE, GENDER, MARRIED, EDUCATION, MMSE, CDRSB, prev_A: ', RACE, AGE, GENDER, MARRIED, EDUCATION, MMSE, CDRSB, prev_A)

    bool_prev_AchEI = 0

    if(prev_A in list(range(1, 6))):
        bool_prev_AchEI = 1

    if(MARRIED == 0):
        married_coeff_AchEI = config.OR_AchEI['marriage']
        married_coeff_Memantine = config.OR_Memantine['marriage']
    else:
        married_coeff_AchEI = 1
        married_coeff_Memantine = 1

    if(GENDER == 1):
        gender_coeff_AchEI = config.OR_AchEI['gender']
        gender_coeff_Memantine = config.OR_Memantine['gender']
    else:
        gender_coeff_AchEI = 1
        gender_coeff_Memantine = 1

    if(RACE == 'White'):
        race_coeff_AchEI = config.OR_AchEI['race_W']
        race_coeff_Memantine = config.OR_Memantine['race_W']
    elif(RACE == 'Black'):
        race_coeff_AchEI = config.OR_AchEI['race_B']
        race_coeff_Memantine = config.OR_Memantine['race_B']
    elif(RACE == 'Non-Black Hispanic'):
        race_coeff_AchEI = config.OR_AchEI['race_NBH']
        race_coeff_Memantine = config.OR_Memantine['race_NBH']
    else:
        race_coeff_AchEI = config.OR_AchEI['race_NBH']
        race_coeff_Memantine = config.OR_Memantine['race_NBH']
    if(EDUCATION < 4):
        edu_coeff_AchEI = config.OR_AchEI['education_l4']
        edu_coeff_Memantine = config.OR_Memantine['education_l4']
    elif(EDUCATION >= 4 and EDUCATION <= 8):
        edu_coeff_AchEI = config.OR_AchEI['education_4_8']
        edu_coeff_Memantine = config.OR_Memantine['education_4_8']
    elif(EDUCATION > 8):
        edu_coeff_AchEI = config.OR_AchEI['education_g8']
        edu_coeff_Memantine = config.OR_Memantine['education_g8']
    else:
        edu_coeff_AchEI = config.OR_AchEI['education_l4']
        edu_coeff_Memantine = config.OR_Memantine['education_l4']

    logit_AchEI = np.log(np.log(config.OR_AchEI['intercept'] + race_coeff_AchEI + gender_coeff_AchEI + config.OR_AchEI['age'] * float(
        AGE) + edu_coeff_AchEI + married_coeff_AchEI + config.OR_AchEI['MMSE'] * float(MMSE) + config.OR_AchEI['CDR'] * float(CDRSB) + 1))

    logit_Memantine = np.log(np.log(config.OR_Memantine['intercept'] + race_coeff_Memantine + gender_coeff_Memantine + config.OR_Memantine['age'] * float(AGE) + edu_coeff_Memantine +
                             married_coeff_Memantine + config.OR_Memantine['MMSE'] * float(MMSE) + config.OR_Memantine['CDR'] * float(CDRSB) + (config.OR_Memantine['prev_AchEI'] * bool_prev_AchEI) + 1))

    #print('logit_AchEI, logit_Memantine : ', logit_AchEI, logit_Memantine)

    # 1 / (1 + np.exp( - logit_AchEI * ((1 - epsilon) / epsilon)))
    p_AchEI = sigmoid(logit_AchEI)
    # 1 / (1 + np.exp( - logit_Memantine * ((1 - epsilon) / epsilon)))
    p_Memantine = sigmoid(logit_Memantine)

    #print('p_AchEI, p_Memantine : ', p_AchEI, p_Memantine)

    c_AchEI = sample_p(p_AchEI)
    c_Memantine = sample_p(p_Memantine)
    #print('c_AchEI, c_Memantine : ', c_AchEI, c_Memantine)

    return c_AchEI, c_Memantine

# calculate the softmax of a vector


def softmax(vector):
    e = exp(vector)
    return e / e.sum()


def sample_p(p=0.5):
    return int(np.random.binomial(size=1, n=1, p=p))


def sigmoid(x):
    sig = np.where(x < 0, np.exp(x) / (1 + np.exp(x)), 1 / (1 + np.exp(-x)))

    return sig

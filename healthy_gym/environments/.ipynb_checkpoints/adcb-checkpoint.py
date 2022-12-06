import time
import pickle as pkl
import os
import sys
import warnings

from scipy.stats import skewnorm, norm

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
with warnings.catch_warnings():
    warnings.simplefilter('ignore')

from .environment import *
from ..data.data import *

from .ADCB import imputation as im
from .ADCB import autoregression as ar
from .ADCB import treatments as tr
from .ADCB import data_models as dm
from .ADCB import config

np.random.seed(config.rnd_seed)

warnings.filterwarnings("ignore")


class ADCBEnvironment(Environment):
    """
    Abstract class specifying which methods environments have and what they return
    """

    def __init__(self, gamma, epsilon, policy='DX_Based', regenerate=False, horizon=5, n_buffer=1, rnd_seed=0, ** kwargs):
        """
        Initializes the environment

        args
            gamma (float): Treatment Effect Heterogeneity ≥ 0
            epsilon (float): Overlap parameter ∈ [ 0, 1]
            policy (string): Behavior policy ∈ {DX_Based, Santiago_Based}
            regenerate (bool): Fit data again; a bit slow for high n_buffer
            horizon: Sample trajectory length (history length) ∈ {0, 1, …, 11}
            n_buffer: The number of samples to store in buffer when resetting
        """
        self.rnd_seed = rnd_seed
        np.random.seed(self.rnd_seed)

        self.horizon = horizon
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy = policy

        config.num_steps = self.horizon

        self.P1 = 0
        self.fitted = False
        self.action_sampler = None

        self.n_buffer = n_buffer
        self.buffer_ = {}
        self.buffer_Z = {}
        self.buffer_complete = {}
        self.expected_rewards = [0]
        self.num_iterations = 0
        # load residuals for sampling noises

        with open(config.residuals_file, 'rb') as f:
            self.residuals = pkl.load(f)

        if horizon < 2:
            print('Can\'t have shorter horizon than 2')

        self.fit_model_(regenerate)

    def fit_model_(self, regenerate=False):
        """
        Loads data for the Income environment ARM and fits the model

        Args:
            regenerate (bool): Fit data afresh
        """
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        t0 = time.time()
        gamma_0 = gamma_1 = self.gamma

        if(regenerate):
            # Imputation
            imputer = im.Imputation()
            ADNIDGPData, P1, PTETHCAT_probs, PTRACCAT_probs, PTGENDER_probs, autoreg_key_lists = imputer.fit_impute()

            #print("Z~Bern(p), p: ", P1)
            # print(ADNIDGPData.head())

            # Autoregression Generation
            ADNIDGPData_bl = ADNIDGPData.loc[ADNIDGPData['VISCODE'] == 0]

            # print("Baseline")
            gen_data_bl = ar.generate_N_Samples_bl(
                self.rnd_seed,
                self.n_buffer,
                P1,
                PTETHCAT_probs,
                PTRACCAT_probs,
                PTGENDER_probs,
                ADNIDGPData_bl,
                gamma_0,
                gamma_1,
                epsilon=self.epsilon
            )

            # print("Autoregression")
            gen_df = ar.gen_autoregression_samples(
                self.n_buffer,
                self.rnd_seed,
                ADNIDGPData_bl,
                gen_data_bl,
                ADNIDGPData,
                autoreg_key_lists,
                config.autoreg_steps,
                self.horizon,
                gamma_0,
                gamma_1,
                P1,
                epsilon=self.epsilon,
                policy='DX_Based'
            )

            gen_df.to_csv(config.data_path + str(self.policy) + '_N_' + str(self.n_buffer) + '_epsilon_' + str(self.epsilon)
                          + '_gamma0_' + str(gamma_0) + '_gamma1_' + str(gamma_1) + '.csv', encoding='utf-8', index=False)
        else:
            gen_df = pd.read_csv(config.data_path + str(self.policy) + '_N_' + str(self.n_buffer) + '_epsilon_' + str(
                self.epsilon) + '_gamma0_' + str(gamma_0) + '_gamma1_' + str(gamma_1) + '.csv')

        self.model_ = gen_df
        self.data = gen_df
        self.fitted = True

        self.P1 = np.mean(self.model_.loc[(
            self.model_['VISCODE'] == 0)]['Z'])
        t1 = time.time()

        self.fit_time = t1 - t0

    def load_model_(self, model_path):
        raise Exception('Loading ARM not implemented')

    def load_cov_model(self, cov, autoreg=False):
        model = None
        model_path = config.data_path + 'models/' + cov + '_model.pkl'
        if autoreg:
            model_path = config.data_path + 'models/' + cov + '_autoreg_model.pkl'

        with open(model_path, 'rb') as f:
            model = pkl.load(f)

        return model

    def sample_noise(self, col):

        # a is the skewness parameter
        a, loc, scale = norm.fit(ADNI_bl[col].values)
        rv = skewnorm(a)
        mean = (rv.mean() * scale) + loc
        noise = ((rv.rvs(size=n) * scale) + loc) - mean

        return noise

    def reset(self):
        """
        Resets the environment and returns an observation. Generates bu

        Returns:
            observation (object)
        """

        if self.n_buffer is None:
            n = 1
        else:
            n = self.n_buffer

        #print('Resetting environment...', end='')
        t0 = time.time()
        outcomes = {}

        S = self.model_.loc[(self.model_['VISCODE'] < (self.horizon) * 12)]

        # Context at stepping time point
        cs = S[S['VISCODE'] == ((self.horizon - 1) * 12)]
        cs_prev = S[S['VISCODE'] == ((self.horizon - 2) * 12)]

        cs = cs[[c for c in S.columns if c not in [
                'ADAS13', 'CDRSB', 'MMSE', 'Delta',  'Y_0', 'Y_1', 'Y_2', 'Y_3', 'Y_4', 'Y_5', 'Y_6', 'Y_7']]]  # 'Y_hat', A, 'RID', has been removed

        D = cs.copy()
        D['prev_ADAS13'] = cs_prev['ADAS13'].values
        D['prev_DX'] = cs_prev['DX'].values
        D['prev_Y_hat'] = cs_prev['Y_hat'].values

        outcome_columns = ['Y_0', 'Y_1', 'Y_2',
                           'Y_3', 'Y_4', 'Y_5', 'Y_6', 'Y_7']

        outcomes = self.model_.loc[(
            self.model_['VISCODE'] == (self.horizon - 1) * 12)][outcome_columns]

        D[outcome_columns] = outcomes[['Y_0', 'Y_1', 'Y_2',
                                       'Y_3', 'Y_4', 'Y_5', 'Y_6', 'Y_7']].values
        # Buffer containing one subtype
        z_vals = list(pd.unique(D['Z']))
        for z in z_vals:
            self.buffer_Z[z] = D[D['Z'] == z][[
                c for c in D.columns if c not in ['Z', 'VISCODE']]]

        rid = np.random.randint(self.n_buffer)
        self.buffer_complete = D[[
            c for c in D.columns if c not in ['VISCODE']]]  # Z included in buffer

        self.buffer_ = D[D['RID'] == rid][[
            c for c in D.columns if c not in ['VISCODE']]]  # Z included in buffer

        self.expected_rewards = [0]
        self.num_iterations = 0

        return cs

    def step(self, action):
        """
        Plays an action, returns a reward and updates or terminates the environment

        Args:
            action: Played action

        Returns:
            observation (object): The observation following the action (None if terminal state reached).
            reward (float): The reward of the submitted action
            done (boolean): True if the environment has reached a terminal state
            info (dict): Returns e.g., the reward distribution of all actions so that regret can be computed
                outcomes (list): Reward distribution
                expected rewards (float): The expected reward so far
                regret (float): The regret of the submitted action

        """
        # if self.buffer_[z].shape[0] < 1:
        #     self.reset()
        #
        # d = self.buffer_[z].iloc[0: 1]
        # self.buffer_[z] = self.buffer_[z].iloc[1:]
        # outcome_columns = ['Y_'
        #                    + str(a) for a in range(8)]
        #
        # outcomes = d[outcome_columns]

        d = self.buffer_
        outcome_columns = ['Y_'
                           + str(a) for a in range(8)]

        outcomes = d[outcome_columns]
        residuals = None

        mu, sigma = norm.fit(self.residuals['ADAS13'])

        outcomes = [-(outcomes['Y_' + str(a)].values[0]
                      - outcomes['Y_' + str(0)].values[0]) + np.random.normal(mu, sigma, 1)[0] for a in range(8)]

        r = outcomes[action]

        r_avg = self.expected_rewards[-1] + \
            (r - self.expected_rewards[-1]) / (self.num_iterations + 1)

        regret = np.max(outcomes) - r
        self.expected_rewards.append(r_avg)
        self.num_iterations += 1

        return None, r, False, {'outcomes': outcomes, 'expected_rewards': self.expected_rewards, 'regret': regret}

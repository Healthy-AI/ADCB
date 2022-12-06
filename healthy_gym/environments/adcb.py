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

    def __init__(self, gamma, epsilon, policy='DX_Based', regenerate=False, horizon=5, n_buffer=1, reward_sigma=0.1, rnd_seed=0, sequential=False, contextualReward=False, z_dim=2, ** kwargs):
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
        self.sequential = sequential

        self.reward_sigma = reward_sigma
        self.rnd_seed = rnd_seed
        np.random.seed(self.rnd_seed)
        self.z_dim = z_dim
        self.horizon = horizon
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy = policy
        self.contextualReward = contextualReward

        config.num_steps = self.horizon
        config.latent_dim = self.z_dim

        self.P1 = 0
        self.fitted = False
        self.action_sampler = None

        self.n_buffer = n_buffer
        self.buffer_ = {}
        self.buffer_Z = {}
        self.buffer_complete = {}
        self.expected_rewards = [0]
        self.expected_regrets = [0]
        self.num_iterations = 0
        # load residuals for sampling noises

        with open(config.residuals_file, 'rb') as f:
            self.residuals = pkl.load(f)
        self.res_mu, self.res_sigma = norm.fit(self.residuals['ADAS13'])

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

            # print("Z~Bern(p), p: ", P1)
            # print(ADNIDGPData.head())

            # Autoregression Generation
            ADNIDGPData_bl = ADNIDGPData.loc[ADNIDGPData['VISCODE'] == 0]

            if(config.latent_dim == 2):
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
            else:
                # print("Baseline")
                gen_data_bl = ar.generate_N_Samples_bl_n_dim(
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
                gen_df = ar.gen_autoregression_samples_n_dim(
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

        # print('Resetting environment...', end='')
        t0 = time.time()
        outcomes = {}
        outcome_columns = ['Y_0', 'Y_1', 'Y_2',
                           'Y_3', 'Y_4', 'Y_5', 'Y_6', 'Y_7']

        X_cols = ['PTETHCAT', 'PTRACCAT', 'PTEDUCAT',
                  'PTGENDER', 'PTMARRY', 'TAU', 'PTAU', 'APOE4', 'FDG', 'AV45', 'ABETARatio']
        if(config.latent_dim > 2):
            X_cols = ['PTETHCAT', 'PTRACCAT', 'PTEDUCAT',
                      'PTGENDER', 'PTMARRY', 'TAU', 'PTAU',  'FDG', 'AV45']

        # Context at stepping time point
        if (self.sequential):
            self.tt = 0  # time step returned for context

            S = self.model_.loc[(self.model_['VISCODE']
                                 <= (self.horizon) * 12)]

            _cs = S.groupby('RID')

            # randomly select an rid for buffer, perhaps there's a better solution here
            rid = np.random.randint(self.n_buffer)
            cs = _cs.get_group(rid)

            outcomes = cs[outcome_columns]

            cs = cs[[c for c in S.columns if c not in [
                    'CDRSB', 'MMSE', 'Delta', ]]]  # 'Y_0', 'Y_1', 'Y_2', 'Y_3', 'Y_4', 'Y_5', 'Y_6', 'Y_7']]]  # 'Y_hat', A, 'RID', has been removed

            D = cs.copy()

            # Filled NAs with -1 perhaps there's a better solution here
            D['prev_ADAS13'] = cs['ADAS13'].shift(1).fillna(-1)
            D['prev_DX'] = cs['DX'].shift(1).fillna(-1)
            D['prev_AV45'] = cs['AV45'].shift(1).fillna(-1)
            D['prev_FDG'] = cs['FDG'].shift(1).fillna(-1)
            D['prev_TAU'] = cs['TAU'].shift(1).fillna(-1)
            D['prev_PTAU'] = cs['PTAU'].shift(1).fillna(-1)
            D['prev_Y_hat'] = cs['Y_hat'].shift(1).fillna(-1)

            self.buffer_ = D

            x = self.buffer_[X_cols]
            self.state = x

            return x

        else:
            S = self.model_.loc[(self.model_['VISCODE'] < (self.horizon) * 12)]

            cs = S[S['VISCODE'] == ((self.horizon - 1) * 12)]
            cs_prev = S[S['VISCODE'] == ((self.horizon - 2) * 12)]

            outcomes = self.model_.loc[(self.model_['VISCODE'] == (
                self.horizon - 1) * 12)][outcome_columns]

            cs = cs[[c for c in S.columns if c not in [
                    'ADAS13', 'CDRSB', 'MMSE', 'Delta',  'Y_0', 'Y_1', 'Y_2', 'Y_3', 'Y_4', 'Y_5', 'Y_6', 'Y_7']]]  # 'Y_hat', A, 'RID', has been removed

            D = cs.copy()
            D['prev_ADAS13'] = cs_prev['ADAS13'].values
            D['prev_DX'] = cs_prev['DX'].values
            D['prev_AV45'] = cs_prev['AV45'].values
            D['prev_FDG'] = cs_prev['FDG'].values
            D['prev_Y_hat'] = cs_prev['Y_hat'].values
            D['prev_TAU'] = cs_prev['TAU'].values
            D['prev_PTAU'] = cs_prev['PTAU'].values

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
            self.expected_regrets = [0]
            self.num_iterations = 0

            x = self.buffer_[X_cols + ['prev_ADAS13']]
            self.state = x

            return x

        # old
        # return cs

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
        done = False
        d = self.buffer_

        if(self.sequential):
            if self.buffer_.shape[0] < 1:
                self.reset()
                # done = True

            d = self.buffer_.iloc[0: 1]
            self.buffer_ = self.buffer_.iloc[1:]
            outcome_columns = ['Y_0', 'Y_1', 'Y_2',
                               'Y_3', 'Y_4', 'Y_5', 'Y_6', 'Y_7']

            outcomes = d[outcome_columns]
            y_noise = np.random.normal(0, self.reward_sigma, size=1)
            outcomes = -np.array([outcomes['Y_' + str(a)].values[0]
                                 + y_noise for a in range(8)], dtype='float64')
            if(not self.contextualReward):
                outcomes = np.array([(outcomes[a] - outcomes[0]) + y_noise
                                    for a in range(8)])
            # rewards = np.array([-(outcomes['Y_' + str(action)] - outcomes['Y_' + str(0)])
            #                    + np.random.normal(0,  self.reward_sigma, 1)[0] for a in range(8)])

            # rewards with contexts
            rewards = outcomes

            r = rewards[action]
            regret = np.max(rewards) - rewards[action]

            cx = d[[c for c in d.columns if c not in [
                'Y_0', 'Y_1', 'Y_2', 'Y_3', 'Y_4', 'Y_5', 'Y_6', 'Y_7']]]

            #ttt = self.tt

            #self.tt += 1
            self.num_iterations += 1

            return self.state, r, False, {'context': cx, 'outcomes': outcomes, 'expected_rewards': self.expected_rewards, 'expected_regrets': self.expected_regrets,
                                          'regret': regret,
                                          'reward': r}

        else:
            #outcomes = d[outcome_columns]

            x_ADAS13 = d[['prev_ADAS13', 'Z', 'PTETHCAT', 'PTRACCAT', 'PTEDUCAT',
                          'PTGENDER', 'PTMARRY', 'TAU', 'PTAU', 'FDG', 'AV45']]

            ADAS13 = self.load_cov_model('ADAS13', autoreg=True).predict(
                dm.check_categorical(x_ADAS13, list(x_ADAS13.columns), '_'))  # + np.random.normal(0, config.Autoreg_ADAS_NOISE, size=1)
            y_noise = np.random.normal(0, self.reward_sigma, size=1)
            outcomes = -np.array([(tr.assign_treatment_effect_n_dim(self.gamma, a, x_ADAS13['Z'].values)
                                   + ADAS13 + y_noise) for a in range(8)])
            # print(outcomes)
            #y_noise = np.random.normal(0, self.reward_sigma, size=1)
            # outcomes = -np.array([outcomes['Y_' + str(a)].values[0] +
            #                     y_noise for a in range(8)], dtype='float64')

            if(not self.contextualReward):
                outcomes = np.array([(outcomes[a] - outcomes[0]) + y_noise
                                     for a in range(8)])
            # rewards = np.array([-(outcomes[a] - outcomes[0])
            #                    + np.random.normal(0,  self.reward_sigma, 1)[0] for a in range(8)])
            # rewards with contexts
            rewards = outcomes

            r = rewards[action]
            regret = np.max(rewards) - rewards[action]

            # r_avg = self.expected_rewards[-1] + \
            #     (r - self.expected_rewards[-1]) / (self.num_iterations + 1)
            #
            # regret_avg = self.expected_regrets[-1] + \
            #     (regret - self.expected_regrets[-1]) / (self.num_iterations + 1)
            #
            # self.expected_rewards.append(r_avg)
            # self.expected_regrets.append(regret_avg)

            cx = d[[c for c in d.columns if c not in [
                'Y_0', 'Y_1', 'Y_2', 'Y_3', 'Y_4', 'Y_5', 'Y_6', 'Y_7']]]

            self.num_iterations += 1

            return self.state, r, False, {'context': cx, 'outcomes': outcomes, 'expected_rewards': self.expected_rewards, 'expected_regrets': self.expected_regrets,
                                          'regret': regret,
                                          'reward': outcomes[action]}

    def correct_model(self, model):
        return model == Z

    def get_models(self, e, n_latent_states=6, contextualR=True):
        '''
        return list of models (one for each latent state).
        '''

        return [LatentState(e, i, contextualReward=contextualR) for i in range(n_latent_states)]


class LatentState:
    def __init__(self, e, z, contextualReward=True):
        self.e = e
        self.contextualReward = contextualReward
        self.z = z

    def predict(self, x):
        return self.gen_potential_Outcomes(x)

    def gen_potential_Outcomes(self, x):
        Y_s = {}
        AV45, FDG, TAU, PTAU = None, None, None, None

        # 'FDG':  ['PTETHCAT', 'PTRACCAT', 'Z', 'TAU', 'PTAU']
        # 'AV45':  ['PTETHCAT', 'PTRACCAT', 'Z', 'TAU', 'PTAU']
        # 'TAU':  ['PTETHCAT', 'PTRACCAT', 'PTGENDER', 'Z', 'AGE'],
        # 'PTAU':  ['PTETHCAT', 'PTRACCAT', 'PTGENDER', 'Z', 'AGE'],

        X_cols_ADAS13 = ['prev_ADAS13', 'PTETHCAT', 'PTRACCAT', 'PTEDUCAT',
                         'PTGENDER', 'PTMARRY', 'TAU', 'PTAU', 'FDG', 'AV45']

        X_cols_TAU = ['prev_TAU', 'PTETHCAT', 'PTRACCAT', 'PTGENDER', 'AGE']
        X_cols_PTAU = ['prev_PTAU', 'PTETHCAT', 'PTRACCAT', 'PTGENDER', 'AGE']

        X_cols_AV45 = ['prev_AV45', 'PTETHCAT', 'PTRACCAT', 'TAU', 'PTAU']
        X_cols_FDG = ['prev_FDG', 'PTETHCAT', 'PTRACCAT', 'TAU', 'PTAU']

        x_ADAS13 = x[X_cols_ADAS13]
        x_ADAS13['Z'] = self.z

        x_AV45 = x[X_cols_AV45]
        x_AV45['Z'] = self.z

        x_FDG = x[X_cols_FDG]
        x_FDG['Z'] = self.z

        x_TAU = x[X_cols_TAU]
        x_TAU['Z'] = self.z

        x_PTAU = x[X_cols_PTAU]
        x_PTAU['Z'] = self.z

        ADAS13 = self.e.load_cov_model('ADAS13', autoreg=True).predict(
            dm.check_categorical(x_ADAS13, list(x_ADAS13.columns), '_'))  # + np.random.normal(0, config.Autoreg_ADAS_NOISE, size=1)

        # if (self.contextualReward):
        TAU = self.e.load_cov_model('TAU', autoreg=True).predict(
            dm.check_categorical(x_TAU, list(x_TAU.columns), '_'))  # + np.random.normal(0, config.Autoreg_TAU_NOISE, size=1)

        PTAU = self.e.load_cov_model('PTAU', autoreg=True).predict(
            dm.check_categorical(x_PTAU, list(x_PTAU.columns), '_'))  # + np.random.normal(0, config.Autoreg_PTAU_NOISE, size=1)

        AV45 = self.e.load_cov_model('AV45', autoreg=True).predict(
            dm.check_categorical(x_AV45, list(x_AV45.columns), '_'))  # + np.random.normal(0, config.Autoreg_AV45_NOISE, size=1)
        #print("AV45: ", AV45)
        FDG = self.e.load_cov_model('FDG', autoreg=True).predict(
            dm.check_categorical(x_FDG, list(x_FDG.columns), '_'))  # + np.random.normal(0, config.Autoreg_FDG_NOISE, size=1)
        #print("FDG: ", FDG)

        ADAS13 = self.e.load_cov_model('ADAS13', autoreg=True).predict(
            dm.check_categorical(x_ADAS13, list(x_ADAS13.columns), '_'))  # + np.random.normal(0, config.Autoreg_ADAS_NOISE, size=1)

        y_noise = 0  # np.random.normal(0, self.sigma, size=1)
        Y_s = -np.array([(tr.assign_treatment_effect_n_dim(self.e.gamma, a, self.z)
                          + ADAS13 + y_noise) for a in range(8)])

        if (not self.contextualReward):
            Y_s = np.array([(Y_s[a] - Y_s[0]) for a in range(8)])

        return Y_s, (x['AV45'].values, AV45), (x['FDG'].values, FDG), (x['TAU'].values, TAU), (x['PTAU'].values, PTAU)

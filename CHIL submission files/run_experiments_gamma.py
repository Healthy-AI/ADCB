# System imports
import seq_model as sl
import config
import effects_experiments as ee
import warnings
from matplotlib import ticker
import matplotlib.pyplot as plt
import pickle
import random
import os
import sys
import numpy as np
import argparse
import pandas as pd
np.random.seed(0)


# plotting

plt.rc('font', size=16, family='serif')
plt.grid(zorder=-100)
plt.switch_backend('agg')
plt.rcParams.update({'figure.max_open_warning': 0})

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

np.random.seed(config.rnd_seed)


def main(gamma=2):
    params = config.SeqParameters()
    history = config.history
    N = config.N
    epsilon = config.epsilon
    month = config.month
    bool_train = config.bool_train
    gamma_0 = gamma_1 = gamma

    # Experiments
    S_ATEs_dict, S_ATEs_random_dict = {k: []
                                       for k in range(8)}, {k: [] for k in range(8)}
    T_ATEs_dict, T_ATEs_random_dict = {k: []
                                       for k in range(8)}, {k: [] for k in range(8)}
    IPW_ATEs_dict, IPW_ATEs_random_dict = {k: []
                                           for k in range(8)}, {k: [] for k in range(8)}

    T_pehes_dict, T_pehes_random_dict = {k: []
                                         for k in range(8)}, {k: [] for k in range(8)}
    S_pehes_dict, S_pehes_random_dict = {k: []
                                         for k in range(8)}, {k: [] for k in range(8)}

    pehes_seq_cov_dict, ATEs_seq_cov_dict = {
        k: [] for k in range(8)}, {k: [] for k in range(8)}
    pehes_seq_random_dict, ATEs_seq_random_dict = {
        k: [] for k in range(8)}, {k: [] for k in range(8)}

    # print(gen_df_Santiago_Based.tail())

    if(bool_train):
        gen_df_DX_Based = pd.read_csv(config.gen_file_prefix_policy1 + 'N_' + str(N) + '_rnd_seed_' + str(
            0) + '_epsilon_' + str(epsilon) + '_gamma0_' + str(gamma_0) + '_gamma1_' + str(gamma_1) + '.csv')
        # print(gen_df_DX_Based.tail())
        gen_df_Santiago_Based = pd.read_csv(config.gen_file_prefix_policy2 + 'N_' + str(N) + '_rnd_seed_' + str(
            0) + '_epsilon_' + str(epsilon) + '_gamma0_' + str(gamma_0) + '_gamma1_' + str(gamma_1) + '.csv')
        # print(gen_df_Santiago_Based.tail())
        # ATEs
        gen_df_DX_Based, gen_df_Santiago_Based, gen_df_DX_Based_train, gen_df_DX_Based_test, gen_df_Santiago_Based_train, gen_df_Santiago_Based_test, RIDs_train, RIDs_test = ee.pre_processing(
            gen_df_DX_Based, gen_df_Santiago_Based, train_size=0.8)

        _ = sl.train(month, history, gen_df_DX_Based_train, gen_df_DX_Based_test,
                     RIDs_train, RIDs_test, "DX", N, epsilon, gamma_0, gamma_1, train=bool_train)
        _ = sl.train(month, history, gen_df_Santiago_Based_train, gen_df_Santiago_Based_test,
                     RIDs_train, RIDs_test, "Santiago", N, epsilon, gamma_0, gamma_1, train=bool_train)

    for num_repetition in range(config.num_repetitions):
        config.rnd_seed = num_repetition
        np.random.seed(config.rnd_seed)

        gen_df_DX_Based = pd.read_csv(config.gen_file_prefix_policy1 + 'N_' + str(N) + '_rnd_seed_' + str(
            0) + '_epsilon_' + str(epsilon) + '_gamma0_' + str(gamma_0) + '_gamma1_' + str(gamma_1) + '.csv')
        # print(gen_df_DX_Based.tail())
        gen_df_Santiago_Based = pd.read_csv(config.gen_file_prefix_policy2 + 'N_' + str(N) + '_rnd_seed_' + str(
            0) + '_epsilon_' + str(epsilon) + '_gamma0_' + str(gamma_0) + '_gamma1_' + str(gamma_1) + '.csv')
        # print(gen_df_Santiago_Based.tail())

        gen_df_DX_Based, gen_df_Santiago_Based, gen_df_DX_Based_train, gen_df_DX_Based_test, gen_df_Santiago_Based_train, gen_df_Santiago_Based_test, RIDs_train, RIDs_test = ee.pre_processing(
            gen_df_DX_Based, gen_df_Santiago_Based, train_size=0.8)
        print("len(RIDs_train): ", len(RIDs_train),
              "len(RIDs_test): ", len(RIDs_test))

        gtruth_effect_means = {0: 0}
        for j in range(1, 8):
            gtruth_effect_means[j] = ee.a_GtruthEffects(
                j, gen_df_Santiago_Based, month)

        # ATEs
        print("\nS-Learner ATE")
        S_ATEs, S_ATEs_random = ee.S_Learner_ATE(
            month, gen_df_DX_Based_train, gen_df_DX_Based_test, gen_df_Santiago_Based_train, gen_df_Santiago_Based_test)
        for key, value in S_ATEs.items():
            #print(key, value)
            S_ATEs_dict[key].append(value)
        for key, value in S_ATEs_random.items():
            S_ATEs_random_dict[key].append(value)
        print("Gtruth ATEs: ", gtruth_effect_means)
        print("ATEs: ", S_ATEs)
        print("ATEs_random: ", S_ATEs_random)

        print("\nT-Learner ATE")
        T_ATEs, T_ATEs_random = ee.T_Learner_ATE(
            month, gen_df_DX_Based_train, gen_df_DX_Based_test, gen_df_Santiago_Based_train, gen_df_Santiago_Based_test)
        for key, value in T_ATEs.items():
            T_ATEs_dict[key].append(value)
        for key, value in T_ATEs_random.items():
            T_ATEs_random_dict[key].append(value)
        print("Gtruth ATEs: ", gtruth_effect_means)
        print("ATEs: ", T_ATEs)
        print("ATEs_random: ", T_ATEs_random)

        """print("\nIPW ATE")
        IPW_ATEs, IPW_ATEs_random = ee.IPW_ATE(month, gen_df_DX_Based_train, gen_df_DX_Based_test, gen_df_Santiago_Based_train, gen_df_Santiago_Based_test)
        for key, value in IPW_ATEs.items():
            IPW_ATEs_dict[key].append(value)
        for key, value in IPW_ATEs_random.items():
            IPW_ATEs_random_dict[key].append(value)
        print("Gtruth ATEs: ", gtruth_effect_means)
        print("ATEs: ", IPW_ATEs)
        print("ATEs_random: ", IPW_ATEs_random)"""

        # CATES
        print("\nT-Learner CATE")
        T_pehes, T_pehes_random = ee.T_Learner_CATE(
            month, gen_df_DX_Based_train, gen_df_DX_Based_test, gen_df_Santiago_Based_train, gen_df_Santiago_Based_test)
        for key, value in T_pehes.items():
            T_pehes_dict[key].append(value)
        for key, value in T_pehes_random.items():
            T_pehes_random_dict[key].append(value)
        print("PEHEs: ", T_pehes)
        print("PEHEs_random: ", T_pehes_random)

        print("\nS-Learner CATE")
        S_pehes, S_pehes_random = ee.S_Learner_CATE(
            month, gen_df_DX_Based_train, gen_df_DX_Based_test, gen_df_Santiago_Based_train, gen_df_Santiago_Based_test)
        for key, value in S_pehes.items():
            S_pehes_dict[key].append(value)
        for key, value in S_pehes_random.items():
            S_pehes_random_dict[key].append(value)
        print("PEHEs: ", S_pehes)
        print("PEHEs_random: ", S_pehes_random)

        # Means, std --- https://www.py4u.net/discuss/225810

        pehes_seq_cov, ATEs_seq_cov = sl.seq_ATE_PEHE(
            RIDs_test, history, gen_df_DX_Based_test, month, "DX", N, epsilon, gamma_0, gamma_1)
        pehes_seq_random, ATEs_seq_random = sl.seq_ATE_PEHE(
            RIDs_test, history, gen_df_Santiago_Based_test, month, "Santiago", N, epsilon, gamma_0, gamma_1)

        for key, value in pehes_seq_cov.items():
            pehes_seq_cov_dict[key].append(value)
        for key, value in ATEs_seq_cov.items():
            ATEs_seq_cov_dict[key].append(value)

        for key, value in pehes_seq_random.items():
            pehes_seq_random_dict[key].append(value)
        for key, value in ATEs_seq_random.items():
            ATEs_seq_random_dict[key].append(value)

    with open(config.plotting_pickle_prefix + 'gamma_' + str(gamma) + '.pickle', 'wb') as f:
        pickle.dump([S_ATEs_dict, S_ATEs_random_dict, T_ATEs_dict, T_ATEs_random_dict, IPW_ATEs_dict, IPW_ATEs_random_dict, T_pehes_dict,
                    T_pehes_random_dict, S_pehes_dict, S_pehes_random_dict, pehes_seq_cov_dict, ATEs_seq_cov_dict, pehes_seq_random_dict, ATEs_seq_random_dict], f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run_experiments gamma')
    parser.add_argument('-g', '--gamma', type=float, default=2, dest='gamma')

    args = parser.parse_args()
    main(gamma=float(args.gamma))

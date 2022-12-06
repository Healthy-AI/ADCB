import seq_model as sl
import config
import effects_experiments as ee
import warnings
import pandas as pd
import numpy as np
import random

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


params = config.SeqParameters()

N = config.N
epsilon = config.epsilon
gamma = 2

gamma_0 = gamma_1 = gamma

gen_autoreg_df = pd.read_csv('data/gen_autoreg_df_N_' + str(N) + '_epsilon_' + str(
    epsilon) + '_gamma0_' + str(gamma_0) + '_gamma1_' + str(gamma_1) + '.csv')
# print(gen_autoreg_df.tail())
gen_autoreg_df_random = pd.read_csv('data/gen_autoreg_df_random_N_' + str(
    N) + '_gamma0_' + str(gamma_0) + '_gamma1_' + str(gamma_1) + '.csv')
# print(gen_autoreg_df_random.tail())

gen_autoreg_df, gen_autoreg_df_random, gen_autoreg_df_train, gen_autoreg_df_test, gen_autoreg_df_random_train, gen_autoreg_df_random_test, RIDs_train, RIDs_test = ee.pre_processing(
    gen_autoreg_df, gen_autoreg_df_random, train_size=0.8)
print("len(RIDs_train): ", len(RIDs_train), "len(RIDs_test): ", len(RIDs_test))

# Experiments
month = 12
gtruth_effect_means = {0: 0}
for i in range(1, 8):
    gtruth_effect_means[i] = ee.a_GtruthEffects(
        i, gen_autoreg_df_random, month)

# ATEs
print("\nS-Learner ATE")
S_ATEs, S_ATEs_random = ee.S_Learner_ATE(
    month, gen_autoreg_df_train, gen_autoreg_df_test, gen_autoreg_df_random_train, gen_autoreg_df_random_test)
print("Gtruth ATEs: ", gtruth_effect_means)
print("ATEs: ", S_ATEs)
print("ATEs_random: ", S_ATEs)

print("\nT-Learner ATE")
T_ATEs, T_ATEs_random = ee.T_Learner_ATE(
    month, gen_autoreg_df_train, gen_autoreg_df_test, gen_autoreg_df_random_train, gen_autoreg_df_random_test)
print("Gtruth ATEs: ", gtruth_effect_means)
print("ATEs: ", T_ATEs)
print("ATEs_random: ", T_ATEs_random)

print("\nIPW ATE")
IPW_ATEs, IPW_ATEs_random = ee.IPW_ATE(
    month, gen_autoreg_df_train, gen_autoreg_df_test, gen_autoreg_df_random_train, gen_autoreg_df_random_test)
print("Gtruth ATEs: ", gtruth_effect_means)
print("ATEs: ", IPW_ATEs)
print("ATEs_random: ", IPW_ATEs_random)

# CATES
print("\nT-Learner CATE")
T_pehes, T_pehes_random = ee.T_Learner_CATE(
    month, gen_autoreg_df_train, gen_autoreg_df_test, gen_autoreg_df_random_train, gen_autoreg_df_random_test)
print("PEHEs: ", T_pehes)
print("PEHEs_random: ", T_pehes_random)

#print("\nS-Learner CATE")
S_pehes, S_pehes_random = ee.S_Learner_CATE(
    month, gen_autoreg_df_train, gen_autoreg_df_test, gen_autoreg_df_random_train, gen_autoreg_df_random_test)
print("PEHEs: ", S_pehes)
print("PEHEs_random: ", S_pehes_random)

# Means, std --- https://www.py4u.net/discuss/225810
bool_train = config.bool_train

_ = sl.train(month, gen_autoreg_df_train, gen_autoreg_df_test, RIDs_train,
             RIDs_test, "cov", N, epsilon, gamma_0, gamma_1, train=bool_train)
_ = sl.train(month, gen_autoreg_df_random_train, gen_autoreg_df_random_test,
             RIDs_train, RIDs_test, "random", N, epsilon, gamma_0, gamma_1, train=bool_train)

pehes_seq_cov, ATEs_seq_cov = sl.seq_ATE_PEHE(
    RIDs_test, gen_autoreg_df_test, month)
pehes_seq_random, ATEs_seq_random = sl.seq_ATE_PEHE(
    RIDs_test, gen_autoreg_df_random_test, month)

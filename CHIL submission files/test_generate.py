import warnings
import pickle as pkl
import config
import autoregression as ar
import imputation as im
import os
import sys
import numpy as np
import pandas as pd
np.random.seed(config.rnd_seed)


with warnings.catch_warnings():
    warnings.simplefilter('ignore')


reimpute = True


# Imputation
imputer = im.Imputation()

imputation_file = 'data/imputed_ADNI.pkl'
imputation_results_reg = 'data/imputation_results_regression.csv'
imputation_results_cla = 'data/imputation_results_classification.csv'

if reimpute or not os.path.isfile(imputation_file):
    print('Imputing ADNI...')
    ADNIDGPData, P1, PTETHCAT_probs, PTRACCAT_probs, PTGENDER_probs, autoreg_key_lists, imputation_results = imputer.fit_impute(
        return_metrics=True)

    pkl.dump((ADNIDGPData, P1, PTETHCAT_probs, PTRACCAT_probs,
             PTGENDER_probs, autoreg_key_lists), open(imputation_file, 'wb'))
    imputation_results[0].to_csv(imputation_results_cla)
    imputation_results[1].to_csv(imputation_results_reg)
else:
    print('Load imputed ADNI...')
    ADNIDGPData, P1, PTETHCAT_probs, PTRACCAT_probs, PTGENDER_probs, autoreg_key_lists = pkl.load(
        open(imputation_file, 'rb'))

print("Z~Bern(p), p: ", P1)
print(ADNIDGPData.head())

# Autoregression Generation
ADNIDGPData_bl = ADNIDGPData.loc[ADNIDGPData['VISCODE'] == 0]

N = config.N  # Number of patient trajectories
epsilon = config.epsilon
num_steps = config.num_steps  # T

epsilons = [0.1]
gammas = [4]
Ns = [100]

for N in Ns:
    for epsilon in epsilons:

        for gamma in gammas:
            print("gamma: ", gamma)
            print("N: ", N)
            print("epsilon: ", epsilon)

            gamma_0 = gamma_1 = gamma

            print("Generating baseline")

            # Generate baseline samples
            gen_data_bl, results_gen_bl = ar.generate_N_Samples_bl(
                N, P1, PTETHCAT_probs, PTRACCAT_probs, PTGENDER_probs, ADNIDGPData_bl, gamma_0, gamma_1, epsilon=epsilon, return_metrics=True)

            # Store results for baseline
            results_gen_bl[0].to_csv('data/results_bl_cla_autoreg_df_random_N_' + str(
                N) + '_gamma0_' + str(gamma_0) + '_gamma1_' + str(gamma_1) + '.csv')
            results_gen_bl[1].to_csv('data/results_bl_reg_autoreg_df_random_N_' + str(
                N) + '_gamma0_' + str(gamma_0) + '_gamma1_' + str(gamma_1) + '.csv')

            print("Autoregression")

            gen_autoreg_df, results_gen_ar = ar.gen_autoregression_samples(
                1, gen_data_bl, ADNIDGPData, autoreg_key_lists, config.autoreg_steps, num_steps,  gamma_0, gamma_1, P1, epsilon=epsilon, policy='covariate', return_metrics=True)

            # Store results for autoregression
            results_gen_ar[0].to_csv('data/results_ar_cla_autoreg_df_random_N_' + str(
                N) + '_gamma0_' + str(gamma_0) + '_gamma1_' + str(gamma_1) + '.csv')
            results_gen_ar[1].to_csv('data/results_ar_reg_autoreg_df_random_N_' + str(
                N) + '_gamma0_' + str(gamma_0) + '_gamma1_' + str(gamma_1) + '.csv')

            gen_autoreg_df_random = ar.gen_autoregression_samples(
                1, gen_data_bl, ADNIDGPData, autoreg_key_lists, config.autoreg_steps, num_steps,  gamma_0, gamma_1, P1, epsilon=epsilon, policy='random')

            gen_autoreg_df_random.to_csv('data/gen_autoreg_df_random_N_' + str(N) + '_gamma0_' + str(
                gamma_0) + '_gamma1_' + str(gamma_1) + '.csv', encoding='utf-8', index=False)
            gen_autoreg_df.to_csv('data/gen_autoreg_df_N_' + str(N) + '_epsilon_' + str(epsilon) + '_gamma0_' + str(
                gamma_0) + '_gamma1_' + str(gamma_1) + '.csv', encoding='utf-8', index=False)

            # print(gen_autoreg_df_random.tail())
            # print(gen_autoreg_df.tail())

        #ar.plot_statistics(ADNIDGPData, gen_autoreg_df)

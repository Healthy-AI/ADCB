# System imports
import pickle
import warnings
import config
import autoregression as ar
import imputation as im
import os
import sys
import numpy as np
import argparse
import pandas as pd
np.random.seed(config.rnd_seed)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')


def main(N=10000, gamma=2, epsilon=0.1):
    # Imputation
    imputer = im.Imputation()
    ADNIDGPData, P1, PTETHCAT_probs, PTRACCAT_probs, PTGENDER_probs, autoreg_key_lists = imputer.fit_impute()

    print("Z~Bern(p), p: ", P1)
    print(ADNIDGPData.head())

    # Autoregression Generation
    ADNIDGPData_bl = ADNIDGPData.loc[ADNIDGPData['VISCODE'] == 0]

    # N = config.N #Number of patient trajectories
    #epsilon = config.epsilon

    num_steps = config.num_steps  # T

    #epsilons = [0.1, 0.5, 0.9]
    #gammas = [1, 2, 4]
    #Ns = [1000, 10000, 50000]

    for num_repetition in range(1):
        print("gamma: ", gamma)
        print("N: ", N)
        print("epsilon: ", epsilon)

        gamma_0 = gamma_1 = gamma

        print("Generating baseline")
        print("gamma: ", gamma)
        print("N: ", N)
        print("epsilon: ", epsilon)

        gen_data_bl, gen_df_DX_Based, gen_df_Santiago_Based = None, None, None

        if(config.return_metrics):
            gen_data_bl, bl_results = ar.generate_N_Samples_bl(
                num_repetition, N, P1, PTETHCAT_probs, PTRACCAT_probs, PTGENDER_probs, ADNIDGPData_bl, gamma_0, gamma_1, epsilon=epsilon)

            print("Autoregression")

            gen_df_DX_Based, autoregression_results_dx = ar.gen_autoregression_samples(N, num_repetition, ADNIDGPData_bl, gen_data_bl, ADNIDGPData,
                                                                                       autoreg_key_lists, config.autoreg_steps, num_steps,  gamma_0, gamma_1, P1, epsilon=epsilon, policy='DX_Based')
            gen_df_Santiago_Based, autoregression_results_santiago = ar.gen_autoregression_samples(N, num_repetition, ADNIDGPData_bl, gen_data_bl, ADNIDGPData,
                                                                                                   autoreg_key_lists, config.autoreg_steps, num_steps,  gamma_0, gamma_1, P1, epsilon=epsilon, policy='Santiago_Based')
            with open(config.metrics_results_file, 'wb') as f:
                pickle.dump([bl_results, autoregression_results_dx,
                            autoregression_results_santiago], f)
        else:
            gen_data_bl = ar.generate_N_Samples_bl(
                num_repetition, N, P1, PTETHCAT_probs, PTRACCAT_probs, PTGENDER_probs, ADNIDGPData_bl, gamma_0, gamma_1, epsilon=epsilon)

            print("Autoregression")

            gen_df_DX_Based = ar.gen_autoregression_samples(N, num_repetition, ADNIDGPData_bl, gen_data_bl, ADNIDGPData,
                                                            autoreg_key_lists, config.autoreg_steps, num_steps,  gamma_0, gamma_1, P1, epsilon=epsilon, policy='DX_Based')
            gen_df_Santiago_Based = ar.gen_autoregression_samples(N, num_repetition, ADNIDGPData_bl, gen_data_bl, ADNIDGPData,
                                                                  autoreg_key_lists, config.autoreg_steps, num_steps,  gamma_0, gamma_1, P1, epsilon=epsilon, policy='Santiago_Based')

        gen_df_DX_Based.to_csv(config.gen_file_prefix_policy1 + 'N_' + str(N) + '_rnd_seed_' + str(num_repetition) + '_epsilon_' + str(
            epsilon) + '_gamma0_' + str(gamma_0) + '_gamma1_' + str(gamma_1) + '.csv', encoding='utf-8', index=False)

        gen_df_Santiago_Based.to_csv(config.gen_file_prefix_policy2 + 'N_' + str(N) + '_rnd_seed_' + str(num_repetition) + '_epsilon_' + str(
            epsilon) + '_gamma0_' + str(gamma_0) + '_gamma1_' + str(gamma_1) + '.csv', encoding='utf-8', index=False)

        #ar.plot_statistics(ADNIDGPData, gen_autoreg_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Samples.')
    parser.add_argument('-n', '--N', type=int, default=10000, dest='N')
    parser.add_argument('-g', '--gamma', type=float, default=2, dest='gamma')
    parser.add_argument('-e', '--epsilon', type=float,
                        default=0.1, dest='epsilon')

    args = parser.parse_args()

    main(N=int(args.N), gamma=float(args.gamma), epsilon=float(args.epsilon))

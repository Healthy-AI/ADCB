import warnings
import matplotlib.pyplot as plt
from . import data_models as dm
from . import latent_fitting as lf
from . import DataLoader as dl
from . import config

import numpy as np
import pandas as pd

np.random.seed(config.rnd_seed)

# plotting
plt.rc('font', size=16, family='serif')
plt.grid(zorder=-100)
plt.switch_backend('agg')
plt.rcParams.update({'figure.max_open_warning': 0})

with warnings.catch_warnings():
    warnings.simplefilter('ignore')


class Imputation:
    def __init__(self):
        pass

    def fit_impute(self, return_metrics=False):
        data_loader = dl.DataLoader()
        ADNIDGPData = data_loader.loadData(config.ADNIFilepath, config.DGPcols)
        # print("Number of columns: ", len(list(ADNIDGPData.columns)))
        # print("Columns: ", list(ADNIDGPData.columns))

        PTETHCAT_probs, PTRACCAT_probs, PTGENDER_probs = data_loader.race_gender_bl_statistics(
            ADNIDGPData)
        # print(PTETHCAT_probs, '\n', PTRACCAT_probs, '\n', PTGENDER_probs)

        lsf = lf.LatentStateFit()
        ADNI_DGP_df, P1 = lsf.fitZ_baseline(ADNIDGPData)
        # print("ADNI_DGP_df.columns", ADNI_DGP_df.head())

        # Selecting data at baseline
        ADNI_DGP_df_bl = ADNI_DGP_df.loc[ADNI_DGP_df['VISCODE'] == 0]
        ADNI_DGP_df_bl.dropna(inplace=True)

        # Checking status of data at bl
        patient_ids = np.unique(ADNI_DGP_df.RID.values)

        autoreg_key_lists = dm.get_autoreg_keys(ADNI_DGP_df, patient_ids)

        # for key, value in autoreg_key_lists.items():
        #     # print("Number of autoregression samples at "
        #           + str(key) + ": " + str(len(value)))

        # Fitting Causal Graph at $bl$
        if(config.latent_dim == 2):
            lrAPOE4, resAPOE4 = dm.fit_X1_model(
                ADNI_DGP_df_bl, 'APOE4', 'CLASSIFICATION', test_size=0.2, return_metrics=True)

        lrEDUCAT, resEDUCAT = dm.fit_X1_model(
            ADNI_DGP_df_bl, 'PTEDUCAT', 'CLASSIFICATION', test_size=0.2, return_metrics=True)
        lrMARRY, resMARRY = dm.fit_X1_model(
            ADNI_DGP_df_bl, 'PTMARRY', 'CLASSIFICATION', test_size=0.2, return_metrics=True)
        lrTAU, TAURMSE, _, resTAU = dm.fit_X1_model(
            ADNI_DGP_df_bl, 'TAU', 'REGRESSION', test_size=0.2, return_metrics=True)
        lrPTAU, PTAURMSE, _, resPTAU = dm.fit_X1_model(
            ADNI_DGP_df_bl, 'PTAU', 'REGRESSION', test_size=0.2, return_metrics=True)
        lrFDG, FDGRMSE, _, resFDG = dm.fit_X1_model(
            ADNI_DGP_df_bl, 'FDG', 'REGRESSION', test_size=0.2, return_metrics=True)
        lrAV45, AV45RMSE, _, resAV45 = dm.fit_X1_model(
            ADNI_DGP_df_bl, 'AV45', 'REGRESSION', test_size=0.2, return_metrics=True)
        lrMMSE, MMSERMSE, _, resMMSE = dm.fit_X1_model(
            ADNI_DGP_df_bl, 'MMSE', 'REGRESSION', test_size=0.2, return_metrics=True)
        lrADAS13, ADAS13RMSE, _, resADAS13 = dm.fit_X1_model(
            ADNI_DGP_df_bl, 'ADAS13', 'REGRESSION', test_size=0.2, return_metrics=True)
        lrDX, resDX = dm.fit_X1_model(
            ADNI_DGP_df_bl, 'DX', 'CLASSIFICATION', test_size=0.2, return_metrics=True)
        lrCDRSB, CDRSBRMSE, _, resCDRSB = dm.fit_X1_model(
            ADNI_DGP_df_bl, 'CDRSB', 'REGRESSION', test_size=0.2, return_metrics=True)

        # Store results of imputation model fitting
        resRegressors = pd.DataFrame(
            [resTAU, resPTAU, resFDG, resAV45, resMMSE, resADAS13, resCDRSB])
        if(config.latent_dim == 2):
            resClassifiers = pd.DataFrame(
                [resAPOE4, resEDUCAT, resMARRY, resDX])
        resClassifiers = pd.DataFrame([resEDUCAT, resMARRY, resDX])
        imputation_results = [resClassifiers, resRegressors]

        # Checking for NaNs to impute
        ADNI_DGP_NoNaNs_df = ADNI_DGP_df.copy()
        RIDs = pd.unique(ADNI_DGP_NoNaNs_df.RID)
        # #print(len(RIDs))

        months = list(range(0, config.num_steps * 12, 12))
        all_rids = set(RIDs)
        imputed_idxs_dict = {}
        observed_idxs_dict = {}
        observed_row_ids_dict = {}
        imputed_row_ids_dict = {}

        for month in months:
            present_rids = list(
                ADNI_DGP_NoNaNs_df.loc[ADNI_DGP_NoNaNs_df['VISCODE'] == month].RID)
            observed_idxs_dict[month] = [int(x) for x in present_rids]
            present_rids = set(present_rids)
            nan_RIDs = list(all_rids.difference(present_rids))
            imputed_idxs_dict[month] = [int(x) for x in nan_RIDs]

            # for rid in nan_RIDs:
            #    df = pd.DataFrame(ADNI_DGP_NoNaNs_df.loc[ADNI_DGP_NoNaNs_df['RID'] == rid].iloc[0][['RID', 'AGE', 'PTGENDER', 'PTEDUCAT', 'PTETHCAT', 'PTRACCAT', 'PTMARRY', 'APOE4','Z']]).T
            #    df['VISCODE'] = month
            #    ADNI_DGP_NoNaNs_df = pd.concat([ADNI_DGP_NoNaNs_df, df])

            # FREDRIK CHANGE BELOW
            if(config.latent_dim == 2):
                df = ADNI_DGP_NoNaNs_df.loc[ADNI_DGP_NoNaNs_df['RID'].isin(nan_RIDs)][[
                    'RID', 'AGE', 'PTGENDER', 'PTEDUCAT', 'PTETHCAT', 'PTRACCAT', 'PTMARRY', 'APOE4', 'Z', 'ABETARatio']].groupby('RID').head(1)
            else:
                df = ADNI_DGP_NoNaNs_df.loc[ADNI_DGP_NoNaNs_df['RID'].isin(nan_RIDs)][[
                    'RID', 'AGE', 'PTGENDER', 'PTEDUCAT', 'PTETHCAT', 'PTRACCAT', 'PTMARRY', 'Z']].groupby('RID').head(1)

            df['VISCODE'] = month
            ADNI_DGP_NoNaNs_df = pd.concat([ADNI_DGP_NoNaNs_df, df])
            ADNI_DGP_NoNaNs_df = ADNI_DGP_NoNaNs_df.sort_values(by='RID')

        ADNI_DGP_NoNaNs_df = ADNI_DGP_NoNaNs_df.reset_index(drop=True)
        ADNI_DGP_NoNaNs_df['row_id'] = ADNI_DGP_NoNaNs_df.index
        ADNI_DGP_NoNaNs_df = ADNI_DGP_NoNaNs_df.reset_index()

        for month in months:
            df = ADNI_DGP_NoNaNs_df[ADNI_DGP_NoNaNs_df['RID'].isin(
                observed_idxs_dict[month])]
            df = df.loc[df['VISCODE'] == month]
            observed_row_ids_dict[month] = [int(x) for x in list(df.row_id)]

            df = ADNI_DGP_NoNaNs_df[ADNI_DGP_NoNaNs_df['RID'].isin(
                imputed_idxs_dict[month])]
            df = df.loc[df['VISCODE'] == month]
            imputed_row_ids_dict[month] = [int(x) for x in list(df.row_id)]

        # for i in range(0, config.num_steps * 12, 12):
        #     # print("\nMonth " + str(i) + ": Number of observed values: ",
        #           len(observed_idxs_dict[i]), len(observed_row_ids_dict[i]))
        #     # print("Month " + str(i) + ": Number of Absent values: ",
        #           len(imputed_idxs_dict[i]), len(imputed_row_ids_dict[i]))

        # cols = config.imputation_cols
        # predcols = config.all_pred_cols

        models = {
            "TAU": {"model": lrTAU,
                    "RMSE": TAURMSE},

            "PTAU": {"model": lrPTAU,
                     "RMSE": PTAURMSE},

            "FDG": {"model": lrFDG,
                    "RMSE": FDGRMSE},

            "AV45": {"model": lrAV45,
                     "RMSE": AV45RMSE},

            "ADAS13": {"model": lrADAS13,
                       "RMSE": ADAS13RMSE},

            "MMSE": {"model": lrMMSE,
                     "RMSE": MMSERMSE},

            "CDRSB": {"model": lrCDRSB,
                      "RMSE": CDRSBRMSE},

            "DX": {"model": lrDX}
        }

        additional_nan_idxs = {}

        for col in config.imputation_cols:
            #print("***\n\n", col)
            for month in range(0, config.num_steps * 12, 12):
                class_or_reg = 'REGRESSION'

                if (col == 'DX'):
                    class_or_reg = 'CLASSIFICATION'

                ADNI_DGP_NoNaNs_df = dm.fit_for_NaNs(
                    ADNI_DGP_NoNaNs_df, imputed_idxs_dict, month, col, config.all_pred_cols[col], models[col], class_or_reg)

        ADNI_DGP_NoNaNs_df[ADNI_DGP_NoNaNs_df == np.inf] = np.nan
        ADNI_DGP_NoNaNs_df.fillna(ADNI_DGP_NoNaNs_df.mean(), inplace=True)

        patient_ids = np.unique(ADNI_DGP_NoNaNs_df.RID.values)
        autoreg_key_lists_imp = dm.get_autoreg_keys(
            ADNI_DGP_NoNaNs_df, patient_ids)

        # for key, value in autoreg_key_lists_imp.items():
        #     # print("Number of autoregression samples at "
        #           + str(key) + ": " + str(len(value)))

        ADNI_DGP_NoNaNs_df['ADAS13'] = ADNI_DGP_NoNaNs_df['ADAS13'].map(
            lambda x: dm.ADAS13_cleanup(x))
        ADNI_DGP_NoNaNs_df['MMSE'] = ADNI_DGP_NoNaNs_df['MMSE'].map(
            lambda x: dm.MMSE_cleanup(x))
        ADNI_DGP_NoNaNs_df['CDRSB'] = ADNI_DGP_NoNaNs_df['CDRSB'].map(
            lambda x: dm.CDRSB_cleanup(x))

        ADNI_DGP_NoNaNs_df.to_csv(
            config.data_path + 'imputed_ADNI.csv', encoding='utf-8', index=False)

        data_cols = ['VISCODE', 'TAU', 'PTAU', 'FDG',
                     'AV45', 'ADAS13', 'MMSE', 'CDRSB', 'DX']

        orig_data = ADNI_DGP_df[data_cols]

        new_data = ADNI_DGP_NoNaNs_df[data_cols]

        # Plotting statistics
        """
        try:
            for month in months:
                for col in data_cols:
                    fig = plt.figure(figsize=(15, 4))
                    # x = fig.add_subplot(1,1,1)
                    nbins=100

                    if(len(np.array(orig_data.loc[orig_data['VISCODE'] == month][col]))>1):
                        plt.subplot(1, 2, 1)
                        plt.hist(
                            np.array(orig_data.loc[orig_data['VISCODE'] == month][col]), nbins)
                        plt.title(""+col+" original data month "+str(month))
                        plt.xlabel(col)
                        plt.grid()
                    else:
                        # print("No samples for "+col+" in original data, month "+str(month))

                    plt.subplot(1, 2, 2)
                    plt.hist(
                        np.array(new_data.loc[new_data['VISCODE'] == month][col]), nbins)
                    plt.title(""+col+" imputed data month "+str(month))
                    plt.xlabel(col)
                    plt.grid()

                    plt.savefig('plots/imputed_'+col+'_'+ \
                                str(month)+'.png', format='png', dpi=500)
                    plt.show()
        except:
            pass
            """

        if not return_metrics:
            return ADNI_DGP_NoNaNs_df, P1, PTETHCAT_probs, PTRACCAT_probs, PTGENDER_probs, autoreg_key_lists
        else:
            return ADNI_DGP_NoNaNs_df, P1, PTETHCAT_probs, PTRACCAT_probs, PTGENDER_probs, autoreg_key_lists, imputation_results

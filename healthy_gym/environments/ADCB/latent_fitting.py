from . import config
import pandas as pd
import numpy as np
import pickle

from sklearn.mixture import GaussianMixture as GMM
from sklearn.model_selection import train_test_split

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')

np.random.seed(config.rnd_seed)


class LatentStateFit:
    def __init__(self):
        # self.ADNIFilepath = ADNIFilepath
        # self.DGPcols = DGPcols
        pass

    def fitZ_baseline(self, ADNI_DGP_df, latent_dim=config.latent_dim):
        P = 0
        AB_ratio_bl_df = ADNI_DGP_df.loc[ADNI_DGP_df['VISCODE'] == 0]

        """fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(1,1,1)

        nbins=50
        plt.hist(AB_ratio_bl_df['ABETARatio'], nbins)
        plt.title("Histogram of ABETA (ABETA42/ABETA40) Ratio at baseline")
        plt.xlabel("ABETARatio")
        plt.grid()
        # plt.ylabel("Frequency")

        plt.show()"""

        #print("latent_dim: ", latent_dim)
        Z_df = self.get_all_unique_RIDs(ADNI_DGP_df)

        AB_ratio = np.array(Z_df.ABETARatio).reshape(-1, 1)
        gmm = GMM(n_components=2, random_state=0).fit(AB_ratio)
        # save gmm model for sampling later
        with open(config.fitted_gmm_Z_file, 'wb') as f:
            pickle.dump(gmm, f)

        labels = gmm.predict(AB_ratio)
        Z_df['Z'] = labels

        P = np.mean(Z_df.Z)

        if(latent_dim == 6):
            Z_df['Z'] = Z_df.apply(lambda x: self.latent_dim_6(
                z=x['Z'], APOE4=x['APOE4']), axis=1)
            P6 = Z_df['Z'].value_counts(normalize=True)
            P6 = dict(sorted(P6.items())).values()
            P6 = list(P6)
            P = P6

        patient_ids = np.unique(ADNI_DGP_df.RID.values)

        ADNI_DGP_df['Z'] = np.nan
        for rid in patient_ids:
            Z = Z_df.loc[Z_df['RID'] == rid]['Z'].values[0]
            ADNI_DGP_df.loc[ADNI_DGP_df['RID'] == rid, 'Z'] = Z

        # Hiding A-Beta Ratio
        # ADNI_DGP_df = ADNI_DGP_df.drop(['ABETARatio'], axis = 1)

        ADNI_DGP_df = ADNI_DGP_df.drop(['ABETA'], axis=1)

        #print("P: ", P)

        return ADNI_DGP_df, P

    def latent_dim_6(self, z, APOE4):
        newZ = None
        if(z == 0 and APOE4 == 0):
            newZ = 0
        elif(z == 0 and APOE4 == 1):
            newZ = 1
        elif(z == 0 and APOE4 == 2):
            newZ = 2
        elif(z == 1 and APOE4 == 0):
            newZ = 3
        elif(z == 1 and APOE4 == 1):
            newZ = 4
        elif(z == 1 and APOE4 == 2):
            newZ = 5
        return newZ

    def get_all_unique_RIDs(self, df):
        df_new = pd.DataFrame()
        unique_RIDs = list(pd.unique(df.RID))
        # print(len(unique_RIDs))
        absent_rids = []
        present_rids_df = pd.DataFrame()

        df2 = df.copy()

        for i in range(0, 132, 12):
            # print("Month: ", i)
            if (i == 0):
                present_rids_df = df[df['VISCODE'] == i][[
                    'VISCODE', 'RID', 'ABETARatio', 'TAU', 'APOE4']]
                present_rids = list(present_rids_df.RID)
                # #print(len(present_rids))

                absent_rids = list(np.setdiff1d(
                    np.array(unique_RIDs), np.array(present_rids)))
                # print("len(absent_rids): ", len(absent_rids))
                # df_absent = df[df['VISCODE']== i & df['RID'].isin(absent_rids)]
                # print("len(df_absent): ", len(df_absent))
                df_new = df_new.append(present_rids_df)

            else:
                # print("\n", i, "len(absent_rids): ", len(absent_rids))
                present_rids_df = df[(df['VISCODE'] == i) & (df['RID'].isin(absent_rids))][[
                    'VISCODE', 'RID', 'ABETARatio', 'TAU', 'APOE4']]

                present_rids = list(present_rids_df.RID)
                # print(i, "len(present_rids): ", len(present_rids))

                new_present_rids = list(set(absent_rids) & set(present_rids))
                # print("len(new_present_rids): ", len(new_present_rids))

                if(len(new_present_rids) > 0):
                    new_present_rids_df = df[(df['VISCODE'] == i) & (
                        df['RID'].isin(new_present_rids))]
                    new_present_rids_df = new_present_rids_df[[
                        'RID', 'ABETARatio', 'TAU', 'APOE4']]
                    df_new = df_new.append(new_present_rids_df)
                    # Remove from absent_rids
                    absent_rids = list(set(absent_rids)
                                       - set(new_present_rids))
                    df = df[~df['RID'].isin(new_present_rids)]
                    # print("len(absent_rids): ",len(absent_rids))

        return df_new

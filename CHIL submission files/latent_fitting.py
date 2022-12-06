import config
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
        #self.ADNIFilepath = ADNIFilepath
        #self.DGPcols = DGPcols
        pass

    def fitZ_baseline(self, ADNI_DGP_df, n_components=config.n_components):
        AB_ratio_bl_df = ADNI_DGP_df.loc[ADNI_DGP_df['VISCODE'] == 0]

        """fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(1,1,1)

        nbins=50
        plt.hist(AB_ratio_bl_df['ABETARatio'], nbins)
        plt.title("Histogram of ABETA (ABETA42/ABETA40) Ratio at baseline")
        plt.xlabel("ABETARatio")
        plt.grid()
        #plt.ylabel("Frequency")

        plt.show()"""

        Z_df = self.get_all_unique_RIDs(ADNI_DGP_df)

        if(n_components == 2):
            AB_ratio = np.array(Z_df.ABETARatio).reshape(-1, 1)
            gmm = GMM(n_components=2, random_state=0).fit(AB_ratio)
            # save gmm model for sampling later
            with open('data/fitted_gmm_Z.pickle', 'wb') as f:
                pickle.dump(gmm, f)

            labels = gmm.predict(AB_ratio)
            Z_df['Z'] = labels

            patient_ids = np.unique(ADNI_DGP_df.RID.values)

            ADNI_DGP_df['Z'] = np.nan
            for rid in patient_ids:
                Z = Z_df.loc[Z_df['RID'] == rid]['Z'].values[0]
                ADNI_DGP_df.loc[ADNI_DGP_df['RID'] == rid, 'Z'] = Z

            # Hiding A-Beta Ratio
            #ADNI_DGP_df = ADNI_DGP_df.drop(['ABETARatio'], axis = 1)

            ADNI_DGP_df = ADNI_DGP_df.drop(['ABETA'], axis=1)

            P1 = np.mean(Z_df.Z)

            #ADNI_DGP_df['proZ'] = ADNI_DGP_df.groupby('Z')['Z'].transform(lambda x : x.count()/len(ADNI_DGP_df))
            #groupZ = ADNI_DGP_df.groupby('Z')
            #Z_probs = pd.DataFrame(groupZ.apply(lambda x: x['proZ'].unique()[0]), columns=["prob"])
            # print(Z_probs)

            #Z_probs = Z_probs.reset_index(drop=False)
            #ADNI_DGP_df = ADNI_DGP_df.drop(['proZ'], axis = 1)

            return ADNI_DGP_df, P1

    def get_all_unique_RIDs(self, df):
        df_new = pd.DataFrame()
        unique_RIDs = list(pd.unique(df.RID))
        print(len(unique_RIDs))
        absent_rids = []
        present_rids_df = pd.DataFrame()

        df2 = df.copy()

        for i in range(0, 132, 12):
            #print("Month: ", i)
            if (i == 0):
                present_rids_df = df[df['VISCODE'] == i][[
                    'VISCODE', 'RID', 'ABETARatio', 'TAU']]
                present_rids = list(present_rids_df.RID)
                # print(len(present_rids))

                absent_rids = list(np.setdiff1d(
                    np.array(unique_RIDs), np.array(present_rids)))
                #print("len(absent_rids): ", len(absent_rids))
                #df_absent = df[df['VISCODE']== i & df['RID'].isin(absent_rids)]
                #print("len(df_absent): ", len(df_absent))
                df_new = df_new.append(present_rids_df)

            else:
                #print("\n", i, "len(absent_rids): ", len(absent_rids))
                present_rids_df = df[(df['VISCODE'] == i) & (df['RID'].isin(absent_rids))][[
                    'VISCODE', 'RID', 'ABETARatio', 'TAU']]

                present_rids = list(present_rids_df.RID)
                #print(i, "len(present_rids): ", len(present_rids))

                new_present_rids = list(set(absent_rids) & set(present_rids))
                #print("len(new_present_rids): ", len(new_present_rids))

                if(len(new_present_rids) > 0):
                    new_present_rids_df = df[(df['VISCODE'] == i) & (
                        df['RID'].isin(new_present_rids))]
                    new_present_rids_df = new_present_rids_df[[
                        'RID', 'ABETARatio', 'TAU']]
                    df_new = df_new.append(new_present_rids_df)
                    # Remove from absent_rids
                    absent_rids = list(set(absent_rids) -
                                       set(new_present_rids))
                    df = df[~df['RID'].isin(new_present_rids)]
                    #print("len(absent_rids): ",len(absent_rids))

        return df_new

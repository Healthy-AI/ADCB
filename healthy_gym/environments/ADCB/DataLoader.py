from . import config

import pandas as pd
import numpy as np
from sklearn import preprocessing

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')

np.random.seed(config.rnd_seed)


class DataLoader:
    def __init__(self):
        #self.ADNIFilepath = ADNIFilepath
        #self.DGPcols = DGPcols
        pass

    def loadData(self, ADNIFilepath, DGPcols):
        Adni = pd.read_csv(ADNIFilepath)  # ../"data/ADNIMERGE.csv"
        Adni.loc[:, 'ABETAHigh'] = Adni['ABETA'] == '>1700'
        Adni.ABETAHigh = Adni.ABETAHigh.astype(int)

        Adni.loc[:, 'ABETALow'] = Adni['ABETA'] == '<200'
        Adni.ABETALow = Adni.ABETALow.astype(int)

        Adni.loc[:, 'TAUHigh'] = Adni['TAU'] == '>1300'
        Adni.TAUHigh = Adni.TAUHigh.astype(int)

        Adni.loc[:, 'TAULow'] = Adni['TAU'] == '<80'
        Adni.TAULow = Adni.TAULow.astype(int)

        Adni.loc[:, 'PTAUHigh'] = Adni['PTAU'] == '>120'
        Adni.PTAUHigh = Adni.PTAUHigh.astype(int)

        Adni.loc[:, 'PTAULow'] = Adni['PTAU'] == '<8'
        Adni.PTAULow = Adni.PTAULow.astype(int)

        Adni = Adni.replace({'ABETA': '>1700'}, 1700)
        Adni = Adni.replace({'TAU': '>1300'}, 1300)
        Adni = Adni.replace({'ABETA': '<200'}, 200)
        Adni = Adni.replace({'TAU': '<80'}, 80)
        Adni = Adni.replace({'PTAU': '>120'}, 120)
        Adni = Adni.replace({'PTAU': '<8'}, 8)
        Adni.TAU = Adni.TAU.astype('float64')
        Adni.ABETA = Adni.ABETA.astype('float64')
        Adni.PTAU = Adni.PTAU.astype('float64')

        #ADNI_LIPIDOMICSRADER = pd.read_csv('ADNI_LIPIDOMICSRADER.csv')
        UPENNMSMSABETA2 = pd.read_csv(
            config.ABETA_filespath + 'UPENNMSMSABETA2.csv')
        UPENNMSMSABETA2 = UPENNMSMSABETA2[[
            'RID', 'VISCODE2', 'VID', 'ABETA42', 'ABETA40', 'ABETA38']]

        ADNI_MESOSCALE = pd.read_csv(
            config.ABETA_filespath + 'ADNI_MESOSCALE.csv')
        ADNI_MESOSCALE = ADNI_MESOSCALE[[
            'RID', 'VISCODE2', 'ABETA38', 'ABETA40', 'ABETA42', 'TAU']]
        ADNI_MESOSCALE = ADNI_MESOSCALE.rename(
            columns={'ABETA38': 'ABETA38_extra2', 'ABETA40': 'ABETA40_extra2', 'ABETA42': 'ABETA42_extra2', 'TAU': 'TAU_extra2'})

        """UPENNPLASMA = pd.read_csv('../data/ABETA_files/UPENNPLASMA.csv')
        UPENNPLASMA = UPENNPLASMA[['RID','VISCODE','AB40','AB42']]
        UPENNPLASMA = UPENNPLASMA.rename(columns={'AB40': 'ABETA40_PLASMA'})
        UPENNPLASMA = UPENNPLASMA.rename(columns={'AB42': 'ABETA42_PLASMA'})"""

        UPENNMSMSABETA = pd.read_csv(
            config.ABETA_filespath + 'UPENNMSMSABETA.csv')
        UPENNMSMSABETA = UPENNMSMSABETA[[
            'RID', 'VISCODE', 'ABETA42', 'ABETA40', 'ABETA38']]
        UPENNMSMSABETA = UPENNMSMSABETA.rename(
            columns={'ABETA42': 'ABETA42_extra', 'ABETA40': 'ABETA40_extra', 'ABETA38': 'ABETA38_extra'})

        FUJIREBIOABETA = pd.read_csv(
            config.ABETA_filespath + 'FUJIREBIOABETA.csv')
        FUJIREBIOABETA = FUJIREBIOABETA.drop(columns={
                                             'VISCODE', 'RUN', 'RUNDATE', 'COMMENTS', 'GUSPECID', 'VID', 'DRAWDTE', 'DER', 'SITE'})
        FUJIREBIOABETA = FUJIREBIOABETA.rename(
            columns={'VISCODE2': 'VISCODE_FUJIREBIOABETA', 'ABETA40': 'ABETA40_extra3', 'ABETA42': 'ABETA42_extra3', 'ABETA42_40': 'ABETA42_40_extra3'})

        theData = pd.merge(Adni,
                           UPENNMSMSABETA2,
                           right_on=['RID', 'VISCODE2'],
                           left_on=['RID', 'VISCODE'],
                           how='left')
        theData = pd.merge(theData,
                           UPENNMSMSABETA,
                           on=['RID', 'VISCODE'],
                           how='left')
        theData = pd.merge(theData,
                           ADNI_MESOSCALE,
                           right_on=['RID', 'VISCODE2'],
                           left_on=['RID', 'VISCODE'],
                           how='left')
        theData = pd.merge(theData,
                           FUJIREBIOABETA,
                           right_on=['RID', 'VISCODE_FUJIREBIOABETA'],
                           left_on=['RID', 'VISCODE'],
                           how='left')

        theData.ABETA40 = theData.ABETA40.astype('float64')
        theData.ABETA42 = theData.ABETA42.astype('float64')

        theData = theData[theData['ABETA42'].notna()]
        theData = theData[theData['ABETA40'].notna()]
        # #print(list(theData.columns))

        # Selecting Columns of Interest
        ADNI_DGP_df = theData[DGPcols]

        # Computing ABeta Ratio
        ADNI_DGP_df['ABETARatio'] = ADNI_DGP_df['ABETA42'] / \
            ADNI_DGP_df['ABETA40']
        ADNI_DGP_df = ADNI_DGP_df.drop(['ABETA42'], axis=1)
        ADNI_DGP_df = ADNI_DGP_df.drop(['ABETA40'], axis=1)

        ADNI_DGP_df.MMSE = ADNI_DGP_df.MMSE.astype('float64')
        ADNI_DGP_df.ADAS13 = ADNI_DGP_df.ADAS13.astype('float64')

        ADNI_DGP_df.APOE4 = ADNI_DGP_df.APOE4.astype(int)
        ADNI_DGP_df.PTEDUCAT = ADNI_DGP_df.PTEDUCAT.astype(int)

        ADNI_DGP_df = self.data_preprocessing(ADNI_DGP_df)
        return ADNI_DGP_df

    def data_preprocessing(self, ADNI_DGP_df):
        ADNI_DGP_df.PTGENDER = ADNI_DGP_df.PTGENDER.astype(
            'category').cat.codes
        ADNI_DGP_df.APOE4 = ADNI_DGP_df.APOE4.astype('category').cat.codes

        #ADNI_DGP_df.PTEDUCAT = ADNI_DGP_df.PTEDUCAT.astype('category').cat.codes
        #ADNI_DGP_df.PTETHCAT = ADNI_DGP_df.PTETHCAT.astype('category').cat.codes
        #ADNI_DGP_df.PTRACCAT = ADNI_DGP_df.PTRACCAT.astype('category').cat.codes
        #ADNI_DGP_df.PTMARRY = ADNI_DGP_df.PTMARRY.astype('category').cat.codes
        #ADNI_DGP_df.DX = ADNI_DGP_df.DX.astype('category').cat.codes
        #ADNI_DGP_df['DX_codes'] = ADNI_DGP_df.DX.cat.codes

        standardscaler = preprocessing.StandardScaler()

        #standard_df = standardscaler.fit_transform(ADNI_DGP_df[["FDG", "AV45", "ABETA", "TAU", "PTAU", "ADAS13", "MMSE"]])
        #standard_df = pd.DataFrame(standard_df, columns =["FDG", "AV45", "ABETA", "TAU", "PTAU", "ADAS13", "MMSE"])

        # ADNI_DGP_df[["FDG", "AV45", "ABETA", "TAU", "PTAU", "ADAS13", "MMSE"]] = standard_df[["FDG", "AV45", "ABETA", "TAU",
        #                                                                                      "PTAU", "ADAS13", "MMSE"]]

        ADNI_DGP_df['DX'] = ADNI_DGP_df['DX'].map(config.DX_Codes)
        ADNI_DGP_df['VISCODE'] = ADNI_DGP_df['VISCODE'].map(config.months)
        ADNI_DGP_df['PTEDUCAT'] = ADNI_DGP_df['PTEDUCAT'].map(
            config.EDUCAT_YEAR_map)
        ADNI_DGP_df['PTETHCAT'] = ADNI_DGP_df['PTETHCAT'].map(
            config.PTETHCAT_Codes)
        ADNI_DGP_df['PTRACCAT'] = ADNI_DGP_df['PTRACCAT'].map(
            config.PTRACCAT_Codes)
        ADNI_DGP_df['PTMARRY'] = ADNI_DGP_df['PTMARRY'].map(
            config.PTMARRY_Codes)
        return ADNI_DGP_df

    def race_gender_bl_statistics(self, ADNI_DGP_df):
        AB_ratio_bl_df = ADNI_DGP_df.loc[ADNI_DGP_df['VISCODE'] == 0]
        AB_ratio_bl_df['proPTETHCAT'] = AB_ratio_bl_df.groupby('PTETHCAT')['PTETHCAT'].transform(lambda x:
                                                                                                 x.count() / len(AB_ratio_bl_df))
        AB_ratio_bl_df['proPTRACCAT'] = AB_ratio_bl_df.groupby('PTRACCAT')['PTRACCAT'].transform(lambda x:
                                                                                                 x.count() / len(AB_ratio_bl_df))
        AB_ratio_bl_df['proPTGENDER'] = AB_ratio_bl_df.groupby('PTGENDER')['PTGENDER'].transform(lambda x:
                                                                                                 x.count() / len(AB_ratio_bl_df))

        group1 = AB_ratio_bl_df.groupby('PTETHCAT')
        PTETHCAT_probs = pd.DataFrame(group1.apply(
            lambda x: x['proPTETHCAT'].unique()[0]), columns=["prob"])
        PTETHCAT_probs = PTETHCAT_probs.reset_index(drop=False)

        group2 = AB_ratio_bl_df.groupby('PTRACCAT')
        PTRACCAT_probs = pd.DataFrame(group2.apply(
            lambda x: x['proPTRACCAT'].unique()[0]), columns=["prob"])
        PTRACCAT_probs = PTRACCAT_probs.reset_index(drop=False)

        group3 = AB_ratio_bl_df.groupby('PTGENDER')
        PTGENDER_probs = pd.DataFrame(group3.apply(
            lambda x: x['proPTGENDER'].unique()[0]), columns=["prob"])
        PTGENDER_probs = PTGENDER_probs.reset_index(drop=False)

        AB_ratio_bl_df = AB_ratio_bl_df.drop(['proPTETHCAT'], axis=1)
        AB_ratio_bl_df = AB_ratio_bl_df.drop(['proPTRACCAT'], axis=1)
        AB_ratio_bl_df = AB_ratio_bl_df.drop(['proPTGENDER'], axis=1)

        return PTETHCAT_probs, PTRACCAT_probs, PTGENDER_probs

import shutil

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import skimage.transform as trans

import numpy as np
import skimage.io as io
from numpy import unique

from code.params import IMG_SIZE


def read_fnol_data(path="../../data/"):
    df_train = pd.read_csv(path + "train_ds.csv")
    df_test = pd.read_csv(path + "test_ds.csv")

    return df_train, df_test

def get_data(df):
    imgs = []
    text = []
    labels = []
    select_feature = unique(['pneumonia', 'CS_another_problems', 'CS', 'olfactory_loss', 'postcovid_disability',
                             'SARS - CoV - 2 IgG(qualit.)', 'SARS - CoV - 2 IgM(quant.)', 'CS_amount', 'CS_total',
                             'CS_duration_weeks', 'KO_RDW', 'VC(abs)', 'FVC(abs)', 'FEV1(abs)', 'KO_Mo %',
                             'PEF( % pred)', 'DLCOc_SB(abs)', 'KCO_SB(abs)', 'persistent_cough', 'persistent_dyspnea',
                             'MEF25(abs)',
                             ])

    for index, row in df.iterrows():
        img_1 = io.imread(row['first_examination_file'])
        img_1 = img_1 / 255
        img_1 = trans.resize(img_1, (IMG_SIZE, IMG_SIZE, 3))
        # img_1 = np.reshape(img_1, (1,) + img_1.shape)
        text_1 = row[select_feature]
        # text_1 = row.drop(labels=drop_features)
        text_1 = np.array(text_1).astype("float")
        label_1 = int(row['CS_and_improved'])

        imgs.append(img_1)
        text.append(text_1)
        labels.append(label_1)

    return np.array(imgs), np.array(text), np.array(labels)

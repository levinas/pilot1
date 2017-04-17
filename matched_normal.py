#! /usr/bin/env python

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import re

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input
from keras.callbacks import Callback, ModelCheckpoint



N1 = 2000
NE = 600


def transform():
    df = pd.read_csv('matched_normal_samples.FPKM-UQ.csv')
    df = df.set_index('Sample')
    df = df.applymap(lambda x: '%.5g' % np.log(1 + x))
    df = df.reset_index()

    df_meta = pd.read_csv('matched_normal_samples/metadata', sep='\t', header=None)

    idmap = {}
    for i, row in df_meta.iterrows():
        sample = row[0]
        id0 = re.search('(.+)\.FPKM-UQ', row[4]).group(1)  # Solid Tissue Normal
        id1 = re.search('(.+)\.FPKM-UQ', row[6]).group(1)  # Primary Tumor
        idmap[id0] = '{}-{}'.format(i, 0)
        idmap[id1] = '{}-{}'.format(i, 1)

    df['Sample'] = df['Sample'].map(idmap)
    df = df.sort_values('Sample')
    df.to_csv('transformed.csv', index=False)


def plot_df(df, mat, png):
    df2 = pd.DataFrame(mat, index=df.index, columns=['d1', 'd2'])
    df2 = df2.reset_index()
    df2['type'] = df2.apply(lambda r: 'Normal' if r['Sample'].endswith('0') else 'Tumor', axis=1)
    g = sns.lmplot('d1', 'd2', df2, hue='type', fit_reg=False, size=8, scatter_kws={'alpha': 0.7, 's': 60})
    g.savefig(png)


def plot_pca2(df):
    pca = PCA(n_components=2)
    mat = pca.fit_transform(df.as_matrix())
    plot_df(df, mat, 'pca2.png')


def plot_pca20_tsne(df):
    pca = PCA(n_components=20)
    mat = pca.fit_transform(df.as_matrix())
    tsne = TSNE(n_components=2, random_state=1)
    mat2 = tsne.fit_transform(mat)
    plot_df(df, mat2, 'pca20_tsne.png')


def plot_tsne(df):
    tsne = TSNE(n_components=2, random_state=1)
    mat = tsne.fit_transform(df.as_matrix())
    plot_df(df, mat, 'tsne.png')


def auen(df):
    x_train = df.as_matrix()
    input_dim = x_train.shape[1]

    input_vector = Input(shape=(input_dim,))
    h = Dense(N1, activation='sigmoid')(input_vector)
    h = Dense(NE, activation='sigmoid')(h)
    encoded = h

    h = Dense(N1, activation='sigmoid')(h)
    h = Dense(input_dim, activation='sigmoid')(h)

    ae = Model(input_vector, h)
    ae.summary()

    encoded_input = Input(shape=(NE,))
    decoder = Model(encoded_input, ae.layers[-1](ae.layers[-2](encoded_input)))
    encoder = Model(input_vector, encoded)

    ae.compile(optimizer='rmsprop', loss='mse')

    ae.fit(x_train, x_train,
           batch_size=100,
           epochs=2,
           validation_split=0.2)

    latent = encoder.predict(x_train)
    plot_tsne(pd.DataFrame(latent, index=df.index))


def classify(df):
    x_train = df.as_matrix()
    y_train = np.array([0 if x.endswith('0') else 1 for x in df.index.tolist()])
    input_dim = x_train.shape[1]

    activation = 'sigmoid'
    model = Sequential()
    model.add(Dense(1000, input_dim=input_dim, activation=activation))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation=activation))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=100,
              epochs=2,
              validation_split=0.2)


def main():
    # transform()
    df = pd.read_csv('transformed.csv', engine='c')
    df = df.set_index('Sample')

    # scaler = StandardScaler()
    scaler = MaxAbsScaler()
    mat = df.as_matrix().astype(np.float32)
    mat = scaler.fit_transform(mat)
    df = pd.DataFrame(mat, index=df.index, columns=df.columns)

    # auen(df)

    # plot_pca2(df)
    # plot_pca20_tsne(df)

    classify(df)




if __name__ == '__main__':
    main()

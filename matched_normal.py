#! /usr/bin/env python

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import re

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input
from keras.callbacks import Callback, ModelCheckpoint



N1 = 1000
NE = 100


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
    g = sns.lmplot('d1', 'd2', df2, hue='type', fit_reg=False, size=8, scatter_kws={'alpha': 0.7, 's': 50})
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


def plot_tsne(df, png):
    tsne = TSNE(n_components=2, random_state=1)
    mat = tsne.fit_transform(df.as_matrix())
    plot_df(df, mat, png)


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
           epochs=100,
           validation_split=0.1)

    latent = encoder.predict(x_train)
    plot_tsne(pd.DataFrame(latent, index=df.index), 'tsne.png')


def denoising_auen(df):
    x_all = df.sample(frac=1.0).as_matrix()
    val_pos = int(x_all.shape[0] * 0.1)
    x_val = x_all[-val_pos:]
    x_train = x_all[:-val_pos]

    x_train_noisy = None
    x_train_target = None

    noise_factor = 0.1
    for _ in range(10):
        noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
        x_train_noisy = np.concatenate((x_train_noisy, noisy)) if x_train_noisy is not None else noisy
        x_train_target = np.concatenate((x_train_target, x_train)) if x_train_target is not None else x_train

    print(x_train_noisy.shape)

    input_dim = x_train_noisy.shape[1]

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

    checkpointer = ModelCheckpoint(filepath="dae_weights.hdf5", save_best_only=True)

    ae.fit(x_train_noisy, x_train_target,
           batch_size=100,
           epochs=40,
           callbacks=[checkpointer],
           validation_data=(x_val, x_val))

    latent = encoder.predict(x_all)
    df_latent = pd.DataFrame(latent, index=df.index)
    df_latent.to_csv('dae_latent.csv')
    plot_tsne(df_latent, 'dae_tsne.png')

    y_all = np.array([0 if x.endswith('0') else 1 for x in df.index.tolist()])

    for i, column in enumerate(latent.T):
        clf = LogisticRegression()
        x1 = column.reshape(-1, 1)
        clf.fit(x1, y_all)
        score = clf.score(x1, y_all)
        print(i, score)


def sprint_features(top_features, num_features=100):
    str = ''
    for i, feature in enumerate(top_features):
        if i >= num_features:
            break
        str += '{}\t{:.5f}\n'.format(feature[1], feature[0])
    return str


def classify_xgboost(df):
    x_train = df.as_matrix()
    y_train = np.array([0 if x.endswith('0') else 1 for x in df.index.tolist()])
    clf = XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.05)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print(scores)
    print(np.mean(scores))


def classify_rf(df):
    x_train = df.as_matrix()
    y_train = np.array([0 if x.endswith('0') else 1 for x in df.index.tolist()])
    clf = RandomForestClassifier(n_estimators=10)
    scores = cross_val_score(clf, x_train, y_train, cv=3)
    print(scores)
    print(np.mean(scores))
    clf.fit(x_train, y_train)
    fi = clf.feature_importances_
    features = [(f, n) for f, n in zip(fi, df.columns.tolist())]
    top = sorted(features, key=lambda f:f[0], reverse=True)[:1]
    with open("RF.top_features", "w") as fea_file:
        fea_file.write(sprint_features(top))
    print(top)

    x_train_top = df[[x[1] for x in top]].as_matrix()
    clf2 = RandomForestClassifier()
    scores2 = cross_val_score(clf2, x_train_top, y_train, cv=3)
    print(scores2)
    print(np.mean(scores2))


def classify(df):
    x_train = df.as_matrix()
    y_train = np.array([0 if x.endswith('0') else 1 for x in df.index.tolist()])
    input_dim = x_train.shape[1]

    activation = 'relu'
    model = Sequential()
    model.add(Dense(20, input_dim=input_dim, activation=activation))
    # model.add(Dropout(0.2))
    # model.add(Dense(20, activation=activation))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=100,
              epochs=10,
              validation_split=0.2)


def main():
    # transform()
    # df = pd.read_csv('transformed.csv', engine='c')
    # df.to_hdf('transformed.h5', 'df')
    df = pd.read_hdf('transformed.h5')
    df = df.set_index('Sample')

    scaler = StandardScaler()
    # scaler = MaxAbsScaler()
    mat = df.as_matrix().astype(np.float32)
    mat = scaler.fit_transform(mat)
    df = pd.DataFrame(mat, index=df.index, columns=df.columns)

    # auen(df)
    denoising_auen(df)

    # plot_pca2(df)
    # plot_pca20_tsne(df)

    # classify_xgboost(df)
    # classify_rf(df)
    # classify(df)




if __name__ == '__main__':
    main()

#! /usr/bin/env python

from __future__ import print_function

import fnmatch
import gzip
import os
import re

import numpy as np
import pandas as pd

from sklearn.feature_extraction import DictVectorizer


def gen_find(pat, top='.'):
    for path, dirlist, filelist in os.walk(top):
        for name in fnmatch.filter(filelist, pat):
            yield os.path.join(path, name)


def make_matrix():
    fnames = gen_find('*FPKM-UQ.txt', 'matched_normal_samples')
    samples = []
    values = []
    for i, fname in enumerate(fnames):
        print(i, fname)
        df = pd.read_csv(fname, sep='\t', header=None)
        df = df.set_index(0)
        values.append(df.to_dict()[1])
        sample_id = re.search('([^/]+)\.FPKM-UQ', fname).group(1)
        samples.append(sample_id)

    dv = DictVectorizer()
    f = dv.fit_transform(values)
    print(f.shape)

    df = pd.DataFrame(f.toarray(), columns=dv.get_feature_names())
    df['Sample'] = samples
    df = df.set_index('Sample')
    df.to_csv('matched_normal_samples.FPKM-UQ.csv')


def main():
    make_matrix()


if __name__ == '__main__':
    main()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

df = pd.read_csv('enc.ae500.c20.csv')

tsne = TSNE(n_components=2, random_state=0)

Z2 = tsne.fit_transform(df.iloc[:, 1:])

df2 = pd.DataFrame(Z2, columns=['x','y'])
df2['type'] = df['Class']

g = sns.lmplot('x', 'y', df2, hue='type', fit_reg=False, size=8, scatter_kws={'alpha':0.7,'s':60})

g.savefig('tsne.ae.png')

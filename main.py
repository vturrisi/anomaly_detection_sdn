import os
from contextlib import suppress
from itertools import combinations, cycle

import pandas as pd
from sklearn import preprocessing

from denstream import DenStream, gen_data_plot, plot_clusters


df = pd.read_csv('datasets/051218_60h6sw_c1_ht5_it0_V2_csv.csv')
features = df.columns[:-1]

data = df.values
# data = data[29000:]

X = data[:, :-1]
Y = data[:, -1]

scaler = preprocessing.MinMaxScaler()
scaler.fit(X)
Xscaled = scaler.transform(X)


for f1, f2 in combinations(range(6), 2):
    ds = DenStream(n_features=2, lambda_=0.1, beta=0.3,
                   epsilon=0.1, mu=0.3, stream_speed=5000,
                   min_points=100)

    X = Xscaled[:, [f1, f2]]

    xfeature_name = features[f1]
    yfeature_name = features[f2]
    print('runing for {} and {}'.format(xfeature_name, yfeature_name))

    folder = os.path.join('plots', '051218_no_infection',
        '{} x {}'.format(xfeature_name, yfeature_name))

    with suppress(Exception):
        os.makedirs(folder)

    for i, (x, y) in enumerate(zip(X, Y)):
        ds.train(x, y)

        window = 10000
        if i != 0 and i % 5000 == 0:
            print(i, end=' ')
            start = i - window
            if start < 0:
                start = 0
            points = list(zip(X[start: i], Y[start: i]))
            normal_points, outliers, c_clusters, \
                p_clusters, outlier_clusters = gen_data_plot(ds, points)


            fname= os.path.join(folder, '{}.png'.format(i))

            plot_clusters(fname, normal_points, outliers, c_clusters, p_clusters,
                          outlier_clusters, ds._epsilon, title='plot {}'.format(i),
                          xfeature_name=xfeature_name, yfeature_name=yfeature_name)

    print()
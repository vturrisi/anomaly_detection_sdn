import os
from contextlib import suppress
from itertools import combinations, cycle, product

import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm

from denstream import DenStream, gen_data_plot, plot_clusters, save_clusters_info

names = ['051218',
         '051218_no_infec',
         '171218']


log_folder = 'csv_logs'


with suppress(Exception):
    os.mkdir(log_folder)


# if 'hp_list.pkl' in os.listdir():
#     import pickle
#     hyperparameters = pickle.load(open('hp_list.pkl', 'rb'))
#     lambdas = hyperparameters['lambdas']
#     betas = hyperparameters['betas']
#     epsilons = hyperparameters['epsilons']
#     mus = hyperparameters['mus']
#     speeds = hyperparameters['speeds']

# else:
#     lambdas = np.random.uniform(0, 0.2, size=5)
#     betas = np.random.uniform(0, 0.6, size=5)
#     epsilons = [0.05, 0.10, 0.15]
#     mus = [50, 100, 250, 500, 1000]
#     speeds = [100, 250, 500, 1000]

#     import pickle
#     hyperparameters = {'lambdas': lambdas,
#                        'betas': betas,
#                        'epsilons': epsilons,
#                        'mus': mus,
#                        'speeds': speeds}
#     pickle.dump(hyperparameters, open('hp_list.pkl', 'wb'))

lambdas = [0.06807737612366145]
betas = [0.3004826601733964]
epsilons = [0.05]
mus = [250]
speeds = [1000]

for name, dataset in zip(names, ['051218_60h6sw_c1_ht5_it0_V2_csv_ddos_portscan.csv',
                                 '051218_60h6sw_c1_ht5_it0_V2_csv.csv',
                                 '171218_60h6sw_c1_ht5_it0_V2_csv_portscan_ddos.csv']):
    dataset = os.path.join('datasets', dataset)
    for lambda_, beta, ep, mu, speed in product(lambdas, betas, epsilons, mus, speeds):
        fname = f'{name}_lambda={lambda_}_beta={beta}_ep={ep}_mu={mu}_speed={speed}'

        df = pd.read_csv(dataset)
        features = list(df.columns[:-1])

        data = df.values
        # data = data[29000:]

        X = data[:, :-1]
        Y = data[:, -1]

        scaler = preprocessing.MinMaxScaler()
        scaler.fit(X)
        Xscaled = scaler.transform(X)

        features_to_use = ('H_src_ip', 'H_dst_ip', 'H_src_port', 'H_dst_port')

        f1, f2, f3, f4 = features_to_use
        f1 = features.index(f1)
        f2 = features.index(f2)
        f3 = features.index(f3)
        f4 = features.index(f4)

        ds = DenStream(n_features=4, lambda_=lambda_, beta=beta,
                       epsilon=ep, mu=mu, stream_speed=speed,
                       min_points=100)

        X = Xscaled[:, [f1, f2, f3, f4]]

        result_file = os.path.join(log_folder, fname + '.csv')
        try:
            os.remove(result_file)
        except:
            pass

        for i, (x, y) in tqdm(list(enumerate(zip(X, Y)))):
            ds.train(x, y)

            window = 10
            if i != 0 and i % 50 == 0:
                start = i - window
                if start < 0:
                    start = 0
                points = list(zip(X[start: i], Y[start: i]))
                normal_points, outliers, c_clusters, \
                    p_clusters, outlier_clusters = gen_data_plot(ds, points)

                save_clusters_info(i, ds, result_file)

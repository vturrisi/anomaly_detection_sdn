import os
import pickle
from contextlib import suppress

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

datasets_folder = 'datasets'
dump_folder = 'dump_logs'
dump_files = os.listdir(dump_folder)
plot_folder = 'plots'

with suppress(Exception):
    os.mkdir(plot_folder)

for dataset, filterfunc, dfname in [('051218',
                                     lambda f: '051218' in f and 'no_infec' not in f,
                                     os.path.join(datasets_folder,
                                                  '051218_60h6sw_c1_ht5_it0_V2_csv_ddos_portscan.csv')),

                                    ('051218_no_infec',
                                     lambda f: '051218_no_infec' in f,
                                     os.path.join(datasets_folder,
                                                  '051218_60h6sw_c1_ht5_it0_V2_csv.csv')),

                                    ('171218',
                                     lambda f: '171218' in f,
                                     os.path.join(datasets_folder,
                                                  '171218_60h6sw_c1_ht5_it0_V2_csv_portscan_ddos.csv'))]:
    df = pd.read_csv(dfname)

    changes = []
    current_c = 0
    for i, c in enumerate(df['class'].values):
        if c != current_c:
            current_c = c
            changes.append((c, i))

    for f in tqdm(list(filter(filterfunc, dump_files))):
        data = pickle.load(open(os.path.join(dump_folder, f), 'rb'))

        x = list(data.keys())
        n_c_clusters = []
        n_p_clusters = []
        n_o_clusters = []
        for instance, partial in data.items():
            n_c_clusters.append(partial['n_c_clusters'])
            n_p_clusters.append(partial['n_p_clusters'])
            n_o_clusters.append(partial['n_o_clusters'])

        plt.plot(x, n_c_clusters, c='blue', label='number of c_clusters')
        plt.plot(x, n_p_clusters, c='red', label='number of p_clusters')
        plt.plot(x, n_o_clusters, c='green', label='number of o_clusters')
        plotted_malicious_to_normal = False
        plotted_normal_to_malicious = False
        for class_, change in changes:
            if class_ == 0:
                color = 'black'
                if not plotted_malicious_to_normal:
                    label = 'changed from malicious to legit'
                    plotted_malicious_to_normal = True
                else:
                    label = None
            else:
                color = 'magenta'
                if not plotted_malicious_to_normal:
                    label = 'changed from legit to malicious'
                    plotted_normal_to_malicious = True
                else:
                    label = None
            plt.plot([change, change], [0, 6], c=color, linestyle='--', label=label)

        plt.legend(loc='upper left', fontsize=8)
        plt.ylim(0, 8)
        plt.xlabel('Number of instances')
        plt.ylabel('Number of micro clusters')
        name = f.replace('.pkl', '.pdf')
        plt.savefig(os.path.join(plot_folder, name))
        plt.close()

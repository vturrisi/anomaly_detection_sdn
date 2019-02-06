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
        plt.figure(figsize=(7, 5))

        x = list(data.keys())
        n_c_clusters = []
        n_p_clusters = []
        n_o_clusters = []
        for instance, partial in data.items():
            n_c_clusters.append(partial['n_c_clusters'])
            n_p_clusters.append(partial['n_p_clusters'])
            n_o_clusters.append(partial['n_o_clusters'])

        plt.plot(x, n_o_clusters, c='red', label='O-MC', alpha=0.4, linewidth=1.8)
        plt.plot(x, n_c_clusters, c='blue', label='C-MC', alpha=0.4, linewidth=1.8)
        plt.plot(x, n_p_clusters, c='green', label='P-MC', linewidth=1.8)
        plotted_malicious_to_normal = False
        plotted_normal_to_malicious = False

        if changes:
            change = changes[0][1]
            if '051218' in dataset:
                label = 'DDoS'
                color = 'magenta'
            else:
                label = 'PortScan'
                color = 'orange'
            plt.plot([change, change], [0, 17], c=color, linestyle='--', label=label, linewidth=1.8)

            change = changes[1][1]
            plt.plot([change, change], [0, 17], c='black', linestyle=':', label='End of infection', linewidth=1.8)

            change = changes[2][1]
            if '051218' in dataset:
                label = 'PortScan'
                color = 'orange'
            else:
                label = 'DDoS'
                color = 'magenta'
            plt.plot([change, change], [0, 17], c=color, linestyle='--', label=label, linewidth=1.8)

            change = changes[3][1]
            plt.plot([change, change], [0, 17], c='black', linestyle=':', label=None, linewidth=1.8)

            # For datasets concatenated
            #
            # change = changes[4][1]
            # if '051218' in dataset:
            #     label = 'DDOS'
            #     color = 'magenta'
            # else:
            #     label = 'PortScan'
            #     color = 'orange'
            # plt.plot([change, change], [0, 17], c=color, linestyle='--', label=None)

            # change = changes[5][1]
            # plt.plot([change, change], [0, 17], c='black', linestyle=':', label=None)

            # change = changes[6][1]
            # if '051218' in dataset:
            #     label = 'PortScan'
            #     color = 'orange'
            # else:
            #     label = 'DDOS'
            #     color = 'magenta'
            # plt.plot([change, change], [0, 17], c=color, linestyle='--', label=None)

            # change = changes[7][1]
            # plt.plot([change, change], [0, 17], c='black', linestyle=':', label=None)

        # plt.legend(loc='upper center', fontsize=8, ncol=6, bbox_to_anchor=(0, 0.15, 1, 1), fancybox=True)
        plt.ylim(0, 17)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Number of instances', fontsize=20)
        plt.ylabel('Number of micro clusters', fontsize=20)
        name = f[:-4].replace('.', '_') + '.pdf'
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, name))
        plt.close()

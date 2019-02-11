import os
import pickle
from collections import defaultdict
from contextlib import suppress

import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from tqdm import tqdm

datasets_folder = 'datasets'
dump_folder = 'dump_logs'
dump_files = os.listdir(dump_folder)
plot_folder = 'plots_weights'

with suppress(Exception):
    os.mkdir(plot_folder)

for dataset, filterfunc, dfname in [('051218_concat',
                                     lambda f: '051218_concat' in f and 'no_infec' not in f,
                                     os.path.join(datasets_folder,
                                                  '051218_60h6sw_c1_ht5_it0_V2_csv_ddos_portscan_concat.csv')),

                                    ('051218_no_infec_concat',
                                     lambda f: '051218_no_infec_concat' in f,
                                     os.path.join(datasets_folder,
                                                  '051218_60h6sw_c1_ht5_it0_V2_csv_concat.csv')),

                                    ('171218_concat',
                                     lambda f: '171218_concat' in f,
                                     os.path.join(datasets_folder,
                                                  '171218_60h6sw_c1_ht5_it0_V2_csv_portscan_ddos_concat.csv'))]:
    df = pd.read_csv(dfname)

    changes = []
    current_c = 0
    for i, c in enumerate(df['class'].values):
        if c != current_c:
            current_c = c
            changes.append((c, i))

    if '051218' in dataset:
        colors = ['blue', 'magenta', 'orange', 'green']
    else:
        colors = ['blue', 'orange', 'magenta', 'green']
    for f in tqdm(list(filter(filterfunc, dump_files))):
        data = pickle.load(open(os.path.join(dump_folder, f), 'rb'))
        plt.figure(figsize=(8, 6))
        starts = {}
        clusters_weights = defaultdict(list)
        for instance, partial in data.items():
            for c in partial['c_clusters']:
                if c['id'] not in clusters_weights:
                    starts[c['id']] = instance
                clusters_weights[c['id']].append(math.log(c['weight']))

            for c in partial['p_clusters']:
                if c['id'] not in clusters_weights:
                    starts[c['id']] = instance
                clusters_weights[c['id']].append(math.log(c['weight']))

        for (id_, weights), color in zip(clusters_weights.items(), colors):
            start = starts[id_]
            x = [start]
            for i in range(len(weights) - 1):
                x.append(50 + x[-1])
            plt.plot(x, weights, c=color, label=f'Cluster id {id_}', linewidth=1.8)

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

            change = changes[2][1]
            if '051218' in dataset:
                label = 'PortScan'
                color = 'orange'
            else:
                label = 'DDoS'
                color = 'magenta'
            plt.plot([change, change], [0, 17], c=color, linestyle='--', label=label, linewidth=1.8)

            change = changes[1][1]
            plt.plot([change, change], [0, 17], c='black', linestyle=':', label='End of infection', linewidth=1.8)

            change = changes[3][1]
            plt.plot([change, change], [0, 17], c='black', linestyle=':', label=None, linewidth=1.8)

            change = changes[4][1]
            if '051218' in dataset:
                label = 'DDoS'
                color = 'magenta'
            else:
                label = 'PortScan'
                color = 'orange'
            plt.plot([change, change], [0, 17], c=color, linestyle='--', label=None, linewidth=1.8)

            change = changes[5][1]
            plt.plot([change, change], [0, 17], c='black', linestyle=':', label=None, linewidth=1.8)

            change = changes[6][1]
            if '051218' in dataset:
                label = 'PortScan'
                color = 'orange'
            else:
                label = 'DDoS'
                color = 'magenta'
            plt.plot([change, change], [0, 17], c=color, linestyle='--', label=None, linewidth=1.8)

            change = changes[7][1]
            plt.plot([change, change], [0, 17], c='black', linestyle=':', label=None, linewidth=1.8)

        # plt.legend(loc='upper center', fontsize=8, ncol=6, bbox_to_anchor=(0, 0.15, 1, 1), fancybox=True)
        plt.legend(loc='center left', fontsize=12, ncol=1, bbox_to_anchor=(0, 0.3, -0.5, 1), fancybox=True)
        plt.ylim(0, 18)
        plt.xticks(fontsize=18, rotation=45)
        plt.yticks(fontsize=18)
        plt.xlabel('Number of instances', fontsize=23)
        plt.ylabel('Weight of micro clusters (log)', fontsize=23)
        name = f[:-4].replace('.', '_') + '_weights' + '.pdf'
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, name))
        plt.close()

import os
import pickle
from contextlib import suppress
from itertools import combinations

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from tqdm import tqdm

dump_folder = 'dump_logs'
plots_folder = 'plots_clusters'

with suppress(Exception):
    os.mkdir(plots_folder)

def plot(data, fname, i, features):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for ax, (f1, f2) in zip(axes, combinations(range(4), 2)):
        f1_name = features[f1]
        f2_name = features[f2]
        for c in data['c_clusters']:
            ax.add_patch(patches.Circle(c['centroid'][[f1, f2]],
                                        c['radius'] + 0.001, fill=False,
                                        color='black', ls='--'))
            ax.annotate('{}'.format(c['id']), xy=c['centroid'][[f1, f2]] + 0.01,
                        color='black', fontsize=7)
        for c in data['p_clusters']:
            ax.add_patch(patches.Circle(c['centroid'][[f1, f2]],
                                        c['radius'] + 0.001, fill=False,
                                        color='blue', ls='--'))
            ax.annotate('{}'.format(c['id']), xy=c['centroid'][[f1, f2]] + 0.01,
                        color='black', fontsize=7)
        for c in data['o_clusters']:
            ax.add_patch(patches.Circle(c['centroid'][[f1, f2]],
                                        c['radius'] + 0.001, fill=False,
                                        color='red', ls='--'))
            ax.annotate('{}'.format(c['id']), xy=c['centroid'][[f1, f2]] + 0.01,
                        color='black', fontsize=7)

        ax.set_xlabel(f1_name)
        ax.set_ylabel(f2_name)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    colors = ['black', 'blue', 'red']
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='--') for c in colors]
    labels = ['c_cluster', 'p_cluster', 'o_cluster']
    leg = fig.legend(lines, labels, ncol=3, loc='lower center', bbox_to_anchor=(0, 0, 1, 1))
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    name = fname.replace('.pkl', '_{}_.pdf'.format(i))
    plt.savefig((os.path.join(plots_folder, name)))
    plt.close()


datasets_folder = 'datasets'

features = ('H_src_ip', 'H_dst_ip', 'H_src_port', 'H_dst_port')

for dataset, f, dfname in [('051218',
                            '051218_lambda=0.06807737612366145_beta=0.3004826601733964_ep=0.05_mu=250_speed=1000.pkl',
                            os.path.join(datasets_folder,
                                         '051218_60h6sw_c1_ht5_it0_V2_csv_ddos_portscan.csv')),

                           ('051218_no_infec',
                            '051218_no_infec_lambda=0.06807737612366145_beta=0.3004826601733964_ep=0.05_mu=250_speed=1000.pkl',
                            os.path.join(datasets_folder,
                                         '051218_60h6sw_c1_ht5_it0_V2_csv.csv')),

                           ('171218',
                            '171218_lambda=0.06807737612366145_beta=0.3004826601733964_ep=0.05_mu=250_speed=1000.pkl',
                            os.path.join(datasets_folder,
                                         '171218_60h6sw_c1_ht5_it0_V2_csv_portscan_ddos.csv'))]:
    df = pd.read_csv(dfname)

    changes = []
    current_c = 0
    for i, c in enumerate(df['class'].values):
        if c != current_c:
            current_c = c
            changes.append((c, i))

    fulldata = pickle.load(open(os.path.join(dump_folder, f), 'rb'))
    n_p_clusters = fulldata[50]['n_p_clusters']
    for i in tqdm(list(range(100, list(fulldata.keys())[-1], 50))):
        data = fulldata[i]
        if n_p_clusters != data['n_p_clusters']:
            n_p_clusters = data['n_p_clusters']

            plot(fulldata[i - 50], f, i - 50, features)

            plot(fulldata[i], f, i, features)

#%%
import os, sys
import numpy as np
from glob import glob
import pandas as pd
sys.path.append('.')
from fit_predictor import load_dataset

labels_file = 'data/20252_wbu_inst2_idps_labels.csv'
res = 64
voltag = '3d'
dset = 'ukbb20252'
nboots = 10
score_files = dict(ukbb20252=dict())
score_files['ukbb20252']['raptor'] = sys.argv[1]
#%%
labels, _ = load_dataset(labels_file, None)
cols = labels.drop('split', axis=1).columns
meta = pd.read_csv('data/brain_biomarkers.metadata.csv').set_index('feature')

descs = [(c, meta.loc[int(c.split('-')[0])].item().lower(), ci) for ci, c in enumerate(cols)]
groups = [
    ('white matter', [
        '24485-2.0',
        '24486-2.0',
        '25007-2.0',
        '25008-2.0',
    ]),
    ('grey matter', [
        '25001-2.0',
        '25002-2.0',
        '25003-2.0',
        '25004-2.0',
        '25005-2.0',
        '25006-2.0',
    ]),
    'cerebellum', 'amygdala', 'hippocampus', 'cortex', 'gyrus', 'pallidum', 'caudate', 'thalamus',
]
col_by_group = {}
for g in groups:
    if type(g) == tuple:
        col_by_group[g[0]] = [d for d in descs if d[0] in g[1]]
        # print(g[0], len(col_by_group[g[0]]))
    else:
        col_by_group[g] = [d for d in descs if g in d[1]]
        # print(g, len(col_by_group[g]))
#%%
scores = dict()
test_ids = labels[labels['split'] == 'test'].index.values

feature_cols = [c for c in labels.columns if c not in ['split']]
cls = labels.loc[test_ids][feature_cols].values
cls.shape
#%%
if not os.path.exists('data/boot_20252_wbu_inst2_idps_labels.npy'):
    boot_ixs = [np.random.choice(len(cls), size=len(cls), replace=True) for _ in range(nboots)]
    np.save('data/boot_20252_wbu_inst2_idps_labels.npy', boot_ixs)
else:
    boot_ixs = np.load('data/boot_20252_wbu_inst2_idps_labels.npy')

for mdl in score_files[dset]:
    load_df = score_files[dset][mdl]
    pdf = pd.read_csv(load_df).values[:, 1:]

    for g, cols in col_by_group.items():
        if g not in scores: scores[g] = dict()

        col_ixs = [j for _, _, j in cols]

        sls = []
        for ci in col_ixs:
            reps = []
            for bixs in boot_ixs:
                reps += [np.corrcoef(cls[:, ci][bixs], pdf[:, ci][bixs].astype(float))[0, 1]**2]
            sls += [reps]
        smat = np.array(sls)

        avgstat = np.mean(smat, 0)
        est = np.mean(avgstat)
        ci95 = 1.95*np.std(avgstat)
        scores[g][mdl] = [est, ci95]
# %%
colnames = list(col_by_group.keys())
for dgroup in [colnames]:
    for mdl in score_files[dset]:
        line = [scores[dset][mdl][0] for dset in dgroup]
        print(f'{mdl} & ' + ' & '.join(['%.3f' % (s) for s in line]) + ' & %.3f \\\\' % np.mean(line))
    print()

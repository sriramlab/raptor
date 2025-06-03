
import argparse
import json
import pandas as pd
import numpy as np
import pickle as pk
import torch
import copy
import torch.nn as nn
import torch.optim as optim
import os, sys
from nns import MLP
from sklearn.multiclass import OneVsRestClassifier
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet, Lasso
from torch.utils.data import Dataset, DataLoader
import torchvision
from nns import MLP

class MM3DEmbDataset(Dataset):
    def __init__(self, split, all_labels, embs_dir, split_ids):

        self.embs_dir = embs_dir
        self.split = split
        self.split_ids = split_ids
        feature_cols = [c for c in all_labels.columns if c not in ['split']]
        self.feature_names = feature_cols
        self.labels = all_labels.loc[split_ids][self.feature_names].values

    def __len__(self):
        return len(self.split_ids)

    def __getitem__(self, index):

        pid = self.split_ids[index]
        pid = pid.split('.')[0] # strip any misc extensions that may be in ids

        evec = np.load(f'{self.embs_dir}/{pid}.npy')
        evec /= 256 # NOTE: assumes embeddings haven't been rescaled/standardized

        return evec, self.labels[index]

def load_dataset(labels_file, embeddings_folder=None):
    labels = pd.read_csv(labels_file)
    labels = labels.set_index(labels.columns[0])

    train_stats = labels.loc[labels['split'] == 'train'].describe()
    train_mean = train_stats.loc['mean']
    train_std = train_stats.loc['std']

    for split in ['train', 'val', 'test']:
        for col in labels.columns:
            if col != 'split':
                labels.loc[labels['split'] == split, col] -= train_mean[col]
                labels.loc[labels['split'] == split, col] /= train_std[col]

    phase_ids = dict()
    for ph in ['train', 'val', 'test']:
        phase_ids[ph] = [i for i, split in zip(labels.index, labels['split']) if ph == split]
        print(f'{ph}: {len(phase_ids[ph])}')


    if embeddings_folder is None:
        return labels, None

    datasets = { ph: MM3DEmbDataset(ph, labels, embeddings_folder, split_ids=phase_ids[ph]) for ph in ['train', 'val', 'test'] }

    return labels, datasets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train embedding model')
    parser.add_argument('--embeddings', required=True, type=str)
    parser.add_argument('--labels', required=True, type=str)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--predict', default=False, action='store_true')
    parser.add_argument('--regression', default=False, action='store_true', help='Whether to use regression or classification. Will be set automatically if labels appear continuous.')
    parser.add_argument('--layers', default=3, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    args = parser.parse_args()

    labels, datasets = load_dataset(args.labels, args.embeddings)

    device = torch.device(args.device)
    model = MLP(len(datasets['train'][0][0]), 256, len(datasets['train'][0][1]), args.layers, args.dropout).to(device)
    print(model)

    if args.regression:
        criterion = nn.MSELoss()
    elif args.multilabel:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    savefolder = args.labels.split('/')[-1].replace('/', '__').split('.')[0]
    savename = '-'.join(args.embeddings.split('/')[-2:])
    ckpt_file = f'checkpoints/{savefolder}/{savename}.pth'
    pred_file = f'checkpoints/{savefolder}/predictions_test_{savename}.csv'
    os.makedirs(f'checkpoints/{savefolder}', exist_ok=True)
    print(ckpt_file)
    if args.predict:
        model.load_state_dict(torch.load(ckpt_file, weights_only=True))

    hist = []
    best_val_loss = float('inf')
    best_model_state = None
    loaders = { ph: DataLoader(d, batch_size=args.batch_size, shuffle=ph=='train') for ph, d in datasets.items() }
    for epoch in range(1 if args.predict else args.epochs):
        for ph in ['test'] if args.predict else ['train', 'val', 'test']:
            loader = loaders[ph]

            model.train() if ph == 'train' else model.eval()

            pbar = tqdm(loader)
            losshist = [0, 0]
            preds = []
            optimizer.zero_grad()
            for i, data in enumerate(pbar):
                inputs = data[0].to(device).float()
                labels = data[1].to(device).float()

                with torch.set_grad_enabled(ph == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if ph == 'train':
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    losshist[0] += loss.item()*len(inputs)
                    losshist[1] += len(inputs)

                if ph == 'test':
                    if args.regression:
                        pass
                    elif args.multilabel:
                        outputs = torch.sigmoid(outputs)
                    else:
                        outputs = torch.softmax(outputs, dim=-1)
                    outputs = outputs.cpu().numpy()
                    preds += [(outputs, labels.cpu().numpy())]

                pbar.set_postfix(dict(e=epoch, p=ph, ls='%.4f'%(losshist[0]/losshist[1])))

            if not args.predict:
                if ph == 'val' and losshist[0]/losshist[1] < best_val_loss:
                    best_val_loss = losshist[0]/losshist[1]
                    torch.save(model.state_dict(), ckpt_file)

    if not args.predict: print('Finished Training')

    preds, labs = [np.concatenate(t) for t in zip(*preds)]

    num_classes = preds.shape[1]
    for colvals, saveto in zip([preds], [pred_file]):
        dfdict = dict(ids=datasets['test'].split_ids)
        for ci, (cname, col) in enumerate(zip(datasets['test'].feature_names, colvals.T)):
            dfdict[cname] = col
        pd.DataFrame(dfdict).set_index('ids').to_csv(saveto)

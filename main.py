import os
import h5py
import torch
import opt
import numpy as np
import train
import scipy.io as sio

from utils import preprocess

Tensor = torch.cuda.FloatTensor

if __name__ == "__main__":
    h5_datasets = ['10X_PBMC', 'PBMC_68k', 'Young']
    mat_datasets = ['macosko', 'adam', 'shekhar']
    path = os.getcwd()
    dataset = opt.args.name
    gene_exp = []
    real_label = []
    if dataset in h5_datasets:
        data_h5 = h5py.File(f"{path}/../data/{opt.args.name}.h5", 'r')
        gene_exp = np.array(data_h5.get('X'))
        real_label = np.array(data_h5.get('Y')).reshape(-1)
        gene_exp = preprocess(gene_exp, opt.args.select_gene)
    elif dataset in mat_datasets:
        data_mat = sio.loadmat(f"{path}/../data/{opt.args.name}.mat")
        gene_exp = np.array(data_mat['feature'])
        real_label = np.array(data_mat['label']).reshape(-1)
        gene_exp = preprocess(gene_exp, opt.args.select_gene)

    print(f"The gene expression matrix shape is {gene_exp.shape}...")
    cluster_number = np.unique(real_label).shape[0]
    print(f"The real clustering num is {cluster_number}...")

    results = train.run(gene_exp=gene_exp, cluster_number=cluster_number, dataset=opt.args.name,
                        real_label=real_label, epochs=opt.args.epoch, lr=opt.args.lr,
                        temperature=opt.args.temperature, dropout=opt.args.dropout,
                        layers=[opt.args.enc_1, opt.args.enc_2, opt.args.enc_3, opt.args.mlp_dim],
                        save_pred=True, cluster_methods=opt.args.cluster_methods, batch_size=opt.args.batch_size,
                        m=opt.args.m, noise=opt.args.noise)

    print("ARI:    " + str(results["ari"]))
    print("NMI:    " + str(results["nmi"]))
    print("Time:   " + str(results['time']))

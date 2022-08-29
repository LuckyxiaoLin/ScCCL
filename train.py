import os
import mlp
import network
import opt
import math
import time
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, calinski_harabasz_score
import models
import data_aug
import st_loss
from save_model import save_model


def adjust_learning_rate(optimizer, epoch, lr):
    p = {
        'epochs': 500,
        'optimizer': 'sgd',
        'optimizer_kwargs':
            {'nesterov': False,
             'weight_decay': 0.0001,
             'momentum': 0.9,
             },
        'scheduler': 'cosine',
        'scheduler_kwargs': {'lr_decay_rate': 0.1},
    }

    new_lr = None

    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        new_lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2

    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            new_lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        new_lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    return lr


def get_device(use_cpu):
    if use_cpu is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    return device


def run(gene_exp, cluster_number, dataset, real_label, epochs,
        lr, temperature, dropout, layers, batch_size, m,
        save_pred=True, noise=None, use_cpu=None,
        cluster_methods=None):
    if cluster_methods is None:
        cluster_methods = []
    results = {}

    start = time.time()
    embedding, max_epoch = train_model(gene_exp=gene_exp, cluster_number=cluster_number, real_label=real_label,
                                       epochs=epochs, lr=lr, temperature=temperature,
                                       dropout=dropout, layers=layers, batch_size=batch_size,
                                       m=m, save_pred=save_pred, noise=noise, use_cpu=use_cpu)

    if save_pred:
        results[f"features"] = embedding
        results[f"max_epoch"] = max_epoch
    elapsed = time.time() - start
    res_eval = cluster_embedding(embedding, cluster_number, real_label, save_pred=save_pred,
                                 cluster_methods=cluster_methods)
    results = {**results, **res_eval, "dataset": dataset, "time": elapsed}

    pre_path = os.path.join(os.getcwd(), "save", opt.args.name, "checkpoint_{}.tar".format(max_epoch))
    os.remove(pre_path)

    return results


def train_model(gene_exp, cluster_number, real_label, epochs, lr,
                temperature, dropout, layers, batch_size, m,
                save_pred=False, noise=None, use_cpu=None, evaluate_training=True):
    device = get_device(use_cpu)

    dims = np.concatenate([[gene_exp.shape[1]], layers])
    data_aug_model = data_aug.DataAug(dropout=dropout)
    encoder_q = models.BaseEncoder(dims)
    encoder_k = models.BaseEncoder(dims)
    instance_projector = mlp.MLP(layers[2], layers[2]+layers[3], layers[2]+layers[3])
    cluster_projector = mlp.MLP(layers[2], layers[3], cluster_number)
    model = network.Network(encoder_q, encoder_k, instance_projector, cluster_projector, cluster_number,
                            m=m)
    data_aug_model.to(device)
    model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        model.parameters()),
                                 lr=lr)

    criterion_instance = st_loss.InstanceLoss(temperature=temperature)
    criterion_cluster = st_loss.ClusterLoss(cluster_number, temperature=temperature)

    max_value = -1
    max_epoch = -1

    idx = np.arange(len(gene_exp))
    for epoch in range(epochs):
        model.train()
        adjust_learning_rate(optimizer, epoch, lr)
        np.random.shuffle(idx)
        loss_instance_ = 0
        loss_cluster_ = 0
        for pre_index in range(len(gene_exp) // batch_size + 1):
            c_idx = np.arange(pre_index * batch_size,
                              min(len(gene_exp), (pre_index + 1) * batch_size))
            if len(c_idx) == 0:
                continue
            c_idx = idx[c_idx]
            c_inp = gene_exp[c_idx]
            input1 = data_aug_model(torch.FloatTensor(c_inp))
            input2 = data_aug_model(torch.FloatTensor(c_inp))

            if noise is None or noise == 0:
                input1 = torch.FloatTensor(input1).to(device)
                input2 = torch.FloatTensor(input2).to(device)
            else:
                noise_vec = torch.FloatTensor(np.random.normal(loc=0, scale=noise, size=input1.shape))
                input1 = torch.FloatTensor(input1 + noise_vec).to(device)
                noise_vec = torch.FloatTensor(np.random.normal(loc=0, scale=noise, size=input2.shape))
                input2 = torch.FloatTensor(input2 + noise_vec).to(device)
            q_instance, q_cluster, k_instance, k_cluster = model(input1, input2)

            features_instance = torch.cat(
                [q_instance.unsqueeze(1),
                 k_instance.unsqueeze(1)],
                dim=1)
            features_cluster = torch.cat(
                [q_cluster.t().unsqueeze(1),
                 k_cluster.t().unsqueeze(1)],
                dim=1)
            loss_instance = criterion_instance(features_instance)
            loss_cluster = criterion_cluster(features_cluster)
            loss = loss_instance + loss_cluster
            loss_instance_ += loss_instance.item()
            loss_cluster_ += loss_cluster.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if evaluate_training and real_label is not None:
            model.eval()
            with torch.no_grad():
                q_instance, _, _, _ = model(torch.FloatTensor(gene_exp).to(device), None)
                features = q_instance.detach().cpu().numpy()
            res = cluster_embedding(features, cluster_number, real_label, save_pred=save_pred)
            print(
                f"Epoch {epoch}: Loss_instance: {loss_instance_}, Loss_cluster: {loss_cluster_}, ARI: {res['ari']}, "
                f"NMI: {res['nmi']} "
            )

            if res['ari'] + res['nmi'] >= max_value:
                max_value = res['ari'] + res['nmi']
                save_model(opt.args.name, model, optimizer, epoch, max_epoch)
                max_epoch = epoch

    model.eval()
    model_fp = os.getcwd() + '/save/' + opt.args.name + "/checkpoint_{}.tar".format(max_epoch)
    model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
    model.to(device)
    with torch.no_grad():
        q_instance, _, _, _ = model(torch.FloatTensor(gene_exp).to(device), None)
        features = q_instance.detach().cpu().numpy()

    return features, max_epoch


def cluster_embedding(embedding, cluster_number, real_label, save_pred=False, cluster_methods=None):
    if cluster_methods is None:
        cluster_methods = ["KMeans"]
    result = {"t_clust": time.time()}
    if "KMeans" in cluster_methods:
        kmeans = KMeans(n_clusters=cluster_number,
                        init="k-means++",
                        random_state=0)
        pred = kmeans.fit_predict(embedding)
        if real_label is not None:
            result[f"ari"] = round(adjusted_rand_score(real_label, pred), 4)
            result[f"nmi"] = round(normalized_mutual_info_score(real_label, pred), 4)
        # result[f"sil"] = round(silhouette_score(embedding, pred), 4)
        # result[f"cal"] = round(calinski_harabasz_score(embedding, pred), 4)
        result["t_k"] = time.time()
        if save_pred:
            result[f"pred"] = pred

    return result
import torch
import numpy as np

from model.base_gnns import train_model
from model.base_nns import MLP
from losses.loss import DeviationLoss, compute_beta, anomaly_mixup, consistency_loss, anomaly_mixup_v2, anomaly_mixup_v3
from torch_geometric.loader import NeighborSampler
from utils import get_optimiser, NodeFeatureAugmentor, tsne_vis_binary
from collections import Counter
from copy import deepcopy
from sklearn.metrics import roc_auc_score, average_precision_score

def train(split_info, labels, graph, args, anomaly_info, logger, ppr_matrix=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_cutoff = torch.tensor(args.p_cutoff).to(device)
    idx_train = split_info['idx_train']
    idx_val = split_info['idx_val']
    idx_train_anomaly = torch.tensor(split_info['idx_anomaly_train'])
    unlabeled_idx = torch.tensor(split_info['idx_test']['all']).to(device) # all the unlabeled nodes

    # create the encoder and classifier
    if args.loss == 'bce':
        print("using binary cross entropy loss...")
        clf_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    elif args.loss == 'dev':
        print("using deviation loss...")
        clf_criterion = DeviationLoss()
    else:
        raise NotImplementedError("Loss is not supported!")

    model = train_model(args)

    model.to(device)
    clf_criterion.to(device)

    # create the optimisers
    optimizer = get_optimiser(args.optimiser, model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Dataloaders
    print("building the dataloader ...")
    dataloader = NeighborSampler(graph.edge_index,
                                      node_idx=idx_train,
                                      sizes=args.sampling_sizes,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      drop_last=False,
                                      pin_memory=True)
    unlabeled_dataloader = NeighborSampler(graph.edge_index,
                                      node_idx=unlabeled_idx,
                                      sizes=args.sampling_sizes,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=0,
                                      pin_memory=True)
    weak_augmentor = NodeFeatureAugmentor({
                        'noise': {'sigma': 0.02},
                        'mask': {'mask_prob': 0.1}})
    strong_augmentor = NodeFeatureAugmentor({
                        'noise': {'sigma': 0.02},
                        'mask': {'mask_prob': 0.1},
                        'mixup': {'alpha': 0.1},
                        'scaling': {'gamma': 0.1}})


    
    # create the labels
    rest_labels = np.zeros(shape=labels.shape[0])
    for i in anomaly_info['all_anomaly']:
        rest_labels[np.where(labels == i)[0]] = 1
    rest_labels = torch.from_numpy(rest_labels).long().to(device)
    classwise_acc = torch.zeros((args.num_classes,)).to(device) # just judge the normal and anomaly class

    selected_label = torch.ones((graph.num_nodes), dtype=torch.long) * -1
    selected_label = selected_label.to(device)

    auroc_test_all_best = 0.0
    aupr_test_all_best = 0.0
    auroc_test_unknown_best = 0.0
    aupr_test_unknown_best = 0.0
    embedds_best = None
    max_counter = {i: 0 for i in [0, 1]}
    # rest_labels_copy = deepcopy(rest_labels)
    # idx_train_copy = deepcopy(idx_train)
    # graph_copy = deepcopy(graph)
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.num_epochs):
        model.train()
        logits_labeled = None

        ###### train the model ######
        for i, (batch_size_labeled, n_id_labeled, adjs_labeled) in enumerate(dataloader):
            # labeled nodes forward pass
            x_labeled = graph.x[n_id_labeled].to(device)
            adjs_labeled = [adj.to(device) for adj in adjs_labeled]
            if i == 0:
                logits_labeled = model(x_labeled, adjs_labeled)
            else:
                logits_labeled = torch.cat((logits_labeled, model(x_labeled, adjs_labeled)), dim=0)
        labeled_loss = clf_criterion(logits_labeled, rest_labels[idx_train].unsqueeze(1).float())
        loss = labeled_loss

        # mixup for the anomaly samples
        mixup_loss = 0.0
        if args.mixup:
            mixup_loss = anomaly_mixup_v3(args, model, graph, idx_train_anomaly, ppr_matrix)
            # embeds = embeds.detach()
            # updata_graph, new_nodes = mixup_by_similarity(embeds, graph_copy, idx_train_anomaly)
            # new_labels = torch.ones(embeds.size(0), dtype=rest_labels.dtype, device=device)
            # rest_labels = torch.cat((rest_labels_copy, new_labels), dim=0)
            # idx_train = torch.cat((idx_train_copy, new_nodes), dim=0)
            # graph = updata_graph
            loss = labeled_loss + args.mixup_loss * mixup_loss

        unnlabeled_loss = 0.0
        if args.used_unlabeled_data:

            pseudo_counter = Counter(selected_label.tolist())
            print(f"pseudo_counter: {pseudo_counter}")
            if max(pseudo_counter.values()) < graph.num_nodes:
                if args.thresh_warmup:
                    for i in range(args.num_classes):
                        # 以下实现区别于Flexmatch，首先无论某一类中的数据量是多少，都可以从0到1，另外，当数据量比较大的时候，得到的classwise_acc就是1
                        # 相当于提高阈值，增加样本可信度，但是当数据量减少了，降低阈值，保持样本接受率
                        max_counter[i] = max(max_counter[i], pseudo_counter[i])
                        if max_counter[i] == 0:
                            classwise_acc[i] = 0.0
                        else:
                            classwise_acc[i] = pseudo_counter[i] / max_counter[i]
                else:
                    wo_negative_one = deepcopy(pseudo_counter)
                    if -1 in wo_negative_one.keys():
                        wo_negative_one.pop(-1)
                    for i in range(args.num_classes):
                        max_counter[i] = max(max_counter[i], wo_negative_one[i])
                        if max_counter[i] == 0:
                            classwise_acc[i] = 0.0
                        else:
                            classwise_acc[i] = wo_negative_one[i] / max_counter[i]
            logits_unlabeled_weak, logits_unlabeled_strong, logits_unlabeled = [], [], []
            for i, (batch_size_unlabeled, n_id_unlabeled, adjs_unlabeled) in enumerate(unlabeled_dataloader):
                # unlabeled nodes forward pass
                if args.used_augment_for_anormaly:
                    x_unlabeled_weak = weak_augmentor.augment(graph.x[n_id_unlabeled].to(device))
                    x_unlabeled_strong = strong_augmentor.augment(graph.x[n_id_unlabeled].to(device))
                    adjs_unlabeled = [adj.to(device) for adj in adjs_unlabeled]
                    logits_w = model(x_unlabeled_weak, adjs_unlabeled)
                    logits_s = model(x_unlabeled_strong, adjs_unlabeled)
                    logits_unlabeled_weak.append(logits_w)
                    logits_unlabeled_strong.append(logits_s)
                else:
                    x_unlabeled = graph.x[n_id_unlabeled].to(device)
                    adjs_unlabeled = [adj.to(device) for adj in adjs_unlabeled]
                    logits_unlabeled.append(model(x_unlabeled, adjs_unlabeled))

            if args.used_augment_for_anormaly:
                logits_unlabeled_weak = torch.cat(logits_unlabeled_weak, dim=0)
                logits_unlabeled_strong = torch.cat(logits_unlabeled_strong, dim=0)
                logits = tuple([logits_unlabeled_weak, logits_unlabeled_strong])
            else:
                logits_unlabeled = torch.cat(logits_unlabeled, dim=0)
                logits = tuple([logits_unlabeled])
            if epoch > 5: # warm up the threshold
                unnlabeled_loss, mask, select, pseudo_labels = consistency_loss(logits, classwise_acc, p_cutoff=p_cutoff)

                if unlabeled_idx[select == 1].nelement() != 0: # select中选择的有正常跟异常的节点
                    selected_label[unlabeled_idx[select == 1]] = pseudo_labels[select == 1]
            loss = labeled_loss + args.unlabeled_loss * unnlabeled_loss + args.mixup_loss * mixup_loss


        # computer energy loss
        energy_loss = 0.0
        if args.energy_loss:
            # beta = compute_beta(model, idx_train_copy, idx_val, graph_copy, rest_labels_copy, dataloader, device, args)
            beta = compute_beta(model, idx_train, idx_val, graph, rest_labels, dataloader, device, args)
            energy = model.energy(logits_labeled)
            energy_loss = torch.mean(beta * energy)
            loss = labeled_loss + args.energy_loss_weight * energy_loss + args.unlabeled_loss * unnlabeled_loss + args.mixup_loss * mixup_loss


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()

        if epoch % 1 == 0:

            auroc_test_all_best, aupr_test_all_best, auroc_test_unknown_best, aupr_test_unknown_best, embedds_best = (
                eval(model, split_info, rest_labels, graph, args, device, auroc_test_all_best, aupr_test_all_best, auroc_test_unknown_best, aupr_test_unknown_best, embedds_best))
            print(f"Epoch: {epoch}, Best AUROC test all: {auroc_test_all_best}, AUPR test all: {aupr_test_all_best}, AUROC test unknown: {auroc_test_unknown_best}, AUPR test unknown: {aupr_test_unknown_best}")
            # print(f"Epoch: {epoch}, loss: {loss}, labeled_loss: {labeled_loss}, unnlabeled_loss: {unnlabeled_loss}, mixup_loss: {mixup_loss}, energy_loss: {energy_loss}")
            logger.info(f"Epoch: {epoch}, loss: {loss}, labeled_loss: {labeled_loss}, unnlabeled_loss: {unnlabeled_loss}, mixup_loss: {mixup_loss}, energy_loss: {energy_loss}")
            logger.info(f"Best AUROC test all: {auroc_test_all_best}, AUPR test all: {aupr_test_all_best}, AUROC test unknown: {auroc_test_unknown_best}, AUPR test unknown: {aupr_test_unknown_best}")

    # visiualize the embedding
    idx_test_all = split_info['idx_test']['all']
    y_true = rest_labels.cpu().numpy()[idx_test_all]
    embeds_best = embedds_best.cpu().numpy()[idx_test_all]
    tsne_vis_binary(embeds_best, y_true, args.dataname)


def eval(model, split_info, labels, graph, args, device, auroc_test_all_best, aupr_test_all_best, auroc_test_unknown_best, aupr_test_unknown_best, embedds_best):
    model.eval()
    y_true = labels.cpu().numpy()
    idx_train = split_info['idx_train']
    idx_test_all = split_info['idx_test']['all'] # normal_test + known_abnormal_test + unknown_abnormal_test
    idx_test_unknown = split_info['idx_test']['unknown'] # normal_test + unknown_abnormal_test
    idx_test_known = split_info['idx_test']['known'] # normal_test + known_abnormal_test
    
    test_dataloader = NeighborSampler(graph.edge_index,
                                      node_idx=None,
                                      sizes=args.sampling_sizes,
                                      batch_size=args.batch_size * 4, #args.batch_size * 4
                                      shuffle=False,
                                      drop_last=False,
                                      pin_memory=True)

    with torch.no_grad():
        logits_test = None
        embeds = None
        for i, (batch_size_test, n_id_test, adjs_test) in enumerate(test_dataloader):
            x_test = graph.x[n_id_test].to(device)
            adjs_test = [adj.to(device) for adj in adjs_test]
            if i == 0:
                results = model(x_test, adjs_test, return_ebds=True)
                logits_test = results[0]
                embeds = results[1]
            else:
                results = model(x_test, adjs_test, return_ebds=True)
                logits_test = torch.cat((logits_test, results[0]), dim=0)
                embeds = torch.cat((embeds, results[1]), dim=0)
        y_pred = torch.sigmoid(logits_test).squeeze(1).cpu().numpy()

    ################# compute the metrics #####################
    # all nodes
    auroc_all, aupr_all = roc_auc_score(y_true, y_pred), average_precision_score(y_true, y_pred)
    auroc_train, aupr_train = roc_auc_score(y_true[idx_train], y_pred[idx_train]), average_precision_score(y_true[idx_train], y_pred[idx_train])
    auroc_test_all, aupr_test_all = roc_auc_score(y_true[idx_test_all], y_pred[idx_test_all]), average_precision_score(y_true[idx_test_all], y_pred[idx_test_all])
    auroc_test_unknown, aupr_test_unknown = roc_auc_score(y_true[idx_test_unknown], y_pred[idx_test_unknown]), average_precision_score(y_true[idx_test_unknown], y_pred[idx_test_unknown])
    auroc_test_known, aupr_test_known = roc_auc_score(y_true[idx_test_known], y_pred[idx_test_known]), average_precision_score(y_true[idx_test_known], y_pred[idx_test_known])
    # print(f"AUROC all: {auroc_all}, AUPR all: {aupr_all}, AUROC train: {auroc_train}, AUPR train: {aupr_train}, AUROC test all: {auroc_test_all}, AUPR test all: {aupr_test_all}, AUROC test unknown: {auroc_test_unknown}, AUPR test unknown: {aupr_test_unknown}, AUROC test known: {auroc_test_known}, AUPR test known: {aupr_test_known}")

    print(f"AUROC test all: {auroc_test_all}, AUPR test all: {aupr_test_all}, AUROC test unknown: {auroc_test_unknown}, AUPR test unknown: {aupr_test_unknown}")
    if auroc_test_all > auroc_test_all_best:
        auroc_test_all_best = auroc_test_all
        embedds_best = embeds.detach()
    if aupr_test_all > aupr_test_all_best:
        aupr_test_all_best = aupr_test_all
    if auroc_test_unknown > auroc_test_unknown_best:
        auroc_test_unknown_best = auroc_test_unknown
    if aupr_test_unknown > aupr_test_unknown_best:
        aupr_test_unknown_best = aupr_test_unknown
    return auroc_test_all_best, aupr_test_all_best, auroc_test_unknown_best, aupr_test_unknown_best, embedds_best






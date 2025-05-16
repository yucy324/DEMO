import argparse, logging
import torch
import numpy as np
import dgl
import yaml, time
from utils import *
from trainer import train
from dgl.data.utils import save_graphs, load_graphs
from addict import Dict


def main(unlabeled_loss, mixup_loss, energy_loss):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, default='computers',
                        choices=['photo', 'computers', 'cs', 'yelp', 'ogbn-arxiv', 'ogbn-mag', 'tfinance'])
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--train_normal_ratio", type=float, default=0.05)
    parser.add_argument("--train_anormaly_num", type=int, default=50)
    parser.add_argument("--val_normal_ratio", type=float, default=0.01)
    parser.add_argument("--val_anormaly_num", type=int, default=30)

    parser.add_argument("--used_unlabeled_data", type=bool, default=True)
    parser.add_argument("--energy_loss", type=bool, default=True)
    parser.add_argument("--mixup", type=bool, default=True)

    parser.add_argument("--batch_size_for_anormaly", type=int, default=20)
    parser.add_argument("--sample_size_for_anormaly", type=int, default=25)

    parser.add_argument("--thresh_warmup", type=bool, default=True)
    parser.add_argument("--used_augment_for_anormaly", type=bool, default=True)


    parser.add_argument("--div_loss_alpha", type=float, default=0.5)
    # 各类损失的权重
    parser.add_argument("--unlabeled_loss", type=float, default=0.3)
    parser.add_argument("--mixup_loss", type=float, default=0.1)
    parser.add_argument("--mixup_loss_type", type=str, default='consistency', choices=['consistency', 'contrastive'])
    parser.add_argument("--energy_loss_weight", type=float, default=0.1)

    args = parser.parse_args()
    args.unlabeled_loss = unlabeled_loss
    args.mixup_loss = mixup_loss
    args.energy_loss_weight = energy_loss
    if args.sample_size_for_anormaly > args.train_anormaly_num:
        args.sample_size_for_anormaly = args.train_anormaly_num
    set_seed(args.seed)
    dataname = args.dataname
    with open(f'data/{dataname}/{dataname}.yaml', 'r') as f:
        yaml_cfg = yaml.safe_load(f)
    args = Dict(merge_configs(vars(args), yaml_cfg))
    time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logging.basicConfig(level=logging.INFO,
                        filename='logs/{}_{}.log'.format(dataname, time_str),
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(args)
    graph, labels, dset_info = load_data(dataname)
    ppr_matrix = None
    if args.mixup:
        path = f'data/{dataname}/ppr_matrix.npy'
        if os.path.exists(path):
            print("Using mixup, loading PPR...")
            ppr_matrix = np.load(path)
        else:
            print("Using mixup, calculating PPR...")
            ppr_matrix = compute_ppr(graph, dataname)
            np.save(path, ppr_matrix)
    if dataname in ['photo', 'computers', 'cs', 'yelp', 'tfinance']:
        anomaly_class_idx = np.where((dset_info['class_per'] <= 0.05) & (dset_info['class_per'] >= 0.00))[0]
    elif dataname in ['ogbn-arxiv']:
        anomaly_class_idx = np.where((dset_info['class_per'] <= 0.05) & (dset_info['class_per'] >= 0.03))[0]
    elif dataname in ['ogbn-mag']:
        anomaly_class_idx = np.where((dset_info['class_per'] <= 0.0003) & (dset_info['class_per'] >= 0.00))[0]
    else:
        raise ValueError("Unknown dataset")
    for idx in anomaly_class_idx:
        class_name = dset_info['class_names'][idx] if dset_info['class_names'] is not None else str(idx)
        print("using class %d: %s as known anomalies" % (idx, class_name))
        print("Known anomalies: %.2f%%" % (dset_info['class_per'][idx] * 100))
        anomaly_info = {'known_anomaly': idx,
                        'unknown_anomaly': [i for i in anomaly_class_idx if i != idx],
                        'normal': [i for i in dset_info['class_idx'] if i not in anomaly_class_idx],
                        'all_anomaly': anomaly_class_idx}
        split_info = ad_split_num(labels, args, anomaly_info)
        train(split_info, labels, graph, args, anomaly_info, logger, ppr_matrix)
        break


        # 保存训练、验证和测试集的节点索引
        # 标签改成0与1，并将graph转化为dgl格式
        # np.savetxt(f"baseline_data/{args.dataname}_{args.train_anormaly_num}_train.txt", np.array(split_info['idx_train']).astype(int), fmt='%d')
        # np.savetxt(f"baseline_data/{args.dataname}_{args.train_anormaly_num}_val.txt", np.array(split_info['idx_val']).astype(int), fmt='%d')
        # np.savetxt(f"baseline_data/{args.dataname}_{args.train_anormaly_num}_test_all.txt", split_info['idx_test']['all'].astype(int), fmt='%d')
        # np.savetxt(f"baseline_data/{args.dataname}_{args.train_anormaly_num}_test_unknown.txt", split_info['idx_test']['unknown'].astype(int), fmt='%d')

        # rest_labels = np.zeros(shape=labels.shape[0])
        # for i in anomaly_info['all_anomaly']:
        #     rest_labels[np.where(labels == i)[0]] = 1
        # rest_labels = torch.from_numpy(rest_labels).long()
        # src, dst = graph.edge_index
        # g = dgl.graph((src, dst), num_nodes=graph.num_nodes)
        # g.ndata['feat'] = graph.x
        # g.ndata['label'] = rest_labels
        # save_graphs(f"baseline_data/{args.dataname}.bin", g)
        # print(f'{args.dataname}_{args.train_anormaly_num} done')





if __name__ == '__main__':
    main(0.3, 0.1, 0.1)
    # main(0.3, 0.1, 0.3)
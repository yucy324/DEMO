import datetime
import os, torch
import tools.io as io
import numpy as np
import torch.optim as optim
import dgl, random
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset
from scipy.linalg import fractional_matrix_power, inv
from dgl import graph as dgl_graph
from dgl.nn import APPNPConv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True)
    return seed

def merge_configs(cmd_args, yaml_args):
    for key, value in cmd_args.items():
        if value is not None:
            # 处理嵌套参数（如 model.hidden_dim）
            keys = key.split('.')
            temp_yaml = yaml_args
            for k in keys[:-1]:
                temp_yaml = temp_yaml.setdefault(k, {})
            temp_yaml[keys[-1]] = value
    return yaml_args

def get_optimiser(name, param, lr, weight_decay):
    if name.lower() == 'adam':
        optimiser = optim.Adam(param, lr=lr, weight_decay=weight_decay)
    elif name.lower() == 'adamw':
        optimiser = optim.AdamW(param, lr=lr, weight_decay=weight_decay)
    elif name.lower() == 'sgd':
        optimiser = optim.SGD(param, lr=lr)
    elif name.lower() == 'rmsprop':
        optimiser = optim.RMSprop(param, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError("Optimiser function not supported!")
    return optimiser

def log(message, data_name=None, level="INFO", log_dir="logs"):

    os.makedirs(log_dir, exist_ok=True)

    if not hasattr(log, "filename"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{data_name}_{timestamp}.log"
        log.filename = os.path.join(log_dir, filename)

    log_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_entry = f"[{log_time}] [{level:<7}] {message}\n"

    with open(log.filename, "a", encoding="utf-8") as f:
        f.write(log_entry)

    return log.filename

def load_data(data_name):
    if data_name in ['photo', 'computers', 'cs']:
        data = np.load(f'data/{data_name}/{data_name}.npz', allow_pickle=True)
        graph, labels, dset_info = io.npz_data_to_pyg_graph(data)
    elif data_name in ['ogbn-arxiv']:
        data = PygNodePropPredDataset(name=data_name, root='data')
        graph = data[0] # x为torch.float32类型
        labels = graph.y.long().squeeze(-1)
        class_idx, class_size = torch.unique(labels, return_counts=True)
        class_per = class_size.float() / labels.shape[0]
        dset_info = {
            'node_names': None,
            'attr_names': None,
            'class_names': None,
            'metadata': None,
            'class_idx': np.array(class_idx),
            'class_size': np.array(class_size),
            'class_per': np.array(class_per),
        }
        # counts = torch.bincount(labels, minlength=data.num_classes)
        # condition = (counts > graph.num_nodes * 0.03) & (counts < graph.num_nodes * 0.05)
        # num_classes_in_range = condition.sum().item()
    elif data_name in ['ogbn-mag']:
        data = PygNodePropPredDataset(name=data_name, root='data')
        paper_x = data[0]['x_dict']['paper'] # x为torch.float32类型 [num_paper_nodes, feat_dim]data
        paper_y = data[0]['y_dict']['paper']# [num_paper_nodes]
        edge_index = data[0]['edge_index_dict']['paper', 'cites', 'paper']# [2, num_edges]

        # 构造一个新的 PyG 同构图对象
        graph = Data(x=paper_x, y=paper_y, edge_index=edge_index)
        labels = graph.y.long().squeeze(-1)
        class_idx, class_size = torch.unique(labels, return_counts=True)
        class_per = class_size.float() / labels.shape[0]
        dset_info = {
            'node_names': None,
            'attr_names': None,
            'class_names': None,
            'metadata': None,
            'class_idx': np.array(class_idx),
            'class_size': np.array(class_size),
            'class_per': np.array(class_per),
        }
    elif data_name in ['yelp', 'tfinance']:
        graph = torch.load(f'data/{data_name}/{data_name}') #x为torch.float32类型
        labels = graph.y.long().squeeze(-1)
        class_idx, class_size = torch.unique(labels, return_counts=True)
        class_per = class_size.float() / labels.shape[0]
        dset_info = {
            'node_names': None,
            'attr_names': None,
            'class_names': None,
            'metadata': None,
            'class_idx': np.array(class_idx),
            'class_size': np.array(class_size),
            'class_per': np.array(class_per),
        }
    elif data_name in ['pubmed', 'cora', 'cityseer']:
        data_name = 'cora'
        data = Planetoid(root=f'data/{data_name}', name=data_name)
        graph = data[0]
        labels = graph.y.long().squeeze(-1)
        class_idx, class_size = torch.unique(labels, return_counts=True)
        class_per = class_size.float() / labels.shape[0]
        dset_info = {
            'node_names': None,
            'attr_names': None,
            'class_names': None,
            'metadata': None,
            'class_idx': np.array(class_idx),
            'class_size': np.array(class_size),
            'class_per': np.array(class_per),
        }
    else:
        raise ValueError(f"Unsupported dataset: {data_name}")
    return graph, labels, dset_info

def ad_split_num(labels, args, class_info):
    known_anomaly = class_info['known_anomaly']
    unknown_anomaly_classes = class_info['unknown_anomaly']
    normal_classes = class_info['normal']
    print(f'Normal classes: {normal_classes} in {args.dataname}')

    known_anomaly_idx = np.where(labels==known_anomaly)[0].flatten()
    normal_idx = select_class_idx_in_list(labels, normal_classes) # all normal nodes idx
    unknown_anomaly_idx = select_class_idx_in_list(labels, unknown_anomaly_classes)

    normal_train, normal_val, normal_test = random_split(normal_idx, args.train_normal_ratio, args.val_normal_ratio)
    known_anomaly_train, known_anomaly_val, known_anomaly_test = num_split(known_anomaly_idx, args.train_anormaly_num, args.val_anormaly_num)
    unknown_anomaly_test = unknown_anomaly_idx

    train_idx = np.hstack((normal_train, known_anomaly_train))
    val_idx = np.hstack((normal_val, known_anomaly_val))
    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    # min_size: photo: 331; computers: 291; cs: 118; ogbn-arxiv: 5000+; ogbn-mag: 200; yelp: 658; tfinance: 483


    test_idx = {
        'all': np.hstack((normal_test, known_anomaly_test, unknown_anomaly_test)),
        'known': np.hstack((normal_test, known_anomaly_test)),
        'unknown': np.hstack((normal_test, unknown_anomaly_test)),
        'normal': normal_test,
        'known_only': known_anomaly_test,
        'unknown_only': unknown_anomaly_test,
    }

    split_info = {
        'idx_train': train_idx,
        'idx_normal_train': normal_train,
        'idx_anomaly_train': known_anomaly_train,
        'idx_val': val_idx,
        'idx_test': test_idx
    }

    return split_info

def select_class_idx_in_list(labels, classes):
    node_idx = None
    for i in classes:
        cur_idx = np.where(labels == i)[0].flatten()
        node_idx = cur_idx if node_idx is None else np.hstack((node_idx, cur_idx))
    return node_idx

def random_split(idx, train_ratio, valid_ratio):
    n_train = int(idx.shape[0] * train_ratio)
    n_valid = int(idx.shape[0] * valid_ratio)
    randperm = torch.randperm(idx.shape[0])
    return idx[randperm[:n_train]], idx[randperm[n_train:n_train+n_valid]], idx[randperm[n_train+n_valid:]]


def num_split(idx, n_train, n_val):
    randperm = torch.randperm(idx.shape[0])
    return idx[randperm[:n_train]], idx[randperm[n_train:n_train+n_val]], idx[randperm[n_train+n_val:]]

def compute_ppr(graph, dataname: str, alpha=0.2, self_loop=True):
    if dataname in ['photo', 'computers']:
        graph = to_networkx(graph)
        a = nx.convert_matrix.to_numpy_array(graph)
        if self_loop:
            a = a + np.eye(a.shape[0])                                # A^ = A + I_n
        d = np.diag(np.sum(a, 1))                                     # D^ = Sigma A^_ii
        dinv = fractional_matrix_power(d, -0.5)                       # D^(-1/2)
        at = np.matmul(np.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
        ppr_matrix = alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1
    else:
        appnp = APPNPConv(20, 0.2)
        id = torch.eye(graph.num_nodes).float()
        src, dst = graph.edge_index
        g = dgl_graph((src, dst), num_nodes=graph.num_nodes)
        g.ndata['feat'] = graph.x
        ppr_matrix = appnp(g.add_self_loop(), id).numpy()
    return ppr_matrix


import torch
import torch.nn.functional as F


def mixup_by_similarity(z, graph, idx_train_anomaly):
    # Step 1: L2 normalize every sample along feature dimension
    z_norm = F.normalize(z, p=2, dim=1)  # shape: (N, D)

    # Step 2: Compute similarity matrix H(z_i, z_j) = dot product of normalized features
    sim_matrix = torch.matmul(z_norm, z_norm.T)  # shape: (N, N)

    # Step 3: Apply softmax along dim=1 to get λ_ij for each z_i
    lambda_weights = F.softmax(sim_matrix, dim=1)  # shape: (N, N)

    # Step 4: constract the mixup graph
    num_orig_nodes = graph.num_nodes
    num_new_nodes = z.shape[0]
    ori_feat = graph.x[idx_train_anomaly].to(z.device).detach()
    z_mixup = torch.matmul(lambda_weights, ori_feat)
    x_updated = torch.cat([graph.x.to(z.device), z_mixup], dim=0)

    src_nodes = idx_train_anomaly  # shape: [50]
    dst_nodes = torch.arange(num_orig_nodes, num_orig_nodes + num_new_nodes)
    edge_src = torch.cat([src_nodes, dst_nodes])
    edge_dst = torch.cat([dst_nodes, src_nodes])
    new_edge_index = torch.stack([edge_src, edge_dst], dim=0)
    edge_updated = torch.cat([graph.edge_index.to(z.device), new_edge_index.to(z.device)], dim=1)
    graph_updated = Data(x = x_updated, edge_index = edge_updated)
    return graph_updated, dst_nodes


class NodeFeatureAugmentor:
    def __init__(self, augmentation_config):
        """
        参数说明：
        augmentation_config: 增强配置字典，例如：
            {
                'noise': {'sigma': 0.05},
                'mask': {'mask_prob': 0.3},
                'mixup': {'alpha': 0.2},
                'scaling': {'gamma': 0.1}
            }
        """
        self.config = augmentation_config

    def _standardize(self, x):
        """标准化处理（自动检测是否已归一化）"""
        self.orig_mean = x.mean(dim=0, keepdim=True)
        self.orig_std = x.std(dim=0, keepdim=True) + 1e-8

        # 判断是否已标准化（均值为0标准差为1）
        if torch.allclose(self.orig_mean, torch.zeros_like(self.orig_mean), atol=1e-3) and \
                torch.allclose(self.orig_std, torch.ones_like(self.orig_std), atol=1e-2):
            return x.clone()
        return (x - self.orig_mean) / self.orig_std

    def _restore(self, x_normalized):
        """恢复原始数据尺度"""
        return x_normalized * self.orig_std + self.orig_mean

    def _gaussian_noise(self, x, sigma=0.05):
        noise = torch.randn_like(x) * sigma
        return x + noise

    def _feature_mask(self, x, mask_prob=0.3):
        mask = torch.rand_like(x) > mask_prob
        return x * mask.float()

    def _feature_mixup(self, x, alpha=0.2):
        idx = torch.randperm(x.size(0))
        lam = np.random.beta(alpha, alpha)
        return lam * x + (1 - lam) * x[idx]

    def _scaling_jitter(self, x, gamma=0.1):
        scale = torch.FloatTensor(x.size(1)).uniform_(1 - gamma, 1 + gamma).to(x.device)
        return x * scale

    def augment(self, x):
        """执行特征增强"""
        # 标准化处理
        x_norm = self._standardize(x)

        # 按配置顺序执行增强
        if 'noise' in self.config:
            x_norm = self._gaussian_noise(x_norm, **self.config['noise'])

        if 'mask' in self.config:
            x_norm = self._feature_mask(x_norm, **self.config['mask'])

        if 'mixup' in self.config:
            x_norm = self._feature_mixup(x_norm, **self.config['mixup'])

        if 'scaling' in self.config:
            x_norm = self._scaling_jitter(x_norm, **self.config['scaling'])

        # 恢复原始数据分布
        return self._restore(x_norm)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

def tsne_vis_binary(features, labels, dataset_name):
    # 转为 NumPy 格式
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # 设定最大每类样本数
    max_per_class = 2000
    selected_samples = []
    selected_labels = []

    for class_label in [0, 1]:  # 假设为二分类
        indices = np.where(labels == class_label)[0]
        sampled_indices = indices[:]
        selected_samples.append(features[sampled_indices])
        selected_labels.append(labels[sampled_indices])

    selected_samples = np.vstack(selected_samples)
    selected_labels = np.hstack(selected_labels)

    # t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_features = tsne.fit_transform(selected_samples)

    # 可视化绘图
    plt.figure(figsize=(10, 8))
    colors = ['#59D898', '#557BE4']  # 绿色 / 蓝色
    plt.scatter(
        tsne_features[:, 0],
        tsne_features[:, 1],
        c=selected_labels,
        cmap=ListedColormap(colors),
        s=8,
        alpha=0.6
    )

    # 去除坐标轴和网格
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    # 保存图像
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{dataset_name}_tsne_vis.svg", format='svg')
    plt.close()




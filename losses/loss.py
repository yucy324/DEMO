import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import NeighborSampler
from tqdm import tqdm
import time


class DeviationLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        """
        y_pred: 模型输出的异常分数（未经过sigmoid）
        y_true: 二进制标签（0:正常, 1:异常）
        """
        # 计算正常样本和异常样本的统计量
        normal_mask = (y_true == 0)
        abnormal_mask = (y_true == 1)

        # 正常样本的均值和标准差
        normal_scores = y_pred[normal_mask]
        normal_mean = normal_scores.mean()
        normal_std = normal_scores.std()

        # 异常样本的均值和标准差
        abnormal_scores = y_pred[abnormal_mask]
        abnormal_mean = abnormal_scores.mean()
        abnormal_std = abnormal_scores.std()

        # 计算损失：最大化两类分布的差异
        loss = torch.abs((abnormal_mean - normal_mean) / (normal_std + 1e-8))

        # 根据任务需求调整损失方向
        loss = 1.0 / (loss + 1e-8)  # 鼓励增大两类差异

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def consistency_loss(logits, class_acc, p_cutoff):
    if len(logits) == 2:
        logits_weak, logits_strong = logits
    else:
        logits_weak = logits[0]
    pseudo_label = torch.sigmoid(logits_weak)
    max_probs = pseudo_label.view(-1)  # 转为一维张量 (batch_size)
    # max_idx表示得到的伪标签
    max_idx = (max_probs >= 0.5).long()  # 根据概率判断类别（0或1）
    print(f'max_probs: {torch.max(max_probs)}， num of >0.5: {torch.sum(max_idx == 1)}')

    # mask就是根据不同类别的概率来选择样本，也就是说之前是固定阈值进行操作，而现在则是不同类别拥有一个动态适应的概率值来选择样本
    # 下面的操作中，max_probs和max_idx是一一对应的，而max_idx则表示样本属于哪一类，class_acc[max_idx]表示max_idx的占比，
    # 该类别被选中的样本越多，则概率越大
    # 对于abnormal的probs，我们希望越大越好，但是对于normal的probs，我们希望越小越好，所以分成两个部分进行选择mask，最终进行结合
    # 得到的mask用于损失计算，select用于选择样本
    threshold = p_cutoff[max_idx] * (class_acc[max_idx] / (2. - class_acc[max_idx]))
    ge_mask = (max_idx == 1)
    le_mask = (max_idx == 0)
    threshold[le_mask] = 2 * p_cutoff[0] - threshold[le_mask] # 对于le而言，阈值应该取反
    mask = torch.zeros_like(max_probs)
    mask[ge_mask] = max_probs[ge_mask].ge(threshold[ge_mask]).float()
    mask[le_mask] = max_probs[le_mask].le(threshold[le_mask]).float()

    # 选择max_probs符合条件的样本，这里是预测值，必须是0.5以上才算异常
    abnormal_probs = max_probs[max_probs >= 0.5]
    normal_probs = max_probs[max_probs < 0.5]
    # 如果存在正/异常样本，则计算置信度分位数，作为置信度阈值，否则默认,abnormal_probs.abnormal_probs.quantile从小到大排序
    abnormal_cutoff = abnormal_probs.quantile(0.95) if len(abnormal_probs) > 0 else p_cutoff[0]
    normal_cutoff = normal_probs.quantile(0.05) if len(normal_probs) > 0 else p_cutoff[1]
    select = ((max_probs >= abnormal_cutoff) | (max_probs <= normal_cutoff)).long()

    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    target = max_idx.float()  # 类别标签转换为浮点数（0或1）
    # logits_strong (batch_size, 1)，需要squeeze成 (batch_size)
    if len(logits) == 2:
        loss = criterion(logits_strong.squeeze(), target)
    else:
        loss = criterion(logits_weak.squeeze(), target)
    masked_loss = loss * mask
    return masked_loss.mean(), mask.mean(), select, max_idx.long()


def compute_beta(model, train_indices, val_indices, graph, labels, dataloader, device, args):
    gamma = float(args.gamma)
    damping = float(args.damping)
    model.eval()
    # Step 1: Compute Hessian diagonal approximation over entire training set
    hessian_diag = None
    logits_train = None
    for i, (batch_size, n_id, adjs) in enumerate(dataloader):
        x = graph.x[n_id].to(device)
        adjs = [adj.to(device) for adj in adjs]
        if i == 0:
            logits_train = model(x, adjs)
        else:
            logits_train = torch.cat((logits_train, model(x, adjs)), dim=0)
    loss = nn.BCEWithLogitsLoss()(logits_train, labels[train_indices].unsqueeze(1).float())

    # First-order gradient
    grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True, allow_unused=True)
    # grad_vec = torch.cat([g.view(-1) for g in grad_params])
    grad_vec = [g.view(-1) if g is not None else torch.zeros_like(p).view(-1)
             for g, p in zip(grad_params, model.parameters())]
    grad_vec = torch.cat(grad_vec)

    # Second-order diagonal Hessian (per-sample)
    v = torch.ones_like(grad_vec)
    grad_v_dot = torch.dot(grad_vec, v)
    hvp_params = torch.autograd.grad(grad_v_dot, model.parameters(), retain_graph=True, create_graph=True, allow_unused=True)
    hessian_vec = [g.view(-1) if g is not None else torch.zeros_like(p).view(-1)
             for g, p in zip(hvp_params, model.parameters())]
    hessian_vec = torch.cat(hessian_vec)

    # Accumulate Hessian
    if hessian_diag is None:
        hessian_diag = hessian_vec.detach()
    else:
        hessian_diag +=  hessian_vec.detach()

    # Average and invert with damping
    hessian_diag /= len(train_indices)
    damping = damping * torch.mean(hessian_diag)
    H_inv = 1.0 / (hessian_diag + damping)  # 对角Hessian逆

    # Step 2: Compute influence scores
    # Compute validation loss
    val_dataloader = NeighborSampler(graph.edge_index,
                                      node_idx=val_indices,
                                      sizes=args.sampling_sizes,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      drop_last=False,
                                      pin_memory=True,
                                      num_workers=8)

    for i_num, (batch_size, n_id, adjs) in enumerate(val_dataloader):
        x = graph.x[n_id].to(device)
        adjs = [adj.to(device) for adj in adjs]
        if i_num == 0:
            logits_train = model(x, adjs)
        else:
            logits_train = torch.cat((logits_train, model(x, adjs)), dim=0)
    loss_val = nn.BCEWithLogitsLoss()(logits_train, labels[val_indices].unsqueeze(1).float())
    grad_val_loss = torch.autograd.grad(loss_val, model.parameters(), create_graph=True, allow_unused=True)
    grad_val_loss_vec = [g.view(-1) if g is not None else torch.zeros_like(p).view(-1) for g, p in zip(grad_val_loss, model.parameters())]
    grad_val_loss_vec = torch.cat(grad_val_loss_vec)

    # compute the influence score for each training sample to the validation loss

    single_dataloader = NeighborSampler(graph.edge_index,
                                      node_idx=train_indices,
                                      sizes=args.sampling_sizes,
                                      batch_size=1,
                                      shuffle=False,
                                      drop_last=False,
                                      pin_memory=True)
    # Compute validation influence
    influences = []
    for i, (batch_size, n_id, adjs) in enumerate(tqdm(single_dataloader)):  # 遍历训练集, 每个batch是单个训练样本
        single_sample = graph.x[n_id].to(device)
        single_adjs = [adj.to(device) for adj in adjs]
        single_logits = model(single_sample, single_adjs)
        single_energy = model.energy(single_logits)
        grad_energy = torch.autograd.grad(
            single_energy, model.parameters(), retain_graph=True, create_graph=True, allow_unused=True
        )
        grad_energy_vec = [g.view(-1) if g is not None else torch.zeros_like(p).view(-1) for g, p in zip(grad_energy, model.parameters())]
        grad_energy_vec = torch.cat(grad_energy_vec)
        influence = -1 * torch.dot(grad_val_loss_vec, H_inv * grad_energy_vec).item()
        influences.append(influence)

    # Step 3: Normalize and scale
    influences = torch.tensor(influences)
    normed_influences = -gamma * influences / influences.abs().max()
    if torch.sum(normed_influences < 0) > 0:
        print("Warning: some influences are negative.")
    return normed_influences.to(device)

def anomaly_mixup(args, model, graph, idx_train_anomaly, ppr_matrix=None):
    model.train()
    b_xent = nn.CrossEntropyLoss()
    sample_size_for_anormaly = int(args.sample_size_for_anormaly)
    selected_idx = np.random.choice(idx_train_anomaly, sample_size_for_anormaly, replace=False)
    selected_idx_tensor = torch.from_numpy(selected_idx)
    anomaly_tainloader = NeighborSampler(graph.edge_index,
                                         node_idx=selected_idx_tensor,
                                         sizes=args.sampling_sizes,
                                         batch_size=len(selected_idx),
                                         shuffle=False,
                                         drop_last=False,
                                         pin_memory=True)
    for i, (batch_size, n_id, adjs) in enumerate(anomaly_tainloader):
        x = graph.x[n_id].to(args.device)
        adjs = [adj.to(args.device) for adj in adjs]
        sub_ppr_matrix = torch.from_numpy(ppr_matrix[np.ix_(n_id.cpu().numpy(),  n_id.cpu().numpy())]).float().to(args.device)
        logits, div_loss = model(x, adjs, sub_ppr_matrix)
        logits1 = logits
        logits2 = logits.T.contiguous()
        new_label = torch.arange(logits.size(0)).long().cuda()
        loss = b_xent(logits1, new_label) / 2 + b_xent(logits2, new_label) / 2 - args.div_loss_alpha * div_loss
        return loss


def anomaly_mixup_v2(args, model, graph, idx_train_anomaly, ppr_matrix=None):
    model.train()
    b_xent = nn.CrossEntropyLoss()
    # sample_size_for_anormaly = int(args.sample_size_for_anormaly)
    # selected_idx = np.random.choice(idx_train_anomaly, sample_size_for_anormaly, replace=False)
    # selected_idx_tensor = torch.from_numpy(selected_idx)
    anomaly_tainloader = NeighborSampler(graph.edge_index,
                                         node_idx=idx_train_anomaly,
                                         sizes=args.sampling_sizes,
                                         batch_size=len(idx_train_anomaly),
                                         shuffle=False,
                                         drop_last=False,
                                         pin_memory=True,
                                         num_workers=8)
    for i, (batch_size, n_id, adjs) in enumerate(anomaly_tainloader):
        x = graph.x[n_id].to(args.device)
        adjs = [adj.to(args.device) for adj in adjs]
        sub_ppr_matrix = torch.from_numpy(ppr_matrix[np.ix_(n_id.cpu().numpy(),  n_id.cpu().numpy())]).float().to(args.device)
        embeds, div_loss = model(x, adjs, sub_ppr_matrix)
    return embeds, div_loss

def anomaly_mixup_v3(args, model, graph, idx_train_anomaly, ppr_matrix=None):
    model.train()
    loss_type = args.mixup_loss_type
    if loss_type == 'consistency':
        b_xent = nn.MSELoss()
    elif loss_type == 'contrastive':
        b_xent = nn.CrossEntropyLoss()
    sample_size_for_anormaly = int(args.sample_size_for_anormaly)
    selected_idx = np.random.choice(idx_train_anomaly, sample_size_for_anormaly, replace=False)
    selected_idx_tensor = torch.from_numpy(selected_idx)
    anomaly_tainloader = NeighborSampler(graph.edge_index,
                                         node_idx=selected_idx_tensor,
                                         sizes=args.sampling_sizes,
                                         batch_size=len(selected_idx),
                                         shuffle=False,
                                         drop_last=False,
                                         pin_memory=True)
    for i, (batch_size, n_id, adjs) in enumerate(anomaly_tainloader):
        x = graph.x[n_id].to(args.device)
        adjs = [adj.to(args.device) for adj in adjs]
        sub_ppr_matrix = torch.from_numpy(ppr_matrix[np.ix_(n_id.cpu().numpy(),  n_id.cpu().numpy())]).float().to(args.device)
        logits, div_loss = model(x, adjs, sub_ppr_matrix, loss_type)
    if loss_type == 'consistency':
        logits1, logits2 = logits
        loss = b_xent(logits1, logits2) - args.div_loss_alpha * div_loss
    elif loss_type == 'contrastive':
        logits1 = logits
        logits2 = logits.T.contiguous()
        new_label = torch.arange(logits.size(0)).long().cuda()
        loss = b_xent(logits1, new_label) / 2 + b_xent(logits2, new_label) / 2 - args.div_loss_alpha * div_loss
    return loss
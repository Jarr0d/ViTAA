import torch
import logging
import numpy as np

import torch.nn.functional as F


def rank(similarity, q_pids, g_pids, max_rank=10):
    num_q, num_g = similarity.size()
    indices = torch.argsort(similarity, dim=1, descending=True)
    matches = g_pids[indices].eq(q_pids.view(-1, 1))

    # compute cmc curve for each query
    all_cmc = [] # number of valid query
    for q_idx in range(num_q):
        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx]
        cmc = orig_cmc.cumsum(0)
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])

    all_cmc = torch.stack(all_cmc).float()
    all_cmc = all_cmc.sum(0) / num_q
    return all_cmc


def re_rank(similarity, q_pids, g_pids, q_locals, g_locals, q_masks, max_rank=10):
    num_q, num_g = similarity.size()
    indices = torch.argsort(similarity, dim=1, descending=True)
    matches = g_pids[indices].eq(q_pids.view(-1, 1))

    # compute cmc curve for each query
    all_cmc = [] # number of valid query
    for q_idx in range(num_q):
        # get query pid
        q_mask = q_masks[q_idx]
        order = indices[q_idx][:max_rank]

        g_local = g_locals[order]
        q_local = q_locals[q_idx].expand_as(g_local)
        g_local = F.normalize(g_local, p=2, dim=2)
        q_local = F.normalize(q_local, p=2, dim=2)

        # remove gallery samples that have the same pid with query
        local_similarity = torch.matmul(q_local, g_local.permute(0, 2, 1))
        local_similarity = local_similarity.diagonal(dim1=-2, dim2=-1)
        local_similarity = local_similarity
        local_similarity = local_similarity.sum(1)

        global_similarity = similarity[q_idx][order]
        total = global_similarity + local_similarity / 5
        index = torch.argsort(total, descending=True)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][:max_rank]
        if q_mask.nonzero().numel() >= 2:
            orig_cmc = orig_cmc[index]
        cmc = orig_cmc.cumsum(0)
        cmc[cmc > 1] = 1
        all_cmc.append(cmc)

    all_cmc = torch.stack(all_cmc).float()
    all_cmc = all_cmc.sum(0) / num_q

    return all_cmc


def evaluation(
    dataset,
    predictions,
    output_folder,
    topk,
):
    logger = logging.getLogger("PersonSearch.inference")
    annotations = dataset.dataset['annotations']

    image_ids, pids = [], []
    image_global, text_global = [], []
    for idx, prediction in predictions.items():
        image_id, pid = dataset.get_id_info(idx)
        image_ids.append(image_id)
        pids.append(pid)
        image_global.append(prediction[0])
        text_global.append(prediction[1])

    image_pid = torch.tensor(pids)
    text_pid = torch.tensor(pids)
    image_global = torch.stack(image_global, dim=0)
    text_global = torch.stack(text_global, dim=0)

    keep_id = []
    tmp = 0
    for idx, image_id in enumerate(image_ids):
        if (image_id - tmp) > 0:
            keep_id.append(idx)
            tmp = image_id
    keep_id = torch.tensor(keep_id)
    image_global = image_global[keep_id]
    image_pid = image_pid[keep_id]

    image_global = F.normalize(image_global, p=2, dim=1)
    text_global = F.normalize(text_global, p=2, dim=1)
    similarity = torch.matmul(text_global, image_global.t())
    t2i_cmc = rank(similarity, text_pid, image_pid, max(topk))
    logger.info(
        'R@1: {:.3f}%, R@5: {:.3f}%, R@10: {:.3f}%'.format(
            t2i_cmc[topk[0] - 1] * 100,
            t2i_cmc[topk[1] - 1] * 100,
            t2i_cmc[topk[2] - 1] * 100)
    )


def cuhkpedes_evaluation(
    dataset,
    predictions,
    output_folder
):
    return evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        topk=[1, 5, 10],
    )
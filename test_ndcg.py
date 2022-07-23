import os
import json
import math
import shutil
import argparse
import numpy as np
from sklearn.metrics import ndcg_score


def load_json(path: str):
    file = open(path)
    res = json.load(file)
    file.close()
    return res


def save_json(obj: object, path: str):
    file = open(path, 'w')
    json.dump(obj, file)
    file.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', help='list of the data json file', required=True)
    parser.add_argument('--answer', '-a', help='list of the answer txt file', required=True)
    parser.add_argument('--path', '-p', help='the path of models (file or directory)', required=True)
    parser.add_argument('--res_path', '-rp', help='the original score result path', required=True)
    parser.add_argument('--model_name', '-mn', help='the ndcg result path', required=True)
    parser.add_argument('--ndcg_n', '-n', help='the ndcg parameter n', type=int, default=-1)
    parser.add_argument('--use_exp', '-ue', help='if use exponential in ndcg (depreciated)', action='store_true')
    parser.add_argument('--mid_bound', '-b', help='the upper bound of model_id', type=int, default=120)
    parser.add_argument('--ignore_index', '-i', help='the ignored group_id', type=str, default='')
    return parser.parse_args()


# depreciated
def calculate_ndcg(pmid_score, pmid_label, use_exp=False, n: int = -1):
    rank_by_score = sorted(list(pmid_score.items()), key=lambda x: x[1], reverse=True)
    rank_by_score = [item[0] for item in rank_by_score]
    dcg = 0.0
    if n > 0:
        rank_by_score = rank_by_score[:n]
    for rank, pmid in enumerate(rank_by_score):
        dcg += ((2 ** pmid_label[pmid] - 1) if use_exp else pmid_label[pmid]) / math.log2(rank + 2)

    rank_by_label = sorted(list(pmid_label.items()), key=lambda x: x[1], reverse=True)
    rank_by_label = [item[0] for item in rank_by_label]
    idcg = 0.0
    if n > 0:
        rank_by_label = rank_by_label[:n]
    for rank, pmid in enumerate(rank_by_label):
        idcg += ((2 ** pmid_label[pmid] - 1) if use_exp else pmid_label[pmid]) / math.log2(rank + 2)
    return dcg / idcg


def calculate_ndcg_sklearn(pmid_score: dict, pmid_label: dict, n: int = -1):
    if n <= 0:
        n = None
    assert set(pmid_score.keys()) == set(pmid_label.keys())
    score_list = [item[1] for item in sorted(list(pmid_score.items()), key=lambda x: x[0])]
    label_list = [item[1] for item in sorted(list(pmid_label.items()), key=lambda x: x[0])]
    assert len(score_list) == len(label_list)
    return ndcg_score(y_true=[label_list], y_score=[score_list], k=n, ignore_ties=False)


def calc_minimum_dist(entity_h, entity_t, sent_lengths, eps=1e-5):
    min_dist = float('inf')
    for i in range(len(entity_h)):
        for j in range(len(entity_t)):
            offset_h = sum(sent_lengths[:entity_h[i]['sent_id']]) + entity_h[i]['pos'][0]
            offset_t = sum(sent_lengths[:entity_t[j]['sent_id']]) + entity_t[j]['pos'][0]
            dist = abs(offset_h - offset_t)
            min_dist = min(min_dist, dist)
            if min_dist == 0:
                return eps
    return min_dist


def get_baseline_metric(args, data, method, pmid_label):
    pmid_score = {}
    pmid_key = 'pmid' if 'pmid' in data[0] else 'pmsid'
    if method == 'total_num':
        for doc in data:
            assert len(doc['vertexSet']) == 2
            pmid_score[str(doc[pmid_key])] = len(doc['vertexSet'][0]) + len(doc['vertexSet'][1])
    elif method == 'minimum_distance':
        for doc in data:
            assert len(doc['vertexSet']) == 2
            entity_h = doc['vertexSet'][0]
            entity_t = doc['vertexSet'][1]
            sent_l = [len(sent) for sent in doc['sents']]
            pmid_score[str(doc[pmid_key])] = 1 / calc_minimum_dist(entity_h, entity_t, sent_l)
    else:
        raise RuntimeError('Unknown Method!')
    return calculate_ndcg_sklearn(pmid_score, pmid_label, n=args.ndcg_n)


def get_model_metric(args, data_list, ans_list, pkl_path, model_res_path, pmid_label_list, model_name=''):
    total_ndcg, cur_data_id = 0., 0
    assert len(data_list) == len(ans_list) == len(pmid_label_list)

    for data_path, and_path in zip(data_list, ans_list):
        rank_file = os.path.basename(data_path)
        score_path = f'{os.path.splitext(rank_file)[0]}_score.npy'
        title_path = f'{os.path.splitext(rank_file)[0]}_pmid2range.json'
        if model_name == '':
            target_score_path = os.path.join(model_res_path, score_path)
            target_title_path = os.path.join(model_res_path, title_path)
        else:
            target_score_path = f'{os.path.splitext(rank_file)[0]}_score_{model_name}.npy'
            target_title_path = f'{os.path.splitext(rank_file)[0]}_pmid2range_{model_name}.json'
            target_score_path = os.path.join(model_res_path, target_score_path)
            target_title_path = os.path.join(model_res_path, target_title_path)

        if not (os.path.exists(target_score_path) and os.path.exists(target_title_path)):
            if os.path.getsize(pkl_path) < 100:
                # placeholder virtual model
                return 0.0
            log_path = os.path.join(model_res_path, 'err_log.txt')
            cmd = f'python main.py -t denoise -m test -rf {data_path} -c {pkl_path} 1>> {log_path} 2>> {log_path}'
            print('------', cmd, '------')
            assert os.system(cmd) == 0
            shutil.move(os.path.join(args.res_path, score_path), target_score_path)
            shutil.move(os.path.join(args.res_path, title_path), target_title_path)

        pmid_score = {}
        titles = load_json(target_title_path)
        scores = np.load(target_score_path)
        assert len(scores) == len(titles)
        for pmid, pos in titles.items():
            assert pos[0] + 1 == pos[1]
            pmid_score[str(pmid)] = scores[pos[0]]
        total_ndcg += calculate_ndcg_sklearn(pmid_score, pmid_label_list[cur_data_id], n=args.ndcg_n)
        cur_data_id += 1

    return total_ndcg / len(data_list)


def main():
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    args = parse_args()
    os.makedirs('test_ndcg', exist_ok=True)
    model_res_path = os.path.join('test_ndcg', args.model_name)
    os.makedirs(model_res_path, exist_ok=True)

    label_to_rank = {
        (1, 3): 5, (1, 2): 4, (1, 1): 3,
        (0, 1): 2, (0, 2): 1, (0, 3): 0,
        (1, 2.5): 4.5, (1, 1.5): 3.5,
        (0, 1.5): 1.5, (0, 2.5): 0.5,
    }
    # Dict[pmid, similarity]
    fout = open(os.path.join(model_res_path, 'result.txt'), 'a')

    # ignore groups
    ignore_indexes = []
    for idx in args.ignore_index.split(','):
        if '-' in idx:
            start, end = idx.split('-')
            for k in range(int(start), int(end) + 1):
                ignore_indexes.append(k)
        elif idx != '':
            ignore_indexes.append(int(idx))

    p_list = args.data.split(',')
    data_list = []
    for pth in p_list:
        if os.path.isdir(pth):
            data_list += [os.path.join(pth, f) for f in os.listdir(pth) if f.endswith('_sample.json')]
        else:
            assert pth.endswith('_sample.json')
            data_list.append(pth)
    data_list = [x for x in data_list if int(os.path.basename(x).split('_')[0]) not in ignore_indexes]
    p_list = args.answer.split(',')
    ans_list = []
    for pth in p_list:
        if os.path.isdir(pth):
            ans_list += [os.path.join(pth, f) for f in os.listdir(pth)
                         if f.endswith('_ans.txt') or f.endswith('_ans_unchecked.txt')]
        else:
            assert pth.endswith('_ans.txt') or pth.endswith('_ans_unchecked.txt')
            ans_list.append(pth)
    ans_list = [x for x in ans_list if int(os.path.basename(x).split('_')[0]) not in ignore_indexes]
    data_list.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
    ans_list.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))

    assert len(data_list) == len(ans_list)
    print('total number of files:', len(data_list))

    total_num_ndcg, min_dist_ndcg, pmid_label_list = 0., 0., []

    for data_path, ans_path in zip(data_list, ans_list):
        data = load_json(data_path)
        pmid_key = 'pmid' if 'pmid' in data[0] else 'pmsid'
        with open(ans_path, 'r') as fin:
            answers = [line.strip() for line in fin.readlines() if len(line.strip()) > 0]
        assert len(data) == len(answers)
        pmid_label = {}
        for doc, ans in zip(data, answers):
            tokens = ans.split(',')
            assert len(tokens) == 3
            pmid, h, r = tokens[0], int(tokens[1]), float(tokens[2])
            assert pmid == str(doc[pmid_key])
            pmid_label[str(pmid)] = label_to_rank[(h, r)]
        total_num_ndcg += get_baseline_metric(args, data, 'total_num', pmid_label)
        min_dist_ndcg += get_baseline_metric(args, data, 'minimum_distance', pmid_label)
        pmid_label_list.append(pmid_label)

    total_num_ndcg /= len(data_list)
    min_dist_ndcg /= len(data_list)

    print('------', 'total_num', round(total_num_ndcg, 4), '------')
    print('------', 'minimum_distance', round(min_dist_ndcg, 4), '------')
    print('------', args.data, args.model_name, args.ndcg_n, '------', file=fout)
    print('------', 'total_num', round(total_num_ndcg, 4), '------', file=fout)
    print('------', 'minimum_distance', round(min_dist_ndcg, 4), '------', file=fout)
    fout.flush()

    def model_sort_key(pkl_x):
        pre_x = pkl_x[:-4]
        if '-' in pre_x:
            ep, st = pre_x.split('-')
        else:
            ep, st = pre_x, 1e8
        return int(ep), int(st)

    assert os.path.exists(args.path)
    if os.path.isdir(args.path):
        perform_list, model_name_list = [], []
        model_list = sorted([item for item in os.listdir(args.path) if item.endswith('.pkl')],
                            key=lambda x: model_sort_key(x))
        for pkl_name in model_list:
            model_name = pkl_name[:-4]
            epoch_num = int(model_name.split('-')[0]) if '-' in model_name else int(model_name)
            if epoch_num > args.mid_bound:
                continue
            pkl_path = os.path.join(args.path, f'{model_name}.pkl')
            ndcg = get_model_metric(args, data_list, ans_list, pkl_path, model_res_path, pmid_label_list, model_name)
            perform_list.append(ndcg)
            model_name_list.append(model_name)
            fout.write(f'{args.model_name}\t{model_name}\t{round(ndcg, 4)}\n')
            fout.flush()
        max_idx = np.argmax(perform_list)
        max_name = model_name_list[max_idx]
        print('------', args.path, max_name, round(perform_list[max_idx], 4), '------')
        print('------', args.path, max_name, round(perform_list[max_idx], 4), '------', file=fout)
    else:
        pkl_name = os.path.basename(args.path)
        assert pkl_name.endswith('.pkl')
        model_name = pkl_name[:-4]
        ndcg = get_model_metric(args, data_list, ans_list, args.path, model_res_path, pmid_label_list, model_name)
        fout.write(f'{args.model_name}\t{round(ndcg, 4)}\n')
        print('------', args.path, round(ndcg, 4), '------')
        print('------', args.path, round(ndcg, 4), '------', file=fout)
    fout.write('\n')
    fout.close()


if __name__ == '__main__':
    main()

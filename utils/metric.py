from .io_utils import load_json
import time
from tqdm import tqdm
import numpy as np
from config import ConfigFineTune, ConfigPretrain


last_tag = None
counts = {}
global_metric_flag = False
global_file = None


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def set_metric():
    global global_metric_flag
    global_metric_flag = True


def unset_metric():
    global global_metric_flag
    global_metric_flag = False


def set_file(file):
    global global_file
    global_file = file


def unset_file():
    global global_file
    global_file = None


def clear_count():
    counts.clear()


def time_tag(idx: int, output=False, *args):
    global last_tag, counts, global_metric_flag, global_file
    if not global_metric_flag:
        return
    if idx == 0:
        last_tag = None
    if last_tag is None:
        last_tag = time.time()
        if output:
            print(f'[metric time tag] id: {idx} begin time: '
                  f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_tag))} args:', end='', file=global_file)
    else:
        cur = time.time()
        pass_time = cur - last_tag
        last_tag = cur
        if output:
            print(f'[metric time tag] id: {idx} passed time: '
                  f'{pass_time} seconds args:', end='', file=global_file)
        if idx not in counts:
            counts[idx] = [pass_time, 1]
        else:
            counts[idx][0] += pass_time
            counts[idx][1] += 1
    if output:
        for arg in args:
            print(' ', str(arg), end='', file=global_file)
        print(file=global_file)


def print_time_stat(file=None):
    global global_file
    if file is None:
        file = global_file
    print('------ stat ------', file=file)
    total_list = []
    for idx, cnt in counts.items():
        mean_time = 'null'
        if cnt[1] > 0:
            mean_time = cnt[0] / cnt[1]
            total_list.append((idx, mean_time, cnt[1]))
        print(f'[metric time stat] id: {idx} mean_time: {mean_time} count: {cnt[1]}', file=file)
    print('------ sorted ------', file=file)
    total_list.sort(key=lambda x: x[1], reverse=True)
    for idx, mean_time, num in total_list:
        print(f'[metric time stat] id: {idx} mean_time: {mean_time} count: {num}', file=file)


def get_denoise_pair_num(mode: str):
    if mode == 'pretrain':
        config = ConfigPretrain
        mode = 'train'
    else:
        config = ConfigFineTune
    data = load_json(config.data_path[mode])
    use_cp = 'chemprot' in config.data_path[mode].lower()
    res_list = {}
    if use_cp:
        for item in data:
            cnt = 0
            entities = item['vertexSet']
            entity_num = len(entities)
            for i in range(entity_num):
                for j in range(entity_num):
                    if entities[i][0]['type'].lower().startswith('chemical') and \
                            entities[j][0]['type'].lower().startswith('gene'):
                        cnt += 1
            res_list[item['title']] = cnt
    else:
        for item in data:
            entities = item['vertexSet']
            entity_num = len(entities)
            res_list[item['title']] = entity_num * (entity_num - 1)
    return res_list


def check_threshold(threshold: float):
    """找到最好的正例得分分界点"""
    mode = 'valid'
    if mode == 'pretrain':
        config = ConfigPretrain
        # 统计实际参与降噪的实体对个数
        num_list = get_denoise_pair_num('pretrain')
        mode = 'train'
    else:
        config = ConfigFineTune
        num_list = get_denoise_pair_num(mode)
    scores = np.load(config.score_path[mode])
    titles = load_json(config.title_path[mode])
    assert len(num_list) == scores.shape[0] == len(titles)
    data = load_json(config.data_path[mode])
    use_cp = 'chemprot' in config.data_path[mode].lower()
    title_to_item = {}
    for item in data:
        assert item['title'] not in title_to_item
        title_to_item[item['title']] = item
    tp, predict, instance = 0, 0, 0
    for idx in tqdm(range(len(titles))):
        item = title_to_item[titles[idx]]
        score = scores[idx, :num_list[titles[idx]]]
        label_set = set()
        for lab in item['labels']:
            label_set.add((lab['h'], lab['t']))
        instance += len(label_set)
        pair_id = 0
        if use_cp:
            entities = item['vertexSet']
            entity_num = len(entities)
            for i in range(entity_num):
                for j in range(entity_num):
                    if entities[i][0]['type'].lower().startswith('chemical') and \
                            entities[j][0]['type'].lower().startswith('gene'):
                        if score[pair_id] > threshold:
                            # predict to be positive
                            predict += 1
                            # true positive
                            tp += int((i, j) in label_set)
                        pair_id += 1
        else:
            entities = item['vertexSet']
            entity_num = len(entities)
            for i in range(entity_num):
                for j in range(entity_num):
                    if i != j:
                        if score[pair_id] > threshold:
                            # predict to be positive
                            predict += 1
                            # true positive
                            tp += int((i, j) in label_set)
                        pair_id += 1
        # print(pair_id, use_cp, num_list[titles[idx]])
        assert pair_id == num_list[titles[idx]]
    precision = tp / predict if predict > 0 else 0
    recall = tp / instance if instance > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f'threshold: {threshold}\ttp: {tp}\tpredict: {predict}\tinstance: {instance}\t'
          f'precision: {precision}\trecall: {recall}\tf1: {f1}')
    return tp, predict, instance, precision, recall, f1

import os
import torch
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable
from config import ConfigBase
from timeit import default_timer as timer
from utils import name_to_metric, print_value, time_to_str, save_json


def test(model, datasets, mode: str, config: ConfigBase, path: str = None, epoch: int = None):
    model.eval()

    eval_res = None
    total_loss = 0
    dataset = datasets[mode]
    test_len = len(dataset)
    start_time = timer()
    use_gpu = config.use_gpu

    output_time = config.output_step
    step = -1

    # for test
    docred_res = []
    na_label = config.label2id['NA']

    pbar = tqdm(range(len(dataset))) if mode == 'test' else None

    for step, data in enumerate(dataset):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = Variable(value.to(config.gpu_device)) if use_gpu else Variable(value)

        if mode == 'test':
            result = model(data, 'test')
            predict, pair_ids, titles, scores = result['predict'], result['pair_ids'], result['titles'], result['score']
            try:
                assert len(predict) == len(pair_ids) == len(titles) == len(scores)
                assert len(predict[0]) == len(scores[0])
                assert len(predict[0]) >= len(pair_ids[0])
            except AssertionError as err:
                print(len(pair_ids[0]), len(pair_ids[0][0]), len(scores[0]))
                print(np.shape(predict), np.shape(pair_ids), np.shape(titles), np.shape(scores))
                raise err
            for doc_id in range(len(pair_ids)):  # per batch
                for pair_id in range(len(pair_ids[doc_id])):  # per pair
                    label = predict[doc_id][pair_id]
                    score = float(scores[doc_id][pair_id])
                    if label == na_label:  # filter NA predictions
                        continue
                    label = config.id2label[label]
                    pair = pair_ids[doc_id][pair_id]
                    docred_res.append({'title': titles[doc_id], 'h_idx': pair[0],
                                       't_idx': pair[1], 'r': label, 's': score})
            pbar.update()
            continue

        result = model(data, 'valid', eval_res)
        loss, eval_res = result['loss'], result['eval_res']
        total_loss += float(loss)

        if step % output_time == 0:
            metric_json = name_to_metric[config.output_metric](eval_res, 'valid')
            time_spent = timer() - start_time
            print_value(epoch, 'valid', f'{step + 1}/{test_len}',
                        f'{time_to_str(time_spent)}/{time_to_str(time_spent*(test_len-step-1)/(step+1))}',
                        f'{(total_loss / (step + 1)):.3f}', metric_json,
                        os.path.join(path, f'{epoch}.txt'), '\r')

    if mode == 'valid':
        time_spent = timer() - start_time
        metric_json = name_to_metric[config.output_metric](eval_res, 'valid')
        print_value(epoch, 'valid', f'{step + 1}/{test_len}',
                    f'{time_to_str(time_spent)}/{time_to_str(time_spent * (test_len - step - 1) / (step + 1))}',
                    f'{(total_loss / (step + 1)):.3f}', metric_json,
                    os.path.join(path, f'{epoch}.txt'))
    else:
        pbar.close()
        test_output_path = os.path.join(config.model_path, config.model_name, 'test')
        os.makedirs(test_output_path, exist_ok=True)
        save_json(docred_res, os.path.join(test_output_path, 'result.json'))

    model.train()

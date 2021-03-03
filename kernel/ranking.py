import os
import torch
import numpy as np
from torch.autograd import Variable
from config import ConfigBase
from timeit import default_timer as timer
from utils import print_value, time_to_str, save_json


def rank(config: ConfigBase, models, datasets):
    os.makedirs(config.model_path, exist_ok=True)
    task_path = os.path.join(config.model_path, config.model_name)
    os.makedirs(task_path, exist_ok=True)
    rank_output_path = os.path.join(task_path, 'rank')
    os.makedirs(rank_output_path, exist_ok=True)

    with torch.no_grad():
        scores, titles = gen_score(config, models['model'], datasets['test'], rank_output_path)

    rank_file = os.path.basename(config.data_path['test'])
    score_path = os.path.join(config.rank_result_path, f'{os.path.splitext(rank_file)[0]}_score.npy')
    title_path = os.path.join(config.rank_result_path, f'{os.path.splitext(rank_file)[0]}_title.json')

    scores = np.vstack(scores)
    np.save(score_path, scores)
    save_json(titles, title_path)

    assert scores.shape[0] == len(titles)
    print(np.shape(scores), np.shape(titles))


def gen_score(config: ConfigBase, model, dataset, path: str):
    model.eval()

    eval_res = None
    total_loss = 0
    rank_len = len(dataset)
    start_time = timer()
    use_gpu = config.use_gpu

    output_time = config.output_step
    step = -1
    scores = []
    titles = []

    for step, data in enumerate(dataset):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = Variable(value.cuda()) if use_gpu else Variable(value)

        result = model(data, 'test', eval_res)
        scores += result['score'].cpu().tolist()
        titles += result['titles']

        if step % output_time == 0:
            time_spent = timer() - start_time
            print_value(0, 'test', f'{step + 1}/{rank_len}',
                        f'{time_to_str(time_spent)}/{time_to_str(time_spent*(rank_len-step-1)/(step+1))}',
                        f'{(total_loss / (step + 1)):.3f}', 'ranking',
                        os.path.join(path, 'log.txt'), '\r')

    time_spent = timer() - start_time
    print_value(0, 'test', f'{step + 1}/{rank_len}',
                f'{time_to_str(time_spent)}/{time_to_str(time_spent * (rank_len - step - 1) / (step + 1))}',
                f'{(total_loss / (step + 1)):.3f}', 'ranking',
                os.path.join(path, 'log.txt'))

    return scores, titles

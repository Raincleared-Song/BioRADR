import os
import shutil
import torch
from kernel import init_all, train, test, rank


def inspect(task_: str, mode_='valid'):
    print(f'inspecting {task_}/{mode_}')
    see_loss = False
    bound = -1
    path_base = os.path.join('checkpoint', task_, mode_)

    file_list = sorted(os.listdir(path_base), key=lambda x: int(x[:-4]))
    result = {}
    print(file_list)
    for file_name in file_list:
        fin = open(os.path.join(path_base, file_name))
        lines = [line for line in fin.readlines() if len(line.strip()) > 0]
        if see_loss:
            tokens = [token for token in lines[-1].split(' ') if len(token) > 0]
            result[int(file_name[:-4])] = float(tokens[5])
        else:
            find_key = '"f1":' if task_.startswith('finetune') else (
                '"RD": ' if task_.startswith('denoise') else '"MEM":')
            pos = lines[-1].find(find_key) + len(find_key) + 1
            end = pos
            while lines[-1][end] not in (',', '}'):
                end += 1
            result[int(file_name[:-4])] = float(lines[-1][pos: end])
        fin.close()
    print(result)
    if see_loss:
        min_key, min_value = -1, 1e9
        for key, value in result.items():
            if value < min_value and key > bound:
                min_key, min_value = key, value
        print(min_key, min_value)
        return min_key, min_value
    else:
        max_key, max_value = -1, -1
        for key, value in result.items():
            if value > max_value:
                max_key, max_value = key, value
        print(max_key, max_value)
        return max_key, max_value


if __name__ == '__main__':
    config, models, datasets, task, mode = init_all()
    if mode == 'train':
        train(config, models, datasets)
    elif task == 'denoise':
        rank(config, models, datasets)
    else:
        with torch.no_grad():
            test(models['model'], datasets, 'test', config)

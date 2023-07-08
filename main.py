import os
import shutil
import torch
from config import ConfigFineTune as Config
from kernel import init_all, init_args, train, test, rank


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
            find_key = '"f1":' if 'finetune' in task_ else (
                '"RD": ' if 'denoise' in task_ else '"RD":')
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
    args = init_args()
    # repeated experiments for cdr5
    is_cdr = args.task == 'finetune' and args.mode == 'train' and 'CDR' in Config.data_path['train']
    if is_cdr:
        task_path = Config.model_name

        seeds = list(range(90, 100))
        model_path = f'checkpoint/{task_path}/model'
        target_path = f'checkpoint/{task_path}/model_all'
        os.makedirs(target_path, exist_ok=True)
        for it in range(5):
            torch.cuda.empty_cache()
            print(f'epoch {it} for {task_path} ......')
            config, models, datasets, task, mode = init_all(seeds[it])
            train(config, models, datasets, it)
            # os.system(f'python fake.py test{it}')

            m_key, m_value = inspect(task_path, f'valid{it}')
            print(f'{model_path}/{m_key}.pkl', f'{target_path}/{it}-{m_key}.pkl')
            shutil.copy(f'{model_path}/{m_key}.pkl', f'{target_path}/{it}-{m_key}.pkl')
            os.system(f'rm -rf {model_path}/*')

            del config, models, datasets, task, mode
            import torch
            torch.cuda.empty_cache()
        exit()

    # normal setting
    config, models, datasets, task, mode = init_all()
    if mode == 'train':
        train(config, models, datasets)
    elif task == 'denoise':
        rank(config, models, datasets)
    else:
        with torch.no_grad():
            test(models['model'], datasets, 'test', config)

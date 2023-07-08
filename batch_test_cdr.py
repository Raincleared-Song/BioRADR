import os
import json
import argparse
import subprocess
import numpy as np
from config import ConfigFineTune as Config


def calculate_bound(x):
    if x[0] < 1:
        x = np.array(x) * 100
    return f'{np.round(np.mean(x), 2)}±{np.round(np.std(x), 2)}'


def test_cdr_model(model_name):
    out_file = open('test_cdr_log_new.txt', 'a', encoding='utf-8')
    out_train = open('test_cdr_train_log_new.txt', 'a', encoding='utf-8')

    config_out = open('config/auto.json', 'w', encoding='utf-8')
    json.dump({
        'model_name': model_name,
        'learning_rate': 2e-5,
    }, config_out)
    config_out.close()

    base_path = os.path.join('checkpoint', model_name)
    model_path = os.path.join(base_path, 'model_all')
    model_list = sorted(os.listdir(model_path))
    valid_epochs, valid_f1 = [], []
    test_epochs, test_f1 = [], []
    for it, model in enumerate(model_list):
        itt, epoch = model[:-4].split('-')
        assert str(it) == itt
        valid_path = os.path.join(base_path, f'valid{it}', f'{epoch}.txt')
        fin = open(valid_path, 'r', encoding='utf-8')
        line = [x for x in fin.readlines() if len(x.strip()) > 0][-1]
        fin.close()
        find_key = '"f1":'
        pos = line.find(find_key) + len(find_key) + 1
        end = pos
        while line[end] not in (',', '}'):
            end += 1
        valid_epochs.append(int(epoch))
        valid_f1.append(round(float(line[pos: end]) * 100, 2))

        model_whole = os.path.join(model_path, model)
        print(f'------ testing model {model_whole} ------')
        print(f'------ testing model {model_whole} ------', file=out_file)
        args = ['python', 'main.py', '-t', 'finetune', '-m', 'test', '-c', model_whole]
        print(' '.join(args))
        print('------', ' '.join(args), '------', file=out_train)
        ret = subprocess.Popen(args, stdout=out_train, stderr=out_train)
        assert ret.wait() == 0

        # eval step
        args = ['python', 'eval_cdr.py', '-r', f'checkpoint/{model_name}/test/result.json', '-o',
                f'checkpoint/{model_name}/test/score-{model}.txt']
        cmd = ' '.join(args)
        print(cmd)
        print('------', cmd, '------', file=out_file)
        out_file.flush()
        fin = os.popen(cmd, 'r')
        lines = fin.readlines()
        fin.close()
        for line in lines:
            out_file.write(line)
            if line.startswith('RE_F1:'):
                test_epochs.append(int(epoch))
                test_f1.append(round(float(line.split(' ')[1]), 4))
        out_file.flush()
    print(valid_epochs)
    print(valid_epochs, file=out_file)
    print(valid_f1)
    print(valid_f1, file=out_file)
    print(calculate_bound(valid_f1))
    print(calculate_bound(valid_f1), file=out_file)
    print(test_epochs)
    print(test_epochs, file=out_file)
    print(test_f1)
    print(test_f1, file=out_file)
    print(calculate_bound(test_f1))
    print(calculate_bound(test_f1), file=out_file)
    out_file.close()
    out_train.close()


if __name__ == '__main__':
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--task', '-t', help='the model name', type=str, required=True)
    args = arg_parser.parse_args()
    assert Config.model_name == args.task
    test_cdr_model(args.task)

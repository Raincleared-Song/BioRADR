import json
import numpy as np
import torch
import sys
from config import ConfigBase
# from apex import amp


def load_json(path: str):
    print(f'loading file {path} ......')
    file = open(path)
    res = json.load(file)
    file.close()
    return res


def save_json(obj: object, path: str):
    print(f'saving file {path} ......')
    file = open(path, 'w')
    json.dump(obj, file)
    file.close()


def save_model(path: str, model, optimizer, trained_epoch: int, global_step: int, config: ConfigBase):
    if hasattr(model, 'module'):
        model = model.module
    ret = {
        'model': model.state_dict(),
        'optimizer_name': config.optimizer,
        'optimizer': optimizer.state_dict(),
        'trained_epoch': trained_epoch,
        'global_step': global_step
    }
    # if config.fp16:
    #     ret['amp'] = amp.state_dict()
    try:
        torch.save(ret, path)
    except Exception as err:
        print(f'Save model failure with error {err}', file=sys.stderr)


def print_value(epoch, mode, step, time, loss, info, path: str, end='\n'):
    s = str(epoch) + " "
    while len(s) < 7:
        s += " "
    s += str(mode) + " "
    while len(s) < 14:
        s += " "
    s += str(step) + " "
    while len(s) < 25:
        s += " "
    s += str(time)
    while len(s) < 40:
        s += " "
    s += str(loss)
    while len(s) < 48:
        s += " "
    s += str(info)
    print(s, end=end)
    file = open(path, 'a')
    print(s, file=file)
    file.close()


def time_to_str(time):
    time = int(time)
    minute = time // 60
    second = time % 60
    return '%2d:%02d' % (minute, second)


def calculate_bound(x):
    if x[0] < 1:
        x = np.array(x) * 100
    return f'{np.round(np.mean(x), 2)}±{np.round(np.std(x), 2)}'


def print_json(obj):
    print(json.dumps(obj, sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))


def get_unused_token(index: int):
    if ConfigBase.model_type == 'bert':
        return f'[unused{index}]'
    else:
        return f'<0x{index:02X}>'

import json
import torch
import sys
from config import ConfigBase
from apex import amp


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
        'optimizer': optimizer.state_dict(),
        'trained_epoch': trained_epoch,
        'global_step': global_step
    }
    if config.fp16:
        ret['amp'] = amp.state_dict()
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

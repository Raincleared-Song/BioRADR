import random
import numpy.random
import torch
import argparse
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from search_model import SearchRankModel
from search_preprocess import process_denoise


def init_arg_model():
    args = init_args()
    init_seed(args.seed)
    model = init_models(args)
    return args, model


def init_args():
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.system('export TOKENIZERS_PARALLELISM=false')

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--checkpoint', '-c', help='path of the checkpoint file', required=True, type=str)
    arg_parser.add_argument('--seed', '-s', help='the random seed', default=66, type=int)
    arg_parser.add_argument('--batch_size', '-b', help='the batch size', default=16, type=int)
    arg_parser.add_argument('--device', '-d', help='the device name', default='cuda:0', type=str)
    return arg_parser.parse_args()


def init_seed(seed):
    assert seed is not None
    random.seed(seed)
    numpy.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_deterministic(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


class SearchDataset(Dataset):
    def __init__(self, docs: list):
        self.data = docs

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def init_data(args, docs: list):

    def collate_fn(data):
        return process_denoise(data, 'test')

    return DataLoader(
        dataset=SearchDataset(docs),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=5,
        collate_fn=collate_fn,
        drop_last=False,
        worker_init_fn=seed_worker
    )


def init_models(args):
    torch.cuda.set_device(args.device)
    model = SearchRankModel()
    use_gpu = args.device.strip().lower().startswith('cuda')
    if use_gpu:
        model = model.to(args.device)
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
        os.system("export CUDA_VISIBLE_DEVICES=0,1,2,3")

    assert args.checkpoint is not None
    params = torch.load(args.checkpoint, map_location={f'cuda:{k}': args.device for k in range(8)})
    model.load_state_dict(params['model'])

    return model

import random
from torch.utils.data import Dataset
from config import task_to_config
from utils import load_json


class DenoiseDataset(Dataset):
    def __init__(self, task: str, mode: str):
        assert task == 'denoise'
        config = task_to_config[task]
        self.data = load_json(config.data_path[mode])
        if mode != 'test':
            # abandon those data without labels
            self.data = [item for item in self.data if len(item['labels']) != 0 and self.test_cp_negative(item)]
            random.shuffle(self.data)
        if mode != 'train':
            self.total_len = 50 * config.batch_size[mode] * config.train_steps
        else:
            self.total_len = 100 * config.batch_size[mode] * config.train_steps

    def __getitem__(self, item):
        doc1, doc2 = random.sample(self.data, 2)
        return {
            'doc1': doc1, 'doc2': doc2
        }

    def __len__(self):
        return self.total_len

    @staticmethod
    def test_cp_negative(item: dict):
        num_c, num_g = 0, 0
        for entity in item['vertexSet']:
            if entity[0]['type'].lower().startswith('chemical'):
                num_c += 1
            else:
                num_g += 1
        return num_c * num_g > len([lab for lab in item['labels'] if lab['r'] != 'NA'])

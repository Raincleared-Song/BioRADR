from .FineTuneDataset import FineTuneDataset
from .PretrainDataset import PretrainDataset

task_to_dataset = {
    'denoise': FineTuneDataset,
    'finetune': FineTuneDataset,
    'pretrain': PretrainDataset
}

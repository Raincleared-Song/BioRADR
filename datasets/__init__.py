from .DenoiseDataset import DenoiseDataset
from .FineTuneDataset import FineTuneDataset
from .PretrainDataset import PretrainDataset

task_to_dataset = {
    'denoise_train': DenoiseDataset,
    # 'denoise_train': FineTuneDataset,
    'denoise_test': FineTuneDataset,
    'finetune': FineTuneDataset,
    'pretrain': PretrainDataset
}

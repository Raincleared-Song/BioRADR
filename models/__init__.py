from .FineTuneModel import FineTuneModel
from .DenoiseModel import DenoiseModel
from .PretrainModel import PretrainModel

task_to_model = {
    'denoise': DenoiseModel,
    'finetune': FineTuneModel,
    'pretrain': PretrainModel
}

from .config_base import ConfigBase
from .config_finetune import ConfigFineTune
from .config_pretrain import ConfigPretrain
from .config_denoise import ConfigDenoise

task_to_config = {
    'denoise': ConfigDenoise,
    'finetune': ConfigFineTune,
    'pretrain': ConfigPretrain
}

from .process_finetune import process_finetune
from .process_denoise import process_denoise
from .process_pretrain import process_pretrain

task_to_process = {
    'denoise': process_denoise,
    'finetune': process_finetune,
    'pretrain': process_pretrain
}

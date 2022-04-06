from .process_finetune import process_finetune
from .process_denoise import process_denoise
from .process_pretrain import process_pretrain
from .document_crop import document_crop, sentence_mention_crop

task_to_process = {
    'denoise': process_denoise,
    # 'denoise': process_finetune,
    'finetune': process_finetune,
    'pretrain': process_pretrain
}

import os
from transformers import AutoTokenizer
from torch.optim import Adam, SGD
from transformers.optimization import AdamW


class ConfigBase:
    tokenizer = AutoTokenizer.from_pretrained('../huggingface/scibert_scivocab_cased' if os.path.exists(
        '../huggingface/scibert_scivocab_cased') else 'allenai/scibert_scivocab_cased')
    relation_num = 2
    bert_hidden = 768
    valid_instance_cnt = 20683  # ctd_all
    optimizer_dict = {
        'adam': Adam,
        'adamw': AdamW,
        'sgd': SGD
    }
    rank_result_path = 'CTDRED/temp_range'

    label2id = {"NA": 0, "Pos": 1}
    id2label = ["NA", "Pos"]

    seed = 66

    label2id: dict
    id2label: list

    data_path: dict
    score_path: dict
    title_path: dict
    batch_size: dict

    reader_num: int
    token_padding: int  # token reserved for each document
    entity_padding: dict  # entity reserved for each document
    mention_padding: int  # mention reserved for each entity
    train_sample_limit: int
    score_sample_limit: int
    test_sample_limit: int
    hidden_size: int
    crop_documents: bool  # remove sentences not containing or surrounded by entities
    crop_mention_option: int  # remove redundant mentions, 0-not, 1-single, 2-more
    entity_marker_type: str

    do_validation: bool
    use_gpu: bool
    gpu_device: str
    fp16: bool

    optimizer: str
    model_path: str
    model_name: str
    model_class: str
    bert_path: str

    learning_rate: float
    weight_decay: int
    adam_epsilon: float
    warmup_ratio: float
    from_epoch: int = -1
    epoch_num: int
    real_epoch_num: int

    output_step: int
    save_global_step: int
    test_step: int
    lr_step_size: int
    lr_gamma: int
    train_steps: int  # optimizer step epoch number
    output_metric: str
    kept_pair_num: int  # number of pairs kept after predenoising

    positive_num: int
    negative_num: int

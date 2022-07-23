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


class ConfigDenoise(ConfigBase):
    data_path = {
        'train': 'CTDRED/ctd_train.json',
        'valid': 'CTDRED/ctd_dev.json',
        'test': 'CTDRED/ctd_test.json'
    }
    batch_size = {
        'train': 1,
        'valid': 16,
        'test': 16
    }

    reader_num = 4
    bert_path = '../huggingface/scibert_scivocab_cased' if os.path.exists(
        '../huggingface/scibert_scivocab_cased') else 'allenai/scibert_scivocab_cased'
    score_path = None
    token_padding = 1024  # token reserved for each document
    entity_padding = {
        'train': 37,
        'valid': 37,
        'test': 37
    }
    mention_padding = 3  # mention reserved for each entity
    train_sample_limit = 32
    test_sample_limit = 324
    do_validation = False
    use_gpu = True
    gpu_device = 'cuda:0'
    hidden_size = 256
    block_size = 64

    optimizer = 'adamw'
    learning_rate = 1e-5
    weight_decay = 0
    adam_epsilon = 1e-6
    warmup_ratio = 0.06

    from_epoch = 0
    epoch_num = 30
    real_epoch_num = 15
    output_step = 1
    save_global_step = 1600
    crop_documents = False
    crop_mention_option = 4
    entity_marker_type = 't'
    assert crop_mention_option in [0, 1, 2, 3, 4]
    assert entity_marker_type in ['mt', 'm', 't', 't-m', 'm*']
    test_step = 1
    model_path = 'checkpoint'
    model_name = 'ctd_all_denoise_biodre_men_base_star_tpmk4'
    model_class = 'DenoiseModel'
    fp16 = False
    lr_step_size = 1  # step_size
    lr_gamma = 1
    train_steps = 8
    dataset_multiplier = 4
    output_metric = 'binary_metric'
    kept_pair_num = None

    use_group_bilinear = True
    positive_num = 16
    negative_num = 15
    use_inter = True
    negative_lambda = 1.0

    loss_func = 'cross_entropy'
    assert loss_func in ['contrastive_mrl', 'contrastive_sml', 'adaptive_threshold', 'cross_entropy', 'log_exp']

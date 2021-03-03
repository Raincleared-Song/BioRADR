from .config_base import ConfigBase


class ConfigDenoise(ConfigBase):
    data_path = {
        'train': 'CTDRED/train_mixed.json',
        'valid': 'CTDRED/dev.json',
        'test': None  # determined by terminal args
    }
    batch_size = {
        'train': 4,
        'valid': 16,
        'test': 96
    }

    reader_num = 5
    bert_path = 'dmis-lab/biobert-base-cased-v1.1'
    score_path = None
    token_padding = 512  # token reserved for each document
    entity_padding = {
        'train': 42,
        'valid': 42,
        'test': 42
    }  # entity reserved for each document
    mention_padding = 3  # mention reserved for each entity
    train_sample_limit = 32
    test_sample_limit = 1600
    do_validation = True
    use_gpu = True
    hidden_size = 256

    optimizer = 'adamw'
    learning_rate = 3e-5
    weight_decay = 0

    epoch_num = 128
    output_step = 1
    test_step = 1
    model_path = 'checkpoint'
    model_name = 'denoise'
    fp16 = False
    lr_step_size = 1  # step_size
    lr_gamma = 1
    train_steps = 8
    output_metric = 'binary_metric'
    kept_pair_num = None

    positive_num = 16
    negative_num = 7

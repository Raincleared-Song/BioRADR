from .config_base import ConfigBase


class ConfigDenoise(ConfigBase):
    data_path = {
        'train': 'DocRED/train_annotated.json',
        'valid': 'DocRED/dev.json',
        'test': None  # determined by terminal args
    }
    batch_size = {
        'train': 4,
        'valid': 4,
        'test': 96
    }

    reader_num = 5
    bert_path = 'bert-base-uncased'
    score_path = None
    token_padding = 512  # token reserved for each document
    entity_padding = 42  # entity reserved for each document
    mention_padding = 3  # mention reserved for each entity
    train_sample_limit = 90
    test_sample_limit = 1800
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

    positive_num = 8
    negative_num = 7

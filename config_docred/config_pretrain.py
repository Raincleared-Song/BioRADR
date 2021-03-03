from .config_base import ConfigBase


class ConfigPretrain(ConfigBase):
    data_path = {
        'train': 'DocRED/train_distant.json',
        'valid': 'DocRED/dev.json',
        'test': None,
    }
    batch_size = {
        'train': 2,
        'valid': 2,
        'test': None
    }
    score_path = {
        'train': f'{ConfigBase.rank_result_path}/train_distant_score.npy',
        'valid': f'{ConfigBase.rank_result_path}/dev_score.npy',
        'test': None
    }
    title_path = {
        'train': f'{ConfigBase.rank_result_path}/train_distant_title.json',
        'valid': f'{ConfigBase.rank_result_path}/dev_title.json',
        'test': None
    }

    reader_num = 5
    bert_path = 'bert-base-uncased'  # can be modified by terminal args
    token_padding = 512  # token reserved for each document
    entity_padding = 42  # entity reserved for each document
    mention_padding = 3  # mention reserved for each entity
    train_sample_limit = 90
    test_sample_limit = 1800
    score_sample_limit = 60

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
    model_name = 'pretrain'
    fp16 = True
    lr_step_size = 1
    lr_gamma = 1
    train_steps = 8
    output_metric = 'binary_metric'
    kept_pair_num = 20

    positive_num = 5
    negative_num = 8

    data_ratio = 1.0
    same_pair_ratio = 0.6
    blank_ratio = 0.8

    mention_sample_num = 3
    mention_candidate_num = 5
    rd_sample_num = 3

    pretrain_tasks = 'MEM|MEM_X|RD|RD_X|RFA|RFA_X'

from .config_base import ConfigBase


class ConfigPretrain(ConfigBase):
    data_path = {
        'train': 'CTDRED/pretrain_mixed.json',
        'valid': 'CTDRED/dev.json',
        'test': None,
    }
    batch_size = {
        'train': 2,
        'valid': 8,
        'test': None
    }
    score_path = {
        'train': f'{ConfigBase.rank_result_path}/pretrain_mixed_score.npy',
        'valid': f'{ConfigBase.rank_result_path}/dev_score.npy',
        'test': None
    }
    title_path = {
        'train': f'{ConfigBase.rank_result_path}/pretrain_mixed_title.json',
        'valid': f'{ConfigBase.rank_result_path}/dev_title.json',
        'test': None
    }

    reader_num = 5
    bert_path = 'dmis-lab/biobert-base-cased-v1.1'  # can be modified by terminal args
    token_padding = 512  # token reserved for each document
    entity_padding = {
        'train': 24,
        'valid': 33,
        'test': None
    }  # entity reserved for each document
    mention_padding = 3  # mention reserved for each entity
    train_sample_limit = 32
    test_sample_limit = 1600
    score_sample_limit = 50

    do_validation = True
    use_gpu = True
    gpu_device = 'cuda:0'
    hidden_size = 256

    optimizer = 'adamw'
    learning_rate = 3e-5
    weight_decay = 0
    epoch_num = 128

    output_step = 1
    test_step = 1
    model_path = 'checkpoint'
    model_name = 'pretrain_t0_all'
    fp16 = True
    lr_step_size = 1
    lr_gamma = 1
    train_steps = 8
    output_metric = 'binary_metric'
    kept_pair_num = 20
    use_score = True
    score_threshold = 0.7

    positive_num = 5
    negative_num = 8

    data_ratio = 1.0
    same_pair_ratio = 0.6
    blank_ratio = 0.8

    mention_sample_num = 3
    mention_candidate_num = 5
    rd_sample_num = 3

    pretrain_tasks = 'MEM|MEM_X|RD|RD_X|RFA|RFA_X'

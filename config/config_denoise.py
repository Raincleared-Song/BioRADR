from .config_base import ConfigBase


class ConfigDenoise(ConfigBase):
    data_path = {
        'train': 'CTDRED/train_mixed_binary_pos.json',
        'valid': 'CTDRED/dev_binary_pos.json',
        'test': 'CTDRED/test_binary_pos.json'
    }
    # data_path = {
    #     'train': 'CTDRED/train_mixed.json',
    #     'valid': 'CTDRED/dev.json',
    #     'test': None  # determined by terminal args
    # }
    # data_path = {
    #     'train': 'Chemprot/chemprot_train_sent_fr.json',
    #     'valid': 'Chemprot/chemprot_dev_sent_fr.json',
    #     'test': None  # determined by terminal args
    # }
    batch_size = {
        'train': 4,
        'valid': 16,
        'test': 16
    }

    reader_num = 5
    bert_path = 'allenai/scibert_scivocab_cased'
    score_path = None
    token_padding = 1024  # token reserved for each document
    # token_padding = 448  # token reserved for each document, cps
    entity_padding = {
        'train': 37,
        'valid': 37,
        'test': 37
    }
    # entity_padding = {
    #     'train': 42,
    #     'valid': 42,
    #     'test': 42
    # }  # entity reserved for each document
    # entity_padding = {
    #     'train': 27,
    #     'valid': 27,
    #     'test': 27
    # }  # entity reserved for each document
    mention_padding = 3  # mention reserved for each entity
    # mention_padding = 1  # mention reserved for each entity, chemprot
    train_sample_limit = 32
    test_sample_limit = 324
    # train_sample_limit = 40
    # test_sample_limit = 162  # #chemical * #gene
    do_validation = True
    use_gpu = True
    gpu_device = 'cuda:0'
    hidden_size = 256
    block_size = 64

    optimizer = 'adamw'
    learning_rate = 3e-5
    weight_decay = 0
    adam_epsilon = 1e-6
    warmup_ratio = 0.06

    epoch_num = 120
    output_step = 1
    test_step = 1
    model_path = 'checkpoint'
    model_name = 'ctd_binary_denoise_n15_inter'
    fp16 = False
    lr_step_size = 1  # step_size
    lr_gamma = 1
    train_steps = 8
    output_metric = 'binary_metric'
    kept_pair_num = None
    use_type_marker = True

    positive_num = 16
    negative_num = 15
    use_inter = True

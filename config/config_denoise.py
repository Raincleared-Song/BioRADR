import os
from .config_base import ConfigBase


class ConfigDenoise(ConfigBase):
    data_path = {
        'train': '../project-1/CTDRED/ctd_train.json',
        'valid': '../project-1/CTDRED/ctd_dev.json',
        'test': '../project-1/CTDRED/ctd_test.json'
    }
    # data_path = {
    #     'train': '../project-1/CTDRED/train_mixed_binary_pos.json',
    #     'valid': '../project-1/CTDRED/dev_binary_pos.json',
    #     'test': '../project-1/CTDRED/test_binary_pos.json'
    # }
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

    reader_num = 4
    bert_path = '../huggingface/scibert_scivocab_cased' if os.path.exists(
        '../huggingface/scibert_scivocab_cased') else 'allenai/scibert_scivocab_cased'
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

    epoch_num = 30
    output_step = 1
    save_global_step = 800
    crop_documents = False
    crop_mention_option = 0
    entity_marker_type = 'mt'
    assert crop_mention_option in [0, 1, 2]
    assert entity_marker_type in ['mt', 'm', 't']
    test_step = 1
    model_path = 'checkpoint'
    model_name = 'ctd_all_celoss_p16_n15_denoise_ag'
    fp16 = False
    lr_step_size = 1  # step_size
    lr_gamma = 1
    train_steps = 8
    dataset_multiplier = 4
    output_metric = 'binary_metric'
    kept_pair_num = None

    positive_num = 16
    negative_num = 15
    use_inter = True
    negative_lambda = 1.0

    loss_func = 'cross_entropy'
    assert loss_func in ['contrastive_mrl', 'contrastive_sml', 'adaptive_threshold', 'cross_entropy', 'log_exp']

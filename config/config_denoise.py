import os
from .config_base import ConfigBase


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
    do_validation = True
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
    model_name = 'ctd_all_cotrain_bioradr'
    model_class = 'CoTrainModel'
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

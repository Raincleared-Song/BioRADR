import os
from .config_base import ConfigBase


class ConfigFineTune(ConfigBase):
    data_path = {
        'train': 'CTDRED/ctd_train.json',
        'valid': 'CTDRED/ctd_dev.json',
        'test': 'CTDRED/ctd_test.json'
    }
    batch_size = {
        'train': 4,
        'valid': 16,
        'test': 16
    }

    reader_num = 4
    if ConfigBase.model_type == 'bert':
        bert_path = '../huggingface/scibert_scivocab_cased' if os.path.exists(
            '../huggingface/scibert_scivocab_cased') else 'allenai/scibert_scivocab_cased'
    else:
        bert_path = 'decapoda-research/llama-7b-hf'
    token_padding = 512  # token reserved for each document
    entity_padding = None  # entity reserved for each document
    mention_padding = 3  # mention reserved for each entity
    train_sample_limit = 270  # CTD_binary
    train_sample_number = 60  # cdr_cdr_neg_sample
    test_sample_limit = 294  # CTD_binary
    score_sample_limit = 50
    kept_pair_num = 50
    do_validation = True
    use_gpu = True
    gpu_device = 'cuda:0'
    hidden_size = 256

    optimizer = 'adamw'
    learning_rate = 1e-4  ### ATTENTION! 2e-5 for cdr, 4e-4 group default for DocuNet
    weight_decay = 0
    adam_epsilon = 1e-6
    from_epoch = 0
    epoch_num = 30
    real_epoch_num = 1
    warmup_ratio = 0.06

    output_step = 1
    save_global_step = -1
    crop_documents = False
    crop_mention_option = 0
    entity_marker_type = 't'
    assert crop_mention_option in [0, 1, 2, 3, 4]
    assert entity_marker_type in ['mt', 'm', 't', 't-m', 'm*']
    test_step = 1
    model_path = 'checkpoint'
    model_name = 'ctd_pretrain_pre_sci_type_llama'
    model_class = 'FineTuneModel'
    fp16 = False
    lr_step_size = 1
    lr_gamma = 1
    train_steps = 1
    output_metric = 'f1_auc_metric'

    positive_num = None
    negative_num = None

    na_weight = 1  # loss weight of NA
    use_entity_type = False
    type_embed_size = 128
    use_loss_weight = False
    loss_weight = [1] * ConfigBase.relation_num
    use_stat = 'train_mixed'

    score_threshold = None
    test_score_threshold = None

    do_negative_sample = False  # if do negative samples

    use_group_bilinear = True
    use_logsumexp = False
    output_score_type = 'softmax'
    bilinear_block_size = 64
    only_chem_disease = True

    assert output_score_type in ['pos', 'sigmoid', 'softmax', 'diff', 'sig_diff']
    assert not ((score_threshold is None) ^ (test_score_threshold is None))

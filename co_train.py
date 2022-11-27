from config import ConfigFineTune, ConfigDenoise
from kernel.initialize import init_seed, init_args, init_data, save_config, init_models
from kernel.co_training import co_train


def main():
    init_seed(seed=None)
    args = init_args()
    ConfigFineTune.model_path = ConfigDenoise.model_path
    ConfigFineTune.model_name = ConfigDenoise.model_name
    assert args.task == 'cotrain' and args.mode == 'train'
    args.task = 'finetune'
    finetune_datasets = init_data(args)
    # denoise config is the main config
    ConfigDenoise.dataset_multiplier = len(finetune_datasets['train']) / \
        (100 * ConfigDenoise.batch_size['train'] * ConfigDenoise.train_steps)
    args.task = 'denoise'
    denoise_datasets = init_data(args)
    assert len(denoise_datasets['train']) == len(finetune_datasets['train'])
    save_config(args)
    models = init_models(args)
    co_train(ConfigFineTune, ConfigDenoise, models, finetune_datasets, denoise_datasets)


if __name__ == '__main__':
    main()

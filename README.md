# BioDSDocRE

Source code for BioDSDocRE (under development)

### Training denoise model

Parameter checkpoint_pkl_file is needed if you want to resume training from a checkpoint.

```shell
python3 main.py -t denoise -m train [-c checkpoint_pkl_file]
```

### Generate RD score file

```shell
python3 main.py -t denoise -m test -c pkl_file_used_to_rank -rf file_to_rank
```

### Pretrain

```shell
python3 main.py -t pretrain -m train [-c checkpoint_pkl_file]
```

### Finetune

If pretrain_model_path is not specified, 'dmis-lab/biobert-base-cased-v1.1' will be used by default.

```shell
python3 main.py -t finetune -m train [-c checkpoint_pkl_file] [-pb pretrain_model_path]
```


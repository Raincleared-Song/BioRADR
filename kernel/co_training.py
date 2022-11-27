import os
import torch
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from timeit import default_timer as timer
from apex import amp
from .testing import test
from models import CoTrainModel
from config import ConfigDenoise, ConfigFineTune
from transformers import get_linear_schedule_with_warmup
from utils import name_to_metric, print_value, time_to_str, save_model


def co_train(finetune_config: ConfigFineTune, denoise_config: ConfigDenoise,
             models, finetune_datasets, denoise_datasets):
    model: CoTrainModel = models['model']
    optimizer = models['optimizer']
    trained_epoch = models['trained_epoch']
    global_step = models['global_step']
    finetune_train_set = finetune_datasets['train']
    denoise_train_set = denoise_datasets['train']
    assert len(finetune_train_set) == len(denoise_train_set)
    train_size = len(finetune_train_set)

    real_epoch = denoise_config.real_epoch_num
    output_step = denoise_config.output_step
    test_step = denoise_config.test_step
    use_gpu = denoise_config.use_gpu
    gpu_device = denoise_config.gpu_device

    total_steps = int(train_size * denoise_config.epoch_num // denoise_config.train_steps)
    warmup_steps = int(total_steps * denoise_config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    os.makedirs(denoise_config.model_path, exist_ok=True)
    task_path = os.path.join(denoise_config.model_path, denoise_config.model_name)
    os.makedirs(task_path, exist_ok=True)
    model_output_path = os.path.join(task_path, 'model')
    os.makedirs(model_output_path, exist_ok=True)
    train_output_path = os.path.join(task_path, 'train')
    os.makedirs(train_output_path, exist_ok=True)
    valid_output_path = ''
    if denoise_config.do_validation:
        valid_output_path = os.path.join(task_path, 'valid')
        os.makedirs(valid_output_path, exist_ok=True)

    print(f'start training from epoch {trained_epoch + 1} to {real_epoch} ......')

    for epoch in range(trained_epoch + 1, real_epoch):
        # for each epoch
        start_time = timer()

        denoise_eval_res, finetune_eval_res = None, None
        total_loss = 0
        step = -1
        time_spent = 0
        metric_json = ''
        train_steps = denoise_config.train_steps
        determine = torch.use_deterministic_algorithms if 'use_deterministic_algorithms' in dir(torch) \
            else torch.set_deterministic

        for step, (finetune_data, denoise_data) in enumerate(zip(finetune_train_set, denoise_train_set)):
            for key, value in finetune_data.items():
                if isinstance(value, torch.Tensor):
                    finetune_data[key] = Variable(value.to(gpu_device)) if use_gpu else Variable(value)
            for key, value in denoise_data.items():
                if isinstance(value, torch.Tensor):
                    denoise_data[key] = Variable(value.to(gpu_device)) if use_gpu else Variable(value)

            model.cur_task = 'finetune'
            finetune_result = model(finetune_data, 'train', finetune_eval_res)
            finetune_loss, finetune_eval_res = finetune_result['loss'], finetune_result['eval_res']
            finetune_loss = finetune_loss.mean()

            model.cur_task = 'denoise'
            denoise_result = model(denoise_data, 'train', denoise_eval_res)  # forward
            denoise_loss, denoise_eval_res = denoise_result['loss'], denoise_result['eval_res']
            denoise_loss = denoise_loss.mean()

            loss = finetune_loss + denoise_loss
            total_loss += float(loss)
            loss /= train_steps

            if denoise_config.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                determine(False)
                loss.backward()
                determine(True)

            if step % train_steps == 0:
                if denoise_config.fp16:
                    clip_grad_norm_(amp.master_params(optimizer), 1.0)
                else:
                    clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if step % output_step == 0:
                denoise_metric_json = name_to_metric[denoise_config.output_metric](denoise_eval_res, 'train')
                finetune_metric_json = name_to_metric[finetune_config.output_metric](finetune_eval_res, 'train')
                metric_json = f'Rank: {denoise_metric_json} DocRE: {finetune_metric_json}'
                time_spent = timer() - start_time
                print_value(epoch, 'train', f'{step + 1}/{train_size}',
                            f'{time_to_str(time_spent)}/{time_to_str(time_spent*(train_size-step-1)/(step+1))}',
                            f'{(total_loss / (step + 1)):.3f}', metric_json,
                            os.path.join(train_output_path, f'{epoch}.txt'), '\r')
            global_step += 1
            if denoise_config.save_global_step > 0 and global_step % denoise_config.save_global_step == 0 \
                    and step + 1 != train_size:
                save_model(os.path.join(model_output_path, f'{epoch}-{global_step}.pkl'), model,
                           optimizer, epoch, global_step, denoise_config)

        print_value(epoch, 'train', f'{step + 1}/{train_size}',
                    f'{time_to_str(time_spent)}/{time_to_str(time_spent*(train_size-step-1)/(step+1))}',
                    f'{(total_loss / (step + 1)):.3f}', metric_json,
                    os.path.join(train_output_path, f'{epoch}.txt'))

        save_model(os.path.join(model_output_path, f'{epoch}.pkl'), model,
                   optimizer, epoch, global_step, denoise_config)

        if denoise_config.do_validation:
            assert len(valid_output_path) > 0
            if epoch % test_step == 0:
                with torch.no_grad():
                    # validation
                    model.cur_task = 'finetune'
                    test(model, finetune_datasets, 'valid', finetune_config, valid_output_path, epoch)
                    model.cur_task = 'denoise'
                    test(model, denoise_datasets, 'valid', denoise_config, valid_output_path, epoch)

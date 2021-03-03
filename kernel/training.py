import os
import torch
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from timeit import default_timer as timer
from apex import amp
from .testing import test
from config import ConfigBase
from torch.optim import lr_scheduler
from utils import name_to_metric, print_value, time_to_str, save_model


def train(config: ConfigBase, models, datasets):
    model = models['model']
    optimizer = models['optimizer']
    trained_epoch = models['trained_epoch']
    global_step = models['global_step']
    train_set = datasets['train']
    train_size = len(train_set)

    total_epoch = config.epoch_num
    output_step = config.output_step
    test_step = config.test_step
    use_gpu = config.use_gpu

    os.makedirs(config.model_path, exist_ok=True)
    task_path = os.path.join(config.model_path, config.model_name)
    os.makedirs(task_path, exist_ok=True)
    model_output_path = os.path.join(task_path, 'model')
    os.makedirs(model_output_path, exist_ok=True)
    train_output_path = os.path.join(task_path, 'train')
    os.makedirs(train_output_path, exist_ok=True)
    valid_output_path = ''
    if config.do_validation:
        valid_output_path = os.path.join(task_path, 'valid')
        os.makedirs(valid_output_path, exist_ok=True)

    lr_step_size = config.lr_step_size
    gamma = config.lr_gamma
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma)
    exp_lr_scheduler.step(trained_epoch + 1)

    print('start training ......')

    for epoch in range(trained_epoch + 1, total_epoch):
        # for each epoch
        start_time = timer()
        exp_lr_scheduler.step(epoch)

        eval_res = None
        total_loss = 0
        step = -1
        time_spent = 0
        metric_json = ''
        train_steps = config.train_steps

        assert len(train_set) > 0

        for step, data in enumerate(train_set):
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    data[key] = Variable(value.cuda()) if use_gpu else Variable(value)

            result = model(data, 'train', eval_res)  # forward

            loss, eval_res = result['loss'], result['eval_res']
            loss = loss.mean()
            total_loss += float(loss)
            loss /= train_steps

            if config.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if step % train_steps == 0:
                if config.fp16:
                    clip_grad_norm_(amp.master_params(optimizer), 1.0)
                else:
                    clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % output_step == 0:
                metric_json = name_to_metric[config.output_metric](eval_res, 'train')
                time_spent = timer() - start_time
                print_value(epoch, 'train', f'{step + 1}/{train_size}',
                            f'{time_to_str(time_spent)}/{time_to_str(time_spent*(train_size-step-1)/(step+1))}',
                            f'{(total_loss / (step + 1)):.3f}', metric_json,
                            os.path.join(train_output_path, f'{epoch}.txt'), '\r')
            global_step += 1

        print_value(epoch, 'train', f'{step + 1}/{train_size}',
                    f'{time_to_str(time_spent)}/{time_to_str(time_spent*(train_size-step-1)/(step+1))}',
                    f'{(total_loss / (step + 1)):.3f}', metric_json,
                    os.path.join(train_output_path, f'{epoch}.txt'))

        if config.model_name == 'pretrain':  # pretraining
            pretrain_path = os.path.join(model_output_path, f'epoch_{epoch}')
            os.makedirs(pretrain_path, exist_ok=True)
            model.save(pretrain_path)

        save_model(os.path.join(model_output_path, f'{epoch}.pkl'), model,
                   optimizer, epoch, global_step, config)

        if config.do_validation:
            assert len(valid_output_path) > 0
            if epoch % test_step == 0:
                with torch.no_grad():
                    # validation
                    test(model, datasets, 'valid', config, valid_output_path, epoch)

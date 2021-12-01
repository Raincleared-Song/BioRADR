import os
import torch
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from timeit import default_timer as timer
from apex import amp
from .testing import test
from config import ConfigBase
from transformers import get_linear_schedule_with_warmup
from utils import name_to_metric, print_value, time_to_str, save_model, time_tag


def train(config: ConfigBase, models, datasets, it=None):
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

    total_steps = int(len(train_set) * config.epoch_num // config.train_steps)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    os.makedirs(config.model_path, exist_ok=True)
    task_path = os.path.join(config.model_path, config.model_name)
    os.makedirs(task_path, exist_ok=True)
    model_output_path = os.path.join(task_path, 'model')
    os.makedirs(model_output_path, exist_ok=True)
    train_output_path = os.path.join(task_path, 'train')
    os.makedirs(train_output_path, exist_ok=True)
    valid_output_path = ''
    if config.do_validation:
        valid_output_path = os.path.join(task_path, 'valid' + ('' if it is None else str(it)))
        os.makedirs(valid_output_path, exist_ok=True)

    # lr_step_size = config.lr_step_size
    # gamma = config.lr_gamma
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma)
    # exp_lr_scheduler.step(trained_epoch + 1)

    print(f'start training from epoch {trained_epoch + 1} to {total_epoch} ......')

    # metric_out = open('tmp_tmp.txt', 'w', encoding='utf-8')
    # set_file(metric_out)
    # set_metric()
    # clear_count()

    # epoch = trained_epoch
    # if config.do_validation:
    #     assert len(valid_output_path) > 0
    #     if epoch % test_step == 0:
    #         with torch.no_grad():
    #             # validation
    #             test(model, datasets, 'valid', config, valid_output_path, epoch)
    # return

    for epoch in range(trained_epoch + 1, total_epoch):
        # for each epoch
        start_time = timer()
        # exp_lr_scheduler.step(epoch)

        eval_res = None
        total_loss = 0
        step = -1
        time_spent = 0
        metric_json = ''
        train_steps = config.train_steps
        determine = torch.use_deterministic_algorithms if torch.__version__ == '1.10.0' else torch.set_deterministic

        assert len(train_set) > 0

        for step, data in enumerate(train_set):
            time_tag(0, True)
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    data[key] = Variable(value.to(config.gpu_device)) if use_gpu else Variable(value)

            time_tag(1, True)
            result = model(data, 'train', eval_res)  # forward
            time_tag(11, True)

            loss, eval_res = result['loss'], result['eval_res']
            loss = loss.mean()
            total_loss += float(loss)
            loss /= train_steps
            time_tag(12, True)

            if config.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                determine(False)
                loss.backward()
                determine(True)
            time_tag(13, True)

            if step % train_steps == 0:
                if config.fp16:
                    clip_grad_norm_(amp.master_params(optimizer), 1.0)
                else:
                    clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                time_tag(14, True)

            if step % output_step == 0:
                metric_json = name_to_metric[config.output_metric](eval_res, 'train')
                time_spent = timer() - start_time
                print_value(epoch, 'train', f'{step + 1}/{train_size}',
                            f'{time_to_str(time_spent)}/{time_to_str(time_spent*(train_size-step-1)/(step+1))}',
                            f'{(total_loss / (step + 1)):.3f}', metric_json,
                            os.path.join(train_output_path, f'{epoch}.txt'), '\r')
                time_tag(15, True)
            global_step += 1
            # if step == 20:
            #     print_time_stat()
            #     unset_metric()

        print_value(epoch, 'train', f'{step + 1}/{train_size}',
                    f'{time_to_str(time_spent)}/{time_to_str(time_spent*(train_size-step-1)/(step+1))}',
                    f'{(total_loss / (step + 1)):.3f}', metric_json,
                    os.path.join(train_output_path, f'{epoch}.txt'))

        save_model(os.path.join(model_output_path, f'{epoch}.pkl'), model,
                   optimizer, epoch, global_step, config)

        if config.model_name.startswith('pretrain'):  # pretraining
            pretrain_path = os.path.join(model_output_path, f'epoch_{epoch}')
            os.makedirs(pretrain_path, exist_ok=True)
            model.save(pretrain_path)
            # delete the prev model
            prev_path = os.path.join(model_output_path, f'{epoch - 1}.pkl')
            if os.path.exists(prev_path):
                os.remove(prev_path)

        if config.do_validation:
            assert len(valid_output_path) > 0
            if epoch % test_step == 0:
                with torch.no_grad():
                    # validation
                    test(model, datasets, 'valid', config, valid_output_path, epoch)
    # unset_file()
    # metric_out.close()

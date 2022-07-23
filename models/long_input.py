import torch
import numpy as np
import torch.nn.functional as F


def process_long_input(model, input_ids, attn_mask, config, ret_attn=False):
    """处理较长输入, 句子长度的上限是 1024"""
    n, c = input_ids.size()
    if c <= 512:
        output = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
    else:
        start_tokens = torch.tensor([config.tokenizer.cls_token_id]).to(input_ids)
        end_tokens = torch.tensor([config.tokenizer.sep_token_id]).to(input_ids)
        len_start = start_tokens.size(0)
        len_end = end_tokens.size(0)
        new_input_ids, new_attn_mask, num_seg = [], [], []
        seq_len = attn_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
        for i, l_i in enumerate(seq_len):
            if l_i <= 512:
                new_input_ids.append(input_ids[i, :512])
                new_attn_mask.append(attn_mask[i, :512])
                num_seg.append(1)
            else:
                input_ids1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)
                input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)
                attention_mask1 = attn_mask[i, :512]
                attention_mask2 = attn_mask[i, (l_i - 512): l_i]
                new_input_ids.extend([input_ids1, input_ids2])
                new_attn_mask.extend([attention_mask1, attention_mask2])
                num_seg.append(2)
        input_ids = torch.stack(new_input_ids, dim=0)
        attn_mask = torch.stack(new_attn_mask, dim=0)
        output = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
        idx = 0
        new_output, new_attention = [], []
        for (n_s, l_i) in zip(num_seg, seq_len):
            if n_s == 1:
                output = F.pad(sequence_output[idx], (0, 0, 0, c - 512))
                att = F.pad(attention[idx], (0, c - 512, 0, c - 512))
                new_output.append(output)
                new_attention.append(att)
            elif n_s == 2:
                output1 = sequence_output[idx][:512 - len_end]
                mask1 = attn_mask[idx][:512 - len_end]
                att1 = attention[idx][:, :512 - len_end, :512 - len_end]
                output1 = F.pad(output1, (0, 0, 0, c - 512 + len_end))
                mask1 = F.pad(mask1, (0, c - 512 + len_end))
                att1 = F.pad(att1, (0, c - 512 + len_end, 0, c - 512 + len_end))

                output2 = sequence_output[idx + 1][len_start:]
                mask2 = attn_mask[idx + 1][len_start:]
                att2 = attention[idx + 1][:, len_start:, len_start:]
                output2 = F.pad(output2, (0, 0, l_i - 512 + len_start, c - l_i))
                mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                att2 = F.pad(att2, [l_i - 512 + len_start, c - l_i, l_i - 512 + len_start, c - l_i])
                mask = mask1 + mask2 + 1e-10
                # average pooling
                output = (output1 + output2) / mask.unsqueeze(-1)
                att = (att1 + att2)
                att = att / (att.sum(-1, keepdim=True) + 1e-10)
                new_output.append(output)
                new_attention.append(att)
            idx += n_s
        sequence_output = torch.stack(new_output, dim=0)
        attention = torch.stack(new_attention, dim=0)
    return (sequence_output, attention) if ret_attn else sequence_output

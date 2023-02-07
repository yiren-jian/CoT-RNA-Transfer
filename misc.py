###### some helper functions
import os
import subprocess
import numpy as np
import torch
import math
import pickle

def get_a3m_feat(path):
    with open(path) as fp:
        line = fp.readline()
        if line.startswith(">"):
            line = fp.readline()
        L = len(line.strip())
    program = [
        os.path.join(os.path.dirname(__file__), "bin/a3m_to_feat"),
        "--input",
        path,
        "--max_gap",
        "7",
        "--max_keep",
        "5000",
        "--sample_ratio",
        "1.0",
    ]
    process = subprocess.run(program, capture_output=True)
    assert process.returncode == 0, "Invalid A3M file"
    x = np.copy(np.frombuffer(process.stdout, dtype=np.int8))
    x = x.reshape((-1, L, 7 * 2 + 3)).transpose((0, 2, 1))
    assert (x < 23).all(), "Internal error"
    seq = x[0][0]
    return {
        "seq": torch.tensor(seq).long()[None].cuda(),
        "msa": torch.tensor(x).long()[None].cuda(),
        "index": torch.arange(seq.shape[0]).long()[None].cuda(),
    }


def write_site_pair_score_data_to_file(sorted_data_list, output_file_path, algorithm_used, max_iterations=None, num_threads=None):
    """Since site indices are starting from zero within python we add one to
    each of them when they are being written to output file.
    """
    formater = '#' + '='*100
    formater += '\n'
    with open(output_file_path, 'w') as fh:
        fh.write(formater)
        fh.write('# This result is computed using {}\n'.format(algorithm_used))
        if max_iterations is not None:
            fh.write('# maximum number of gradient decent iterations: {}\n'.format(max_iterations))
        if  num_threads is not None:
            fh.write('# Number of threads used: {}\n'.format(num_threads))
        fh.write('# The first and second columns are site pairs. The third column represents interaction score\n')
        fh.write(formater)

        for site_pair, score in sorted_data_list:
            i, j = site_pair[0] + 1, site_pair[1] + 1
            fh.write('{}\t{}\t{}\n'.format(i, j, score))
    return None


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Source: https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/optimization.py#L75
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    Source: https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/optimization.py#L104
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def distance_to_37(v):
    if v <= 2.5:
        lbl = 0
    elif v <= 3.0:
        lbl = 1
    elif v <= 3.5:
        lbl = 2
    elif v <= 4.0:
        lbl = 3
    elif v <= 4.5:
        lbl = 4
    elif v <= 5.0:
        lbl = 5
    elif v <= 5.5:
        lbl = 6
    elif v <= 6.0:
        lbl = 7
    elif v <= 6.5:
        lbl = 8
    elif v <= 7.0:
        lbl = 9
    elif v <= 7.5:
        lbl = 10
    elif v <= 8.0:
        lbl = 11
    elif v <= 8.5:
        lbl = 12
    elif v <= 9.0:
        lbl = 13
    elif v <= 9.5:
        lbl = 14
    elif v <= 10.0:
        lbl = 15
    elif v <= 10.5:
        lbl = 16
    elif v <= 11.0:
        lbl = 17
    elif v <= 11.5:
        lbl = 18
    elif v <= 12.0:
        lbl = 19
    elif v <= 12.5:
        lbl = 20
    elif v <= 13.0:
        lbl = 21
    elif v <= 13.5:
        lbl = 22
    elif v <= 14.0:
        lbl = 23
    elif v <= 14.5:
        lbl = 24
    elif v <= 15.0:
        lbl = 25
    elif v <= 15.5:
        lbl = 26
    elif v <= 16.0:
        lbl = 27
    elif v <= 16.5:
        lbl = 28
    elif v <= 17.0:
        lbl = 29
    elif v <= 17.5:
        lbl = 30
    elif v <= 18.0:
        lbl = 31
    elif v <= 18.5:
        lbl = 32
    elif v <= 19.0:
        lbl = 33
    elif v <= 19.5:
        lbl = 34
    elif v <= 20.0:
        lbl = 35
    else:
        lbl = 36

    return lbl


def distance_to_2(v):
    if v <= 10.0:
        lbl = 0
    else:
        lbl = 1

    return lbl

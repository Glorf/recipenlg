"""
GPT-2 Finetuning based on https://github.com/allenai/tpu_pretrain/blob/master/pretrain.py and https://github.com/pytorch/xla/blob/master/test/test_train_mp_mnist.py
"""

import argparse
import glob
import logging
import os
import random
import gc
import h5py
import boto3
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from transformers import AdamW, GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from multiprocessing import Lock

logger = logging.getLogger(__name__)

import tarfile
def tardir(path, tar_name):
    with tarfile.open(tar_name, "w") as tar_handle:
        for root, dirs, files in os.walk(path):
            for file in files:
                tar_handle.add(os.path.join(root, file))

class TextDataset(Dataset):
    def __init__(self, file_path='train'):
        cached_features_file = "unsupervised.h5"

        logger.info("Loading features from cached file %s", cached_features_file)
        self.file = h5py.File(cached_features_file, 'r')
        self.examples = file_path
        #self.all = self.file[self.examples][:]

    def __len__(self):
        return self.file[self.examples].shape[0]

    def __getitem__(self, item):
        #return torch.tensor(self.all[item])
        return torch.tensor(self.file[self.examples][item])

import sys
import fcntl
import time

def save_model(model, output_dir):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.config.save_pretrained(output_dir)
    state_dict = model.state_dict()
    for t_name in state_dict:
       t_val = state_dict[t_name]
       state_dict[t_name] = t_val.to('cpu')

    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(state_dict, output_model_file)


def _train_update(device, step, loss, tracker, scheduler, writer):
    print(step)
    writer.add_scalar("Training rate", tracker.rate(), step)
    writer.add_scalar("Loss/train", loss.item(), step)
    writer.add_scalar("Learning rate", scheduler.get_last_lr()[0], step)

def evaluate(model, test_dataloader, device):
    para_loader = pl.ParallelLoader(test_dataloader, [device])
    model.eval()
    eval_loss = 0.0
    eval_steps = 0
    for t_step, t_batch in enumerate(para_loader.per_device_loader(device)):
        inputs, labels = (t_batch, t_batch)
        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        eval_steps += 1

    eval_loss = eval_loss / eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    return perplexity

def tpu_training_loop(index):
    torch.set_default_tensor_type('torch.FloatTensor')
    #To decrease exploing RAM usage, only load and transfer one model at time
    lock_file = "tpu.lock"
    fd = open(lock_file, "w")
    fcntl.lockf(fd, fcntl.LOCK_EX)

    model_class = GPT2LMHeadModel


    model = model_class.from_pretrained("gpt2")
    tokenizer_class = GPT2Tokenizer
    tokenizer = tokenizer_class.from_pretrained("gpt2", do_lower_case=False)

    device = xm.xla_device()

    logger_is_me = False
    if xm.is_master_ordinal():
        logger_is_me = True
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()

    special_tokens = {
        "additional_special_tokens": [
            "<TITLE_START>",
            "<TITLE_END>",
            "<INSTR_START>",
            "<NEXT_INSTR>",
            "<INSTR_END>",
            "<INGR_START>",
            "<NEXT_INGR>",
            "<INGR_END>",
            "<RECIPE_START>",
            "<RECIPE_END>",
            "<INPUT_START>",
            "<INPUT_END>",
            "<NEXT_INPUT>"
        ]
    }

    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = TextDataset(file_path="train")
    test_dataset = TextDataset(file_path="test")

    train_sampler = DistributedSampler(
          train_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=True)
    test_sampler = DistributedSampler(
          test_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=False)

    #PARAMS!!
    train_batch_size = 4
    test_batch_size = 4

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=test_batch_size)

    model.train().to(device)

    import gc
    gc.collect()

    fcntl.lockf(fd, fcntl.LOCK_UN)

    gradient_steps = 1
    epochs = 1
    t_total = len(train_dataloader) // gradient_steps

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # one optimizer and scheduler per TPU core. Both objects are saved in `context` to be reused the next epoch
    lr = 5e-5 * xm.xrt_world_size()
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

    tracker = xm.RateTracker()

    # PARAMS V2!!!
    gradient_steps = 1
    logging_steps = 100
    validation_steps = 1000

    optimizer.zero_grad()

    def single_epoch(big_step, epoch):
        train_sampler.set_epoch(epoch)
        para_loader = pl.ParallelLoader(train_dataloader, [device])
        for step, batch in enumerate(para_loader.per_device_loader(device)):
            inputs, labels = (batch, batch)
            model.train()
            outputs = model(inputs, labels=labels)
            loss = outputs[0]

            loss = loss / gradient_steps

            loss.backward()
            tracker.add(1)

            if (step + 1) % gradient_steps == 0:
                xm.optimizer_step(optimizer)
                scheduler.step()
                optimizer.zero_grad()
                big_step += 1

                if logger_is_me and (big_step + 1) % logging_steps == 0:
                    xm.add_step_closure(_train_update, args=(device, big_step, loss, tracker, scheduler, writer))

                if (big_step + 1) % validation_steps == 0:
                    perplexity = evaluate(model, test_dataloader, device)
                    if logger_is_me:
                        print("Validation loss: ", perplexity)
                        writer.add_scalar("Validation loss", perplexity, big_step)
        return big_step
    big_step = 0
    #Always pretend to have one more epoch to do, otherwise model won't get saved
    for i in range(1, 6):
        print("Epoch: "+str(i))
        big_step = single_epoch(big_step, i)
        if logger_is_me:
            output_dir = "gpt2-refined-epoch-"+str(i)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            save_model(model, output_dir)
            tokenizer.save_pretrained(output_dir)
            print("Model saved")

def main():
#    print(os.environ["XLA_USE_BF16"])
    xmp.spawn(tpu_training_loop, args=())

if __name__ == "__main__":
    main()

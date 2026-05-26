import argparse
import random
from copy import deepcopy
import os
import sys
import glob
import time
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))
from image_classification.models import resnet50
from image_classification.utils import *

def fast_collate(memory_format, batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(
        memory_format=memory_format
    )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array.copy())
    return tensor, targets

def get_pytorch_val_loader(
    data_path,
    image_size,
    batch_size,
    interpolation="bilinear",
    workers=5,
    crop_padding=32,
    memory_format=torch.contiguous_format,
    prefetch_factor=2,
    rank=-1,
):
    interpolation = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR}[
        interpolation
    ]
    valdir = os.path.join(data_path, "val")
    
    transforms_list = [
                transforms.Resize(image_size + crop_padding, interpolation=interpolation),
                transforms.CenterCrop(image_size),
    ]
    
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(transforms_list),
    )

    if dist.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False)
    else:
        val_sampler = None
        
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=batch_size,
        shuffle=(val_sampler is None),
        num_workers=workers,
        worker_init_fn=None,
        pin_memory=True,
        collate_fn=partial(fast_collate, torch.contiguous_format),
        drop_last=False,
        persistent_workers=True if workers > 0 else False,
        prefetch_factor=prefetch_factor if workers > 0 else 2,
    )
    return val_loader

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('Boolean value expected.')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='/data/application/common/imagenet', help='images path')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--target', default='cuda', help='sdaa or cpu or cuda')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--weight-int8', action='store_true', help='INT8 weight loading')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    

    local_rank = int(os.getenv("LOCAL_RANK", -1))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    
    is_distributed = local_rank != -1
    
    if is_distributed:
        if opt.target == 'cuda':
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
            print(f"Initialized process group: rank {local_rank}, world_size {world_size}")
        elif opt.target == 'sdaa':
            torch.sdaa.set_device(local_rank)
            dist.init_process_group(backend='tccl', init_method='env://')
            print(f"Initialized process group: rank {local_rank}, world_size {world_size}")



    torch.manual_seed(1234)
    np.random.seed(seed=1234)
    random.seed(1234)
    
    device = opt.target
    

    if opt.target == 'cuda' and is_distributed:
        device = f'cuda:{local_rank}'
    elif opt.target == 'sdaa' and is_distributed:
        device = f'sdaa:{local_rank}'

    if opt.target=='cuda':
        from torch.cuda.amp import autocast
    else:
        from torch_sdaa.amp import autocast

    val_loader = get_pytorch_val_loader(
        data_path=opt.data_path,
        image_size=224,
        batch_size=opt.batch_size,
        rank=local_rank
    )
    
    total_samples = len(val_loader.dataset)
    if is_distributed:
        pass

    model = resnet50(pretrained=True).to(device)

    if opt.half:
        model = model.half()
        print("***** use FP16 half-precision inference *****")

    mean = (
                torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).to(device)
                .view(1, 3, 1, 1)
            )
    std = (
                torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).to(device)
                .view(1, 3, 1, 1)
            )

    model.eval()
    if opt.weight_int8:
        # print("⚠️ INT8 weight conversion (scale -> clamp -> dequantize)")
        compute_dtype = torch.float16 if opt.half else torch.float32
        with torch.no_grad():
            for m in model.modules():
                if hasattr(m, 'weight') and m.weight is not None:
                    w_fp = m.weight.data.float()
                    max_val = w_fp.abs().max()
                    if max_val > 1e-6:
                        scale = 127.0 / max_val
                        w_int8 = (w_fp * scale).clamp(-127, 127).to(torch.int8)
                        m.weight.data = w_int8.to(compute_dtype) / scale
                    else:
                        m.weight.data = torch.zeros_like(m.weight.data, dtype=compute_dtype)
        print("✅ INT8 weight applied. Compute stays FP16/FP32.")

    acc_list = []
    infer_time_list = []
    e2e_list = []
    sps_list = []
    e2e_start = time.time()
    
    local_processed_samples = 0

    data_iter = enumerate(val_loader)

    for i, (input, target) in data_iter:
        input, target = input.to(device), target.to(device)
        input = input.float()

        input = input.sub_(mean).div_(std)
        
        bs = input.size(0)
        local_processed_samples += bs
        
        input = input.to(memory_format=torch.channels_last)
        if opt.half:
            input = input.half()
        
        # infer start
        infer_start = time.time()
        with torch.no_grad():
            if autocast and opt.half:
                with autocast():
                    output = model(input)
            else:
                output = model(input)
        
        # infer end
        infer_time = time.time() - infer_start
        infer_sps = bs / infer_time
        sps_list.append(infer_sps)
        
        with torch.no_grad():
            precs = accuracy(output.data, target, topk=(1, 1))
        
        precs = map(lambda t: t.item(), precs)
        infer_result = {f"top{k}": (p, bs) for k, p in zip((1, 1), precs)}
       
        acc_list.append(infer_result["top1"][0])
        
        print('acc:',sum(acc_list)/len(acc_list))

        e2e_time = time.time() - e2e_start
        e2e_list.append(e2e_time)
        infer_time_list.append(infer_time)
        e2e_start = time.time()
    
    local_correct = 0.0
    local_avg_acc = sum(acc_list) / len(acc_list) if len(acc_list) > 0 else 0
    local_total_batches = len(acc_list)
    
    acc_tensor = torch.tensor([local_avg_acc, float(local_processed_samples)]).to(device)
    
    if is_distributed:
        dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
        all_acc_lists = [None for _ in range(world_size)]
        dist.all_gather_object(all_acc_lists, acc_list)
        
        if local_rank == 0:
            final_acc_list = []
            for l in all_acc_lists:
                final_acc_list.extend(l)
            
            final_acc = sum(final_acc_list) / len(final_acc_list)
            print(f'\nFinal Evaluation Accuracy: {final_acc:.4f}')
        else:
            final_acc = None
            
    else:
        final_acc = sum(acc_list) / len(acc_list)
        print(f'\nFinal Evaluation Accuracy: {final_acc:.4f}')

    # avg_e2e = sum(e2e_list) / len(e2e_list)
    avg_infer_time = sum(infer_time_list) / len(infer_time_list)
    total_infer_time = sum(infer_time_list)
    avg_sps = sum(sps_list) / len(sps_list)

    if is_distributed:
        total_sps = avg_sps * world_size
    else:
        total_sps = avg_sps

    if local_rank in [-1, 0]:
        print(f'Total Infer Time: {total_infer_time:.4f}s')
        print(f'Avg Infer Time: {avg_infer_time:.4f}s')
        print(f'Total System Infer SPS: {total_sps:.4f}')
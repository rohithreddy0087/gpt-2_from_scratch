
import os
import time
import math
import tiktoken
import torch
import torch.nn.functional as F

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from model import GPT
from config_parser import get_config
from dataset import DataLoader
from hellaswag import render_example, iterate_examples, get_most_likely_row

#torchrun --standalone --nproc_per_node=4 train.py 

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
torch.set_float32_matmul_precision('high')
if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
else:
    compute_dtype = torch.float32

def get_lr(iter):
    if iter < config.warmup_steps:
        return config.max_lr * (iter+1)/config.warmup_steps
    if iter > config.max_steps:
        return config.min_lr
    
    decay_ratio = (iter - config.warmup_steps)/ (config.max_steps - config.warmup_steps)
    assert 0 <= decay_ratio <=1
    cosine_lr = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + cosine_lr*(config.max_lr - config.min_lr)

def save_model():
    output = math.ceil(step / 5000) * 5000
    checkpoint_path = os.path.join(config.saved_weights_path, f"model_{output:05d}.pth")
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': raw_model.config,
        'step': step,
        'val_loss': val_loss.item()
    }
    torch.save(checkpoint, checkpoint_path)
                
def train():
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for accm_step in range(grad_accum_steps):
        inputs, labels = train_dataloader.next_batch()
        inputs, labels = inputs.to(config.device), labels.to(config.device)
        with torch.autocast(device_type = config.device, dtype=compute_dtype):
            logits = model(inputs)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            labels = labels.view(B*T)
            loss = F.cross_entropy(logits, labels) 
        loss = loss/ grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            # In DDP, for each GPU once we compute the gradients, the gradients are averaged using an ALLREDUCE and then the gradients are accumulated
            # but we have to wait till the last accumulated step to that sync for all the GPUs, hence the following.
            # So at the last step, all the GPUs will have the same averaged gradient
            model.require_backward_grad_sync = (accm_step == grad_accum_steps - 1)
        loss.backward()
    
    # We also need to average the loss_accum over all the GPUs
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)    
    
    # calculates the global L2 norm of all the parameters in the model and clip it to make sure its length is less than 1 
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    
    return loss_accum, lr, norm

def eval():
    model.eval()
    val_dataloader.reset()
    with torch.no_grad():
        loss_accum = 0.0
        for accm_step in range(grad_accum_steps):
            inputs, labels = val_dataloader.next_batch()
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            with torch.autocast(device_type = config.device, dtype=compute_dtype):
                logits = model(inputs)
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                labels = labels.view(B*T)
                loss = F.cross_entropy(logits, labels) 
            loss = loss/ grad_accum_steps
            loss_accum += loss.detach()        
        # We also need to average the loss_accum over all the GPUs
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)    
    return loss_accum

def generate():
    model.eval()
    tokens = encoder.encode(config.prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(config.num_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)
    while xgen.size(1) < config.max_length:
        with torch.no_grad():
            with torch.autocast(device_type=config.device, dtype=torch.float16):
                logits = model(xgen) 
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, config.topk, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) 
            xcol = torch.gather(topk_indices, -1, ix) 
            xgen = torch.cat((xgen, xcol), dim=1)
    for i in range(config.num_sequences):
        tokens = xgen[i, :config.max_length].tolist()
        decoded = encoder.decode(tokens)
        config.logger.info(f"rank {ddp_rank} sample {i}: {decoded}")
        
def evaluate():
    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
        # only process examples where i % ddp_world_size == ddp_rank
        if i % ddp_world_size != ddp_rank:
            continue
        # render the example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        # get the logits
        with torch.no_grad():
            with torch.autocast(device_type=config.device, dtype=torch.float16):
                logits = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
    # reduce the stats across all processes
    if ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total
    return num_correct_norm, num_total, acc_norm
    

if __name__ == '__main__':
    config_path = '/root/data/rrr/usr/gpt-2/config.ini'
    config = get_config(config_path)
    encoder = tiktoken.get_encoding("gpt2")
    
    assert config.total_batch_size % (config.batch_size * config. context_size * ddp_world_size) == 0, "make sure total_batch_size is divisible by B*T*ddp"
    grad_accum_steps = config.total_batch_size // (config.batch_size * config. context_size * ddp_world_size)
    if master_process:
        config.logger.debug("Grad accumulated steps are %s", grad_accum_steps)
            
    train_dataloader = DataLoader(config, process_rank=ddp_rank, num_process=ddp_world_size, split='train')
    val_dataloader = DataLoader(config, process_rank=ddp_rank, num_process=ddp_world_size, split='val')
    
    model = GPT(config).to(config.device)
    if config.use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        
    raw_model = model.module if ddp else model
    optimizer = raw_model.configure_optimizer() # uses fused adam, kernel fusion makes it faster
        
    for step in range(config.max_steps):
        t0 = time.time()
        last_step = (step == config.max_steps - 1)

        if ((step > 0 and step % config.eval_steps == 0) or last_step):
            val_loss = eval()
            if master_process:
                config.logger.debug(f"Val Step: {int(step/config.eval_steps)} | Val Loss: {val_loss.item():.6f}")
                if config.wandb_ is not None:
                    metrics = {"val/loss": val_loss.item()}
                    config.wandb_.log(metrics)
                if step > 0 and (step % config.save_weights_freq == 0 or last_step):
                    save_model()
                    
        if ((step > 0 and step % config.eval_steps == 0) or last_step) and (not config.use_compile):
            generate()
            num_correct_norm, num_total, acc_norm = evaluate()
            if master_process:
                config.logger.debug(f"HellaSwag Step: {int(step/config.eval_steps)} | HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                if config.wandb_ is not None:
                    metrics = {"HellaSwag/accuracy": acc_norm}
                    config.wandb_.log(metrics)
                    
                    
        loss_accum, lr, norm = train()
        t1 = time.time()
    
        time_taken = (t1-t0)*1000
        tokens_per_second = (train_dataloader.total_size * grad_accum_steps * ddp_world_size)/(t1-t0)
        if master_process:
            config.logger.debug(f"Step: {step} | Loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | Time Taken: {time_taken:.2f}ms | tokens/second: {tokens_per_second:.2f} | curr_pos: {train_dataloader.current_pos}")
            if config.wandb_ is not None:
                metrics = {"train/loss": loss_accum.item(), 
                           "train/lr": lr,
                           "train/norm": norm,
                           "train/time_taken": time_taken,
                           "train/tokens_per_seconds": tokens_per_second}
                config.wandb_.log(metrics)
        
    if ddp:
        destroy_process_group()
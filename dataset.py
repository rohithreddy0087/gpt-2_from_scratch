import os
import numpy as np
import tiktoken
import torch

class DataLoader:
    def __init__(self, config, process_rank, num_process, split):
        self.batch_size = config.batch_size
        self.context_size = config.context_size
        self.total_size = self.batch_size*self.context_size
        self.process_rank = process_rank
        self.num_process = num_process
        assert split in {'train', 'val'}, "split is incorrect"
        dataset_path = os.path.join(config.dataset_path, f"{split}.bin")
        self.tokens = np.memmap(dataset_path, dtype=np.uint16, mode='r')
            
        if process_rank == 0:
            config.logger.debug("Number of tokens loaded in %s dataset is %s", split, len(self.tokens))
            config.logger.debug("Number of batches in 1 epoch of %s dataset is %s", split, len(self.tokens)//(self.total_size * self.num_process))
        
        self.current_pos = self.total_size*self.process_rank
        
    def reset(self):
        self.current_pos = self.total_size*self.process_rank
        
    def next_batch(self):
        
        tokens_curr = torch.from_numpy(self.tokens[self.current_pos:self.current_pos+self.total_size+1].astype(np.int64))
        inputs = tokens_curr[:-1].view(self.batch_size, self.context_size)
        labels = tokens_curr[1:].view(self.batch_size, self.context_size)
        
        self.current_pos = self.current_pos+self.total_size*self.num_process
        
        if self.current_pos+self.total_size*self.num_process+1 > len(self.tokens):
            self.current_pos = self.total_size*self.process_rank
        
        return inputs, labels

class ToyDataLoader:
    def __init__(self, config, process_rank, num_process):
        self.batch_size = config.batch_size
        self.context_size = config.context_size
        self.total_size = self.batch_size*self.context_size
        self.process_rank = process_rank
        self.num_process = num_process
        
        with open(config.dataset_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        encoder = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(encoder.encode(text), dtype=torch.long)
        if process_rank == 0:
            config.logger.debug("Number of tokens loaded are", len(self.tokens))
            config.logger.debug("Number of batches in 1 epoch is", len(self.tokens)//(self.total_size * self.num_process))
        
        self.current_pos = self.total_size*self.process_rank
        
    def next_batch(self):
        
        tokens_curr = self.tokens[self.current_pos:self.current_pos+self.total_size+1]
        inputs = tokens_curr[:-1].view(self.batch_size, self.context_size)
        labels = tokens_curr[1:].view(self.batch_size, self.context_size)
        
        self.current_pos = self.current_pos+self.total_size*self.num_process
        
        if self.current_pos+self.total_size*self.num_process+1 > len(self.tokens):
            self.current_pos = self.total_size*self.process_rank
        
        return inputs, labels
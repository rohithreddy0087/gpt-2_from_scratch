import os
import math
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from config_parser import get_config
  
class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_fc = nn.Linear(dim, 4*dim)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*dim, dim)
        self.c_proj.SCALE_WTS = 1 # to reduce the variance, see _init_weights
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = F.gelu(self.c_fc(x))
        x = self.dropout(self.c_proj(x))
        return x
      
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        
        self.head_dim = config.embedding_dim//config.num_heads
        self.scale = (1.0/math.sqrt(self.head_dim))
        self.num_heads = config.num_heads
        self.c_attn = nn.Linear(config.embedding_dim, 3*config.embedding_dim)
        self.c_proj = nn.Linear(config.embedding_dim, config.embedding_dim) # change in gpt-2 compared to gpt
        self.c_proj.SCALE_WTS = 1 # to reduce the variance, see _init_weights
        self.register_buffer('bias', torch.tril(torch.ones(config.context_size, config.context_size))
                                        .view(1,1,config.context_size,config.context_size))
       
    def forward(self, x):
        B,T,C = x.shape
        # batched calculation of QKV and then split it
        QKV = self.c_attn(x)
        # split the last dimension into 3 parts of self.embedding_dim
        Q, K, V = QKV.split(self.embedding_dim, dim=-1) # B, T, embedding_dim 
        # change the matrices shape into num_heads and then transpose
        # before transpose (B, T, num_heads, head_dim)
        # after transpose (B, num_heads, T, head_dim)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
                
        if not self.config.flash_attention:
            # The following dot product gives (B, num_heads, T, T)
            scaled_dot_prod = (Q@torch.transpose(K,-2,-1))*self.scale
            mask = scaled_dot_prod.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            attention = F.softmax(mask, dim=-1)
            # (B, num_heads, T, T) x (B, num_heads, T, head_dim) = (B, num_heads, T, head_dim)
            outputs = attention@V
        else:
            outputs = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        # stack the heads
        # after transpose (B, T, num_heads, head_dim)
        # need to make is contiguous before using view to reshape it to (B,T,C)
        outputs = outputs.transpose(1,2).contiguous().view(B,T,C)
        outputs = self.c_proj(outputs)
        return outputs
    
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embedding_dim)
        self.attn = MultiHeadSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embedding_dim)
        self.mlp = FeedForward(config.embedding_dim)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict()
        # add token embedding layer
        self.transformer['wte'] = nn.Embedding(config.vocab_size, config.embedding_dim)
        # add positonal encoding layer
        self.transformer['wpe'] = nn.Embedding(config.context_size, config.embedding_dim)
        # add all the transformer blocks
        self.transformer['h'] = nn.ModuleList()
        for _ in range(config.num_blocks):
            self.transformer['h'].append(TransformerBlock(config))
        # add the final layer norm
        self.transformer['ln_f'] = nn.LayerNorm(config.embedding_dim)
        # add the final head
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        
        # weight sharing scheme, we want token embedding layer to be same as the output linear layer, more than 30% weights are saved in 124M model
        self.transformer.wte.weight = self.lm_head.weight
        
        # apply is a method in nn.Module use to apply any function to all its submodules
        self.apply(self._init_weights)
        
    
    def _init_weights(self, module):
        # std values are from offical gpt-2 src code
        if isinstance(module, nn.Linear):
            # variance grows everytime we add the residuals in the Transformer block, hence we scale the linear layers in
            # feedforward and attention blocks
            std = 0.02
            if hasattr(module, 'SCALE_WTS'):
                std *= (2*self.config.num_blocks)** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, x):
        B, T = x.shape
        inp_emb = self.transformer.wte(x)
        pos_emb = self.transformer.wpe(torch.arange(0, T, device = x.device))
        x = inp_emb+pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        x = self.lm_head(x)
        return x
    
    # https://github.com/karpathy/nanoGPT/blob/9755682b981a45507f6eb9b11eadef8cb83cebd5/model.py#L263
    def configure_optimizer(self):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        self.config.logger.debug(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        self.config.logger.debug(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in self.config.device
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=self.config.max_lr, betas=(self.config.adam_beta1, self.config.adam_beta2), eps=self.config.adam_eps, **extra_args)
        self.config.logger.debug(f"using fused AdamW: {use_fused}")

        return optimizer
    
    # almost similar to from https://github.com/karpathy/nanoGPT/blob/9755682b981a45507f6eb9b11eadef8cb83cebd5/model.py#L207
    @classmethod
    def from_pretrained(cls, model_type, config):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        config.logger.debug("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(num_blocks=12, num_heads=12, embedding_dim=768),  # 124M params
            'gpt2-medium':  dict(num_blocks=24, num_heads=16, embedding_dim=1024), # 350M params
            'gpt2-large':   dict(num_blocks=36, num_heads=20, embedding_dim=1280), # 774M params
            'gpt2-xl':      dict(num_blocks=48, num_heads=25, embedding_dim=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # we can override the dropout rate, if desired
        # create a from-scratch initialized minGPT model
        
        for k, v in config_args.items():
            config.__dict__[k] = v
        
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        for key in sd_keys:
            if key not in sd_keys_hf:
                config.logger.debug(key)
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    
if __name__ == '__main__':
    config = get_config(os.path.join(os.getcwd(), 'config.ini'))
    model = GPT.from_pretrained('gpt2', config)
    model.eval()
    model.to(config.device)
    
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype = torch.long)
    tokens = tokens.unsqueeze(0).repeat(5,1)
    
    x = tokens.to(config.device)
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    max_length = 300
    
    while x.size(1) <  max_length:
        with torch.no_grad():
            logits = model(x)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) 
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            next_token = torch.multinomial(topk_probs, num_samples=1) 
            x_cont = torch.gather(topk_indices, -1, next_token)
            x = torch.cat((x, x_cont), dim=1)
            
    for i in range(5):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        config.logger.debug("> %s", decoded)
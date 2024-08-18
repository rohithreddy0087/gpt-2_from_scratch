from configparser import ConfigParser
import logging 
import os
import wandb
import torch

class ConfigFileparser:
    """
    Parses configfile and stores them in attributes
    """
    def __init__(self, configfile = "config.ini"):
        parser = ConfigParser()
        parser.read(configfile)
        self.cwd = os.getcwd()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.context_size = int(parser.get('GPT','CONTEXT_SIZE',fallback=1024))
        self.vocab_size = int(parser.get('GPT','VOCAB_SIZE',fallback=50257))
        self.embedding_dim = int(parser.get('GPT','EMBEDDING_DIM',fallback=768))
        self.num_heads = int(parser.get('GPT','NUM_HEADS',fallback=12))
        self.num_blocks = int(parser.get('GPT','NUM_TRANSFORMER_BLOCK',fallback=12))
        self.flash_attention = parser.getboolean('GPT','FLASH_ATTN',fallback=True)
        
        self.dataset_path = parser.get('TRAIN','DATASET_PATH')
        self.dataset_path = os.path.join(self.cwd, self.dataset_path)
        
        self.batch_size = int(parser.get('TRAIN','BATCH_SIZE',fallback=32))
        self.total_batch_size = int(parser.get('TRAIN','TOTAL_BATCH_SIZE',fallback=524288))
        self.epochs = int(parser.get('TRAIN','EPOCHS',fallback=200))
        self.max_steps = int(parser.get('TRAIN','MAX_STEPS',fallback=50))
        
        self.max_lr = float(parser.get('TRAIN','MAX_LEARNING_RATE',fallback=3e-4))
        self.min_lr = float(parser.get('TRAIN','MIN_LEARNING_RATE',fallback=self.max_lr*0.1))
        self.warmup_steps = int(parser.get('TRAIN','WARMUP_STEPS',fallback=10))
        
        
        self.adam_beta1 = float(parser.get('TRAIN','ADAM_BETA1',fallback=0.9))
        self.adam_beta2 = float(parser.get('TRAIN','ADAM_BETA2',fallback=0.95))
        self.adam_eps = float(parser.get('TRAIN','ADAM_EPS',fallback=1e-8))
        
        self.weight_decay = float(parser.get('TRAIN','WEIGHT_DECAY',fallback=0.1))
        
        self.load_weights = parser.getboolean('TRAIN','LOAD_WEIGHTS',fallback=False)
        self.saved_weights_path = parser.get('TRAIN','SAVED_WEIGHTS_PATH')
        self.saved_weights_path = os.path.join(self.cwd, self.saved_weights_path)
        self.save_weights_freq = int(parser.get('TRAIN','SAVE_WEIGHTS_FREQ',fallback=1000))
        
        self.eval_steps = int(parser.get('EVAL','STEPS',fallback=250))
        self.num_sequences = int(parser.get('EVAL','NUM_SEQUENCES',fallback=4))
        self.max_length = int(parser.get('EVAL','MAX_LENGTH',fallback=32))
        self.prompt = str(parser.get('EVAL','PROMPT',fallback="Hello, I'm a language model,"))
        self.topk = int(parser.get('EVAL','TOP_K',fallback=50))
        self.use_compile = parser.getboolean('EVAL','USE_COMPILE',fallback=True)
        
        self.log_file_name = parser.get('LOGGING','LOCAL_LOG_FILE_NAME',fallback="logs/debug.log")
        self.project = parser.get('LOGGING','PROJECT',fallback="GPT-2 from scratch")
        self.project_name = parser.get('LOGGING','NAME',fallback="Training from scratch")
        self.dataset_name = parser.get('LOGGING','DATASET',fallback="OpenWebText")
        self.arch = parser.get('LOGGING','ARCH',fallback="GPT-2")
        apikey = parser.get('LOGGING','WANDB_API_KEY',fallback=None)
        
        log_formatter = logging.Formatter("%(asctime)s %(message)s")
        self.logger = logging.getLogger(self.arch)
        fileHandler = logging.FileHandler(os.path.join(self.cwd,self.log_file_name))
        fileHandler.setFormatter(log_formatter)
        self.logger .addHandler(fileHandler)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(log_formatter)
        self.logger .addHandler(consoleHandler)
        self.logger.setLevel(logging.DEBUG)
        
        self.wandb_ = None
        if apikey is not None:
            os.environ["WANDB_API_KEY"] = apikey
            wandb.login()
            self.wandb_ = wandb.init(
                project=self.project, 
                name=self.project_name, 
                config={
                    "learning_rate": self.max_lr,
                    "architecture": self.arch,
                    "dataset": self.dataset_name,
                    "epochs": self.max_steps,
                })
        
def get_config(configfile):
    return ConfigFileparser(configfile)


import os
abs_path : os.path.dirname(__file__)

cfg = {
    'model_path':'../models/',
    'competition':'ventilator',
    'wandb_kernel':'r1ck',
    'apex':False,
    'print_freq':100,
    'num_workers':4,
    'model_name':'simple_lstm',
    'scheduler':'CosineAnnealingLR', # ['linear', 'cosine', 'ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    'batch_scheduler':False,
    #num_warmup_steps:100 # ['linear', 'cosine']
    #num_cycles:0.5 # 'cosine'
    #factor:0.2 # ReduceLROnPlateau
    #patience:4 # ReduceLROnPlateau
    #eps:1e-6 # ReduceLROnPlateau
    'T_max':50, # CosineAnnealingLR
    #T_0:50 # CosineAnnealingWarmRestarts
    'epochs':150,
    'max_grad_norm':1000,
    'gradient_accumulation_steps':1,
    # for simple lstm
    'input_dim':4,
    'lstm_dim':256,
    'dense_dim':256,
    'logit_dim':256,
    'num_classes':1,
    'hidden_size':64,
    'lr':5e-3,
    'min_lr':1e-6,
    'weight_decay':1e-6,
    'batch_size':64,
    'n_fold':5,
    'trn_fold':[0, 1, 2, 3, 4],
    'patience' : 20,
    'cate_seq_cols':['R', 'C'],
    'cont_seq_cols':['time_step', 'u_in', 'u_out'] + ['breath_time', 'u_in_time'],
    'train':True,
    'inference':True,
}
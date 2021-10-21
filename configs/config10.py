import os
abs_path : os.path.dirname(__file__)

cfg = {
    'seed':42,
    'model_path':'../models/',
    'competition':'ventilator',
    'wandb_kernel':'r1ck',
    'feature_version':2,
    'apex':True,
    'print_freq':1000,
    'num_workers':4,
    'model_name':'rnn_v3',
    'scheduler':'CosineAnnealingWarmRestarts', # ['linear', 'cosine', 'ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    'batch_scheduler':False,
    #num_warmup_steps:100 # ['linear', 'cosine']
    #num_cycles:0.5 # 'cosine'
    #factor:0.2 # ReduceLROnPlateau
    #patience:4 # ReduceLROnPlateau
    #eps:1e-6 # ReduceLROnPlateau
    'T_max':50, # CosineAnnealingLR
    'T_0':50, # CosineAnnealingWarmRestarts
    'epochs':200,
    'max_grad_norm':1000,
    'gradient_accumulation_steps':1,
    'hidden_size':1024,
    'lr':1e-3,
    'min_lr':1e-5,
    'weight_decay':1e-6,
    'batch_size':256,
    'n_fold':5,
    'trn_fold':[0, 1, 2, 3, 4],
    'early_stopping':True,
    'es_patience' : 60,
    'cate_seq_cols':[],
    'cont_seq_cols':['R', 'C', 'time_step', 'u_in', 'u_out'], #['time_step', 'u_in', 'u_out'] + ['breath_time', 'u_in_time'],
    'train':True,
    'inference':False,
    'debug':False
}
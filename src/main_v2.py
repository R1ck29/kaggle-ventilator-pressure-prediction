# # Can pytorch scores reach keras?
# I like [ynakama](https://www.kaggle.com/yasufuminakama)'s notebook and often use it as a baseline.  
# I used his [baseline](https://www.kaggle.com/yasufuminakama/ventilator-pressure-lstm-starter) again.  
# However, the same features and close model architecture will not result in the same score as a keras-based model using tpu.  
# Some people in [this discussion](https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/274176) have scored about 0.16x using Pytorch, including ynakama, so my abilities and ideas are lacking.  
# I'm going to share some of the ideas I've been working on for pytorch, so please give me some ideas if you don't mind.  
# 
# ## my experience (My best Score is Version 47)
# (There was a period of time when some of the experiments were not carefully managed, so the detailed hyperparameters are omitted.). 
# - CosineAnnealingWarmRestarts scored the best, while ReduceLROnPlateau, CosineAnnealingLR, and scheduler scored the worst.  
# - I got a better score when I didn't set the scheduler than when I did set it poorly.  
# - I tried lstm layers from 1 to 5, and I got good scores with 3 to 5 layers.  
# - The features were copied from a good public notebook to keep up with the tpu keras model. 
# - I took out and put in the RELU and LayerNorm layers, but the score didn't change much.  
# - The training data is reshape for each breath_id as in the other notebooks. This allows us to train in a batch processing manner, improving scores and reducing training time for me.  
#   
# If you find this notebook useful, it would be great if you could upvote it along with his [notebook](https://www.kaggle.com/yasufuminakama/ventilator-pressure-lstm-starter) and these references.
#   
#   
# ## references  
# https://www.kaggle.com/yasufuminakama/ventilator-pressure-lstm-starter
# https://www.kaggle.com/mistag/optuna-optimized-keras-base-model  
# https://www.kaggle.com/tenffe/finetune-of-tensorflow-bidirectional-lstm  
# https://www.kaggle.com/kensit/improvement-base-on-tensor-bidirect-lstm-0-173  
# https://www.kaggle.com/hamzaghanmi/tensorflow-bi-lstm-with-tpu  
# https://www.kaggle.com/lukaszborecki/ventilator-ts  

# ====================================================
# cfg
# ====================================================
# class cfg:
#     competition='ventilator'
#     apex=True
#     print_freq=1000
#     num_workers=4
#     model_name='rnn'
#     scheduler='CosineAnnealingWarmRestarts' # ['linear', 'cosine', 'ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
#     batch_scheduler=False
#     #num_warmup_steps=100 # ['linear', 'cosine']
#     #num_cycles=0.5 # 'cosine'
#     factor=0.995 # ReduceLROnPlateau
#     patience=7 # ReduceLROnPlateau
#     eps=1e-6 # ReduceLROnPlateau
#     T_max=50 # CosineAnnealingLR
#     T_0=50 # CosineAnnealingWarmRestarts
#     epochs=200
#     max_grad_norm=1000
#     gradient_accumulation_steps=1
#     hidden_size=512
#     lr=1e-3
#     min_lr=1e-5
#     weight_decay=1e-6
#     batch_size=256
#     n_fold=5
#     trn_fold=[0]
#     cate_seq_cols=[]
#     cont_seq_cols=['R', 'C', 'time_step', 'u_in', 'u_out']
#     train=True
#     inference=True
#     debug=False

# if cfg.debug:
#     cfg.epochs = 2
#     cfg.trn_fold=[0]

# ====================================================
# Library
# ====================================================
import os
import gc
from time import time
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from tqdm.auto import tqdm

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

import warnings
warnings.filterwarnings("ignore")

import wandb
from utils import get_score, init_logger, seed_everything, AverageMeter, asMinutes, timeSince, reduce_memory_usage
from conf import *
from model import CustomModel_v2
from loss import L1Loss_masked
from dataset import TrainDataset, TestDataset
from torch.utils.data import DataLoader
from create_features import run_blocks, AbstractBaseBlock, apply_normalization
#AddMultiplyingDividing, AddBreathTimeAndUInTime, RCDummry, LagFeatures


# if cfg.apex:
#     from apex import amp

class AddMultiplyingDividing(AbstractBaseBlock):
    def transform(self, input_df):
        input_df['area'] = input_df['time_step'] * input_df['u_in']
        input_df['area'] = input_df.groupby('breath_id')['area'].cumsum()
        input_df['cross'] = input_df['u_in']*input_df['u_out']
        input_df['cross2'] = input_df['time_step']*input_df['u_out']
        input_df['u_in_cumsum'] = (input_df['u_in']).groupby(input_df['breath_id']).cumsum()
        input_df['one'] = 1
        input_df['count'] = (input_df['one']).groupby(input_df['breath_id']).cumsum()
        input_df['u_in_cummean'] = input_df['u_in_cumsum'] / input_df['count']
        input_df = input_df.merge(
            input_df[input_df["u_out"]==0].groupby('breath_id')['u_in'].agg(["mean", "std", "max"]).add_prefix("u_out0_").reset_index(),
            on="breath_id"
        )
        input_df = input_df.merge(
            input_df[input_df["u_out"]==1].groupby('breath_id')['u_in'].agg(["mean", "std", "max"]).add_prefix("u_out1_").reset_index(),
            on="breath_id"
        )

        output_df = pd.DataFrame(
            {
                "area": input_df['area'],
                #"cross": input_df['cross'],
                #"cross2": input_df['cross2'],
                "u_in_cumsum": input_df['u_in_cumsum'],
                "u_in_cummean": input_df['u_in_cummean'],
                "u_out0_mean": input_df['u_out0_mean'],
                "u_out0_max": input_df['u_out0_max'],
                "u_out0_max": input_df['u_out0_std'],
                "u_out1_mean": input_df['u_out1_mean'],
                "u_out1_max": input_df['u_out1_max'],
                "u_out1_max": input_df['u_out1_std'],
            }
        )
        cfg.cont_seq_cols += output_df.add_suffix(f'@{self.__class__.__name__}').columns.tolist()
        return output_df


class RCDummry(AbstractBaseBlock):
    def transform(self, input_df):
        input_df['R_dummy'] = input_df['R'].astype(str)
        input_df['C_dummy'] = input_df['C'].astype(str)
        #input_df['RC_dummy'] = input_df['R_dummy'] + input_df['C_dummy']
        output_df = pd.get_dummies(input_df[["R_dummy", "C_dummy"]])
        cfg.cont_seq_cols += output_df.add_suffix(f'@{self.__class__.__name__}').columns.tolist()
        return output_df


class AddBreathTimeAndUInTime(AbstractBaseBlock):
    def transform(self, input_df):
        output_df = pd.DataFrame(
            {
                "breath_time": input_df['time_step'] - input_df['time_step'].shift(1),
                "u_in_time": input_df['u_in'] - input_df['u_in'].shift(1)
            }
        )
        output_df.loc[input_df['time_step'] == 0, 'breath_time'] = output_df['breath_time'].mean()
        output_df.loc[input_df['time_step'] == 0, 'u_in_time'] = output_df['u_in_time'].mean()
        cfg.cont_seq_cols += output_df.add_suffix(f'@{self.__class__.__name__}').columns.tolist()
        return output_df

class LagFeatures(AbstractBaseBlock):
    def transform(self, input_df):
        output_df = pd.DataFrame(
            {
                "u_in_lag1": input_df.groupby("breath_id")["u_in"].shift(1).fillna(0),
                "u_in_lag2": input_df.groupby("breath_id")["u_in"].shift(2).fillna(0),
                "u_in_lag3": input_df.groupby("breath_id")["u_in"].shift(3).fillna(0),
                "u_in_lag4": input_df.groupby("breath_id")["u_in"].shift(4).fillna(0),
                "u_in_lag-1": input_df.groupby("breath_id")["u_in"].shift(-1).fillna(0),
                "u_in_lag-2": input_df.groupby("breath_id")["u_in"].shift(-2).fillna(0),
                "u_in_lag-3": input_df.groupby("breath_id")["u_in"].shift(-3).fillna(0),
                "u_in_lag-4": input_df.groupby("breath_id")["u_in"].shift(-4).fillna(0),
                "u_out_lag1": input_df.groupby("breath_id")["u_out"].shift(1).fillna(0),
                "u_out_lag2": input_df.groupby("breath_id")["u_out"].shift(2).fillna(0),
                "u_out_lag3": input_df.groupby("breath_id")["u_out"].shift(3).fillna(0),
                "u_out_lag4": input_df.groupby("breath_id")["u_out"].shift(4).fillna(0),
                #"u_out_lag-1": input_df.groupby("breath_id")["u_out"].shift(-1).fillna(0),
                #"u_out_lag-2": input_df.groupby("breath_id")["u_out"].shift(-2).fillna(0),
                #"u_out_lag-3": input_df.groupby("breath_id")["u_out"].shift(-3).fillna(0),
                #"u_out_lag-4": input_df.groupby("breath_id")["u_out"].shift(-4).fillna(0),
            }
        )
        output_df["u_in_lag1_diff"] = output_df["u_in_lag1"] - input_df["u_in"]
        output_df["u_in_lag2_diff"] = output_df["u_in_lag2"] - input_df["u_in"]
        output_df["u_in_lag3_diff"] = output_df["u_in_lag3"] - input_df["u_in"]
        output_df["u_in_lag4_diff"] = output_df["u_in_lag4"] - input_df["u_in"]

        output_df["u_in_rolling_mean2"] = input_df[["breath_id", "u_in"]].groupby("breath_id").rolling(2).mean()["u_in"].reset_index(drop=True)
        output_df["u_in_rolling_mean4"] = input_df[["breath_id", "u_in"]].groupby("breath_id").rolling(4).mean()["u_in"].reset_index(drop=True)
        output_df["u_in_rolling_mean10"] = input_df[["breath_id", "u_in"]].groupby("breath_id").rolling(10).mean()["u_in"].reset_index(drop=True)
        if not cfg.debug:
            output_df["u_in_rolling_max2"] = input_df[["breath_id", "u_in"]].groupby("breath_id").rolling(2).max()["u_in"].reset_index(drop=True)
            output_df["u_in_rolling_max4"] = input_df[["breath_id", "u_in"]].groupby("breath_id").rolling(4).max()["u_in"].reset_index(drop=True)
            output_df["u_in_rolling_max10"] = input_df[["breath_id", "u_in"]].groupby("breath_id").rolling(10).max()["u_in"].reset_index(drop=True)
            output_df["u_in_rolling_min2"] = input_df[["breath_id", "u_in"]].groupby("breath_id").rolling(2).min()["u_in"].reset_index(drop=True)
            output_df["u_in_rolling_min4"] = input_df[["breath_id", "u_in"]].groupby("breath_id").rolling(4).min()["u_in"].reset_index(drop=True)
            output_df["u_in_rolling_min10"] = input_df[["breath_id", "u_in"]].groupby("breath_id").rolling(10).min()["u_in"].reset_index(drop=True)
            output_df["u_in_rolling_std2"] = input_df[["breath_id", "u_in"]].groupby("breath_id").rolling(2).std()["u_in"].reset_index(drop=True)
            output_df["u_in_rolling_std4"] = input_df[["breath_id", "u_in"]].groupby("breath_id").rolling(4).std()["u_in"].reset_index(drop=True)
            output_df["u_in_rolling_std10"] = input_df[["breath_id", "u_in"]].groupby("breath_id").rolling(10).std()["u_in"].reset_index(drop=True)
        for col in output_df.columns:
            output_df[col] = output_df[col].fillna(output_df[col].mean())
        cfg.cont_seq_cols += output_df.add_suffix(f'@{self.__class__.__name__}').columns.tolist()
        return output_df


def class2dict(f):
    return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))


def train_fn(fold, train_loader, model, criterion, optimizer, scaler, epoch, scheduler, device):
    model.train()
    losses = AverageMeter()
    start = end = time()
    iters = len(train_loader)
    for step, (inputs, y) in enumerate(train_loader):
        inputs, y = inputs.to(device), y.to(device)
        batch_size = inputs.size(0)
        with autocast(enabled=cfg.apex):
            pred = model(inputs)
            loss = criterion(pred, y, inputs[:,:,0].reshape(-1,80,1))
        losses.update(loss.item(), batch_size)
        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps
        # if cfg.apex:
        scaler.scale(loss).backward()
        # else:
        #     loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            # if cfg.apex:
            scaler.step(optimizer)
            # else:
            #     optimizer.step()
            optimizer.zero_grad()
            lr = 0

            if cfg.batch_scheduler:
                lr = scheduler.get_lr()[0]

            if isinstance(scheduler, CosineAnnealingWarmRestarts):
                scheduler.step(epoch + step / iters)
                # scheduler.step()       
                
        # if cfg.apex:
        scaler.update()
        end = time()

        wandb.log({f"[fold{fold}] loss": losses.val,
                   f"[fold{fold}] lr": scheduler.get_lr()[0]})
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    model.eval()
    preds = []
    losses = AverageMeter()
    start = end = time()
    for step, (inputs, y) in enumerate(valid_loader):
        inputs, y = inputs.to(device), y.to(device)
        batch_size = inputs.size(0)
        with torch.no_grad():
            pred = model(inputs)
        loss = criterion(pred, y, inputs[:,:,0].reshape(-1,80,1))
        losses.update(loss.item(), batch_size)
        preds.append(pred.view(-1).detach().cpu().numpy())
        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps
        end = time()
    preds = np.concatenate(preds)
    return losses.avg, preds


def inference_fn(test_loader, model, device):
    model.eval()
    model.to(device)
    preds = []
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    for step, (cont_seq_x) in tk0:
        cont_seq_x = cont_seq_x.to(device)
        with torch.no_grad():
            pred = model(cont_seq_x)
        preds.append(pred.view(-1).detach().cpu().numpy())
    preds = np.concatenate(preds)
    return preds

# ====================================================
# train loop
# ====================================================
def train_loop(LOGGER, X, train, y, fold, trn_idx, val_idx, OUTPUT_DIR, device):
    LOGGER.info(f"========== fold: {fold} training ==========")
    # ====================================================
    # loader
    # ====================================================
    #trn_idx = folds[folds['fold'] != fold].index
    #val_idx = folds[folds['fold'] == fold].index
    
    train_folds = X[trn_idx]
    valid_folds = X[val_idx]
    groups = train["breath_id"].unique()[val_idx]
    oof_folds = train[train["breath_id"].isin(groups)].reset_index(drop=True)
    y_train = y[trn_idx]
    y_true = y[val_idx]

    # train_dataset = TrainDataset(train_folds)
    # valid_dataset = TrainDataset(valid_folds)
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_folds),
        torch.from_numpy(y_train)
    )
    valid_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(valid_folds),
        torch.from_numpy(y_true)
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel_v2(cfg)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, eps=1e-08)
    num_train_steps = int(len(train_folds) / cfg.batch_size * cfg.epochs)
    
    def get_scheduler(optimizer):
        if cfg.scheduler=='linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler=='cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
            )
        elif cfg.scheduler=='ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=cfg.factor, patience=cfg.patience, verbose=True, eps=cfg.eps)
        elif cfg.scheduler=='CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.min_lr, last_epoch=-1)
        elif cfg.scheduler=='CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.T_0, T_mult=1, eta_min=cfg.min_lr, last_epoch=-1)
        return scheduler

    scheduler = get_scheduler(optimizer)

    # ====================================================
    # apex
    # ====================================================
    #if cfg.apex:
    #    model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # ====================================================
    # loop
    # ====================================================
    criterion = L1Loss_masked()

    scaler = GradScaler(enabled=cfg.apex)

    best_score = np.inf
    patience = 0
    avg_losses = []
    avg_val_losses = []
    for epoch in range(cfg.epochs):
        if cfg.early_stopping:
            if patience >= cfg.es_patience:
                print('---------------Early stopping triggered---------------')
                break

        start_time = time()

        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, scaler, epoch, scheduler, device)
        #avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, None, device)
        avg_losses.append(avg_loss)
        
        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)
        avg_val_losses.append(avg_val_loss)
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        # elif isinstance(scheduler, CosineAnnealingWarmRestarts):
        #     scheduler.step()

        # scoring
        score = avg_val_loss #get_score(y_true[non_expiratory_phase_val_idx], preds[non_expiratory_phase_val_idx])

        elapsed = time() - start_time

        best_notice = ""
        if score < best_score:
            patience = 0
            best_notice = "Best Score"
            best_score = score
            # LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'preds': preds},
                        os.path.join(OUTPUT_DIR, f"fold{fold}_best.pth"))
        else:
            patience += 1
    
        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s, lr: {optimizer.param_groups[0]["lr"]:.5f}, MAE Score: {score:.4f}, {best_notice}')
        wandb.log({f"[fold{fold}] epoch": epoch+1, 
                   f"[fold{fold}] avg_train_loss": avg_loss, 
                   f"[fold{fold}] avg_val_loss": avg_val_loss,
                   f"[fold{fold}] score": score})

    # plt.figure(figsize=(14,6))
    # plt.plot(avg_losses, label="Train Loss")
    # plt.plot(avg_val_losses, label="Train Loss")
    # plt.title(f"Fold {fold + 1} - Best score {best_score:.4f}", size=18)
    # plt.show()

    preds = torch.load(os.path.join(OUTPUT_DIR, f"fold{fold}_best.pth"), map_location=torch.device('cpu'))['preds']
    oof_folds['preds'] = preds.flatten()

    torch.cuda.empty_cache()
    gc.collect()
    
    return oof_folds

def get_result(LOGGER, result_df):
    preds = result_df['preds'].values
    labels = result_df['pressure'].values
    non_expiratory_phase_val_idx = result_df[result_df['u_out'] == 0].index # The expiratory phase is not scored
    score = get_score(labels[non_expiratory_phase_val_idx], preds[non_expiratory_phase_val_idx])
    LOGGER.info(f'Score (without expiratory phase): {score:<.4f}')


def main():
    """
    Prepare: 1.train 2.test
    """

    seed_everything(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    OUTPUT_DIR = os.path.join('../models', cfg.experiment_name)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # ====================================================
    # wandb
    # ====================================================
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        secret_value_0 = user_secrets.get_secret("wandb_api")
        wandb.login(key=secret_value_0)
        anony = None
    except:
        wandb.login(key='96c6d7e3179502e6830af9e44d274bfbc358118d')
        anony = None #"must"
        # print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')

    run = wandb.init(project="Ventilator-Pressure-Public", 
                    name=cfg.model_name,
                    config=class2dict(cfg),
                    group=cfg.model_name,
                    job_type="train",
                    anonymous=anony)

    LOGGER = init_logger(log_file=os.path.join(OUTPUT_DIR, 'train.log'))

    features_path = f'../input/features/'
    DATA_DIR = f'../input/ventilator-pressure-prediction/'

    train_fe_path = os.path.join(features_path, f'train_fe_v{cfg.feature_version}.csv')
    test_fe_path = os.path.join(features_path, f'test_fe_v{cfg.feature_version}.csv')
    cont_seq_cols_path = os.path.join(features_path, f'cont_seq_cols_v{cfg.feature_version}.pkl')
    
    if os.path.exists(train_fe_path) and os.path.exists(test_fe_path) and os.path.exists(cont_seq_cols_path):
        print(f'Loading {train_fe_path}')
        train = pd.read_csv(train_fe_path)
        # train = reduce_memory_usage(train)
        print(f'Loading {test_fe_path}')
        test = pd.read_csv(test_fe_path)
        # test = reduce_memory_usage(test)

        print(f'Loading {cont_seq_cols_path}')
        with open(cont_seq_cols_path, 'rb') as fp:
            cont_seq_cols = pickle.load(fp)
        cfg.cont_seq_cols = cont_seq_cols
    else:
        print(f'Loading {DATA_DIR + "train.csv"}')
        train = pd.read_csv(DATA_DIR + 'train.csv')
        # train = reduce_memory_usage(train)
        if cfg.debug:
            train = train[:80*5000]

        test = pd.read_csv(DATA_DIR + 'test.csv')
        # test = reduce_memory_usage(test)
        print(f'Loading {DATA_DIR + "test.csv"}')
        # sub = pd.read_csv(DATA_DIR + 'sample_submission.csv')

        feature_blocks = [
            AddMultiplyingDividing(),
            AddBreathTimeAndUInTime(),
            RCDummry(),
            LagFeatures()
        ]
        train = run_blocks(train, blocks=feature_blocks)
        test = run_blocks(test, blocks=feature_blocks, test=True)
        train.to_csv(train_fe_path, index=False)
        test.to_csv(test_fe_path, index=False)
        print(f'Saving {train_fe_path}')
        print(f'Saving {test_fe_path}')

        cfg.cont_seq_cols = list(set(cfg.cont_seq_cols))

        cont_seq_cols = cfg.cont_seq_cols
        with open(cont_seq_cols_path, 'wb') as fp:
            pickle.dump(cont_seq_cols, fp)
        print(f'Saving {cont_seq_cols_path}')

    # normalization
    train, test = apply_normalization(cfg, train, test)

    # reshape
    print(set(train.drop(["id", "breath_id", "pressure"], axis=1).columns) - set(cfg.cont_seq_cols))
    print(train.drop(["id", "breath_id", "pressure"], axis=1).shape)
    # print('cfg.cont_seq_cols', cfg.cont_seq_cols)
    print('number of cont_seq_cols', len(cfg.cont_seq_cols))

    X = np.float32(train.drop(["id", "breath_id", "pressure"], axis=1)).reshape(-1, 80, len(cfg.cont_seq_cols))
    y = np.float32(train["pressure"]).reshape(-1, 80, 1)
    X_test = np.float32(test.drop(["id", "breath_id"], axis=1)).reshape(-1, 80, len(cfg.cont_seq_cols))
    
    if cfg.train:
        # train 
        oof_df = pd.DataFrame()
        kfold = KFold(n_splits=cfg.n_fold, random_state=cfg.seed, shuffle=True)
        for fold, (trn_idx, val_idx) in enumerate(kfold.split(X=X, y=y)):
            if fold in cfg.trn_fold:
                _oof_df = train_loop(LOGGER, X, train, y, fold, trn_idx, val_idx, OUTPUT_DIR, device)
                
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(LOGGER, _oof_df)
        # CV result
        LOGGER.info(f"========== CV ==========")
        get_result(LOGGER, oof_df)
        # save result
        oof_df.to_csv(os.path.join(OUTPUT_DIR, 'oof_df.csv'), index=False)
    for i, breath_id in enumerate(oof_df["breath_id"].unique()):
        oof_df[oof_df["breath_id"]==breath_id].plot(x="time_step", y=["preds", "pressure", "u_out"], figsize=(16, 5))
        # plt.show()
        if i == 10:
            break
    
    if cfg.inference:
        test_loader = torch.utils.data.DataLoader(X_test, batch_size=512, shuffle=False, pin_memory=True)
        #test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
        for fold in cfg.trn_fold:
            model = CustomModel_v2(cfg)
            path = os.path.join(OUTPUT_DIR, f"fold{fold}_best.pth")
            state = torch.load(path, map_location=torch.device('cpu'))
            model.load_state_dict(state['model'])
            predictions = inference_fn(test_loader, model, device)
            test[f'fold{fold}'] = predictions
            del state, predictions; gc.collect()
            torch.cuda.empty_cache()
        # submission
        test['pressure'] = test[[f'fold{fold}' for fold in cfg.trn_fold]].mean(1)
        test[['id', 'pressure']+[f'fold{fold}' for fold in cfg.trn_fold]].to_csv(os.path.join(OUTPUT_DIR, 'raw_submission.csv'), index=False)
        test[['id', 'pressure']].to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)

    wandb.finish()

if __name__ == '__main__':
    main()
import gc
# ====================================================
# Library
# ====================================================
import os

import time

import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import warnings

import category_encoders as ce
import torch
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts,
                                      ReduceLROnPlateau)
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AdamW, get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)

warnings.filterwarnings("ignore")
import wandb
from utils import get_score, init_logger, seed_everything, AverageMeter, asMinutes, timeSince
from conf import *
from model import CustomModel, CustomModel_v2
from loss import L1Loss_masked
from dataset import TrainDataset, TestDataset

if cfg.apex:
    from apex import amp


def class2dict(f):
    return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    losses = AverageMeter()
    start = end = time.time()
    # for step, (cate_seq_x, cont_seq_x, u_out, y) in enumerate(train_loader):
    for step, (inputs, y) in enumerate(train_loader):
        # loss_mask = u_out == 0
        # cate_seq_x, cont_seq_x, y = cate_seq_x.to(device), cont_seq_x.to(device), y.to(device)
        # batch_size = cont_seq_x.size(0)
        # pred = model(cate_seq_x, cont_seq_x)
        # loss = 2. * criterion(pred[loss_mask], y[loss_mask]) + criterion(pred[loss_mask == 0], y[loss_mask == 0])
        inputs, y = inputs.to(device), y.to(device)
        batch_size = inputs.size(0)
        with autocast():
            pred = model(inputs)
            loss = criterion(pred, y, inputs[:,:,0].reshape(-1,80,1))

        losses.update(loss.item(), batch_size)
        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps
        if cfg.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if cfg.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.6f}  '
                  .format(
                   epoch+1, step, len(train_loader),
                   remain=timeSince(start, float(step+1)/len(train_loader)),
                   loss=losses,
                   grad_norm=grad_norm,
                   lr=scheduler.get_lr()[0],
                   ))
        wandb.log({f"[fold{fold}] loss": losses.val,
                   f"[fold{fold}] lr": scheduler.get_lr()[0]})
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    model.eval()
    preds = []
    losses = AverageMeter()
    start = end = time.time()
    # for step, (cate_seq_x, cont_seq_x, u_out, y) in enumerate(valid_loader):
    for step, (inputs, y) in enumerate(valid_loader):
        # loss_mask = u_out == 0
        # cate_seq_x, cont_seq_x, y = cate_seq_x.to(device), cont_seq_x.to(device), y.to(device)
        # batch_size = cont_seq_x.size(0)
        # with torch.no_grad():
        #     pred = model(cate_seq_x, cont_seq_x)
        # loss = 2. * criterion(pred[loss_mask], y[loss_mask]) + criterion(pred[loss_mask == 0], y[loss_mask == 0])
        inputs, y = inputs.to(device), y.to(device)
        batch_size = inputs.size(0)
        with torch.no_grad():
            pred = model(inputs)
        loss = criterion(pred, y, inputs[:,:,0].reshape(-1,80,1))

        losses.update(loss.item(), batch_size)
        preds.append(pred.view(-1).detach().cpu().numpy())
        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(
                   step, len(valid_loader),
                   remain=timeSince(start, float(step+1)/len(valid_loader)),
                   loss=losses,
                   ))
    preds = np.concatenate(preds)
    return losses.avg, preds


def inference_fn(test_loader, model, device):
    model.eval()
    model.to(device)
    preds = []
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    # for step, (cate_seq_x, cont_seq_x) in tk0:
    for step, (cont_seq_x) in tk0:
        # cate_seq_x, cont_seq_x = cate_seq_x.to(device), cont_seq_x.to(device)
        cont_seq_x = cont_seq_x.to(device)
        with torch.no_grad():
            # pred = model(cate_seq_x, cont_seq_x)
            pred = model(cont_seq_x)
        preds.append(pred.view(-1).detach().cpu().numpy())
    preds = np.concatenate(preds)
    return preds


# ====================================================
# train loop
# ====================================================
def train_loop(LOGGER, folds, fold, OUTPUT_DIR, device):

    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index
    
    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    y_true = valid_folds['pressure'].values
    non_expiratory_phase_val_idx = valid_folds[valid_folds['u_out'] == 0].index # The expiratory phase is not scored

    train_dataset = TrainDataset(cfg, train_folds)
    valid_dataset = TrainDataset(cfg, valid_folds)
    
    # # reshape
    # print(set(folds.drop(["id", "breath_id", "pressure"], axis=1).columns) - set(cfg.cont_seq_cols))
    # print(folds.drop(["id", "breath_id", "pressure"], axis=1).shape)
    # print(len(cfg.cont_seq_cols))

    # X = np.float32(folds.drop(["id", "breath_id", "pressure"], axis=1)).reshape(-1, 80, len(cfg.cont_seq_cols))
    # y = np.float32(folds["pressure"]).reshape(-1, 80, 1)

    # train_folds = X[trn_idx]
    # valid_folds = X[val_idx]
    # groups = folds["breath_id"].unique()[val_idx]
    # oof_folds = folds[folds["breath_id"].isin(groups)].reset_index(drop=True)
    # y_train = y[trn_idx]
    # y_true = y[val_idx]

    # train_dataset = torch.utils.data.TensorDataset(
    #     torch.from_numpy(train_folds),
    #     torch.from_numpy(y_train)
    # )
    # valid_dataset = torch.utils.data.TensorDataset(
    #     torch.from_numpy(valid_folds),
    #     torch.from_numpy(y_true)
    # )

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
    if 'rnn_v3' == cfg.model_name:
        print('Model: CustomModel_v2')
        model = CustomModel_v2(cfg)
    else:
        print('Model: CustomModel')
        model = CustomModel(cfg)
        
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
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
    if cfg.apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # ====================================================
    # loop
    # ====================================================
    criterion = L1Loss_masked() #nn.L1Loss()

    best_score = np.inf

    patience = 0

    for epoch in range(cfg.epochs):

        if patience >= cfg.patience:
            print('Early stopping triggered')
            continue

        start_time = time.time()

        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()

        # scoring
        score = get_score(y_true[non_expiratory_phase_val_idx], preds[non_expiratory_phase_val_idx])

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - MAE Score (without expiratory phase): {score:.4f}')
        wandb.log({f"[fold{fold}] epoch": epoch+1, 
                   f"[fold{fold}] avg_train_loss": avg_loss, 
                   f"[fold{fold}] avg_val_loss": avg_val_loss,
                   f"[fold{fold}] score": score})
        
        if score < best_score:
            patience = 0
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'preds': preds},
                        os.path.join(OUTPUT_DIR, f"fold{fold}_best.pth"))
        else:
            patience += 1
            
    preds = torch.load(os.path.join(OUTPUT_DIR, f"fold{fold}_best.pth"), map_location=torch.device('cpu'))['preds']
    valid_folds['preds'] = preds

    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_folds


def get_result(result_df, LOGGER):
    preds = result_df['preds'].values
    labels = result_df['pressure'].values
    non_expiratory_phase_val_idx = result_df[result_df['u_out'] == 0].index # The expiratory phase is not scored
    score = get_score(labels[non_expiratory_phase_val_idx], preds[non_expiratory_phase_val_idx])
    LOGGER.info(f'Score (without expiratory phase): {score:<.4f}')
    
# ====================================================
# main
# ====================================================
def main():
    
    """
    Prepare: 1.train 2.test
    """

    seed_everything()

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

    # ====================================================
    # Data Loading
    # ====================================================
    # raw_data_dir = '../input/ventilator-pressure-prediction/'
    features_path = f'../input/features/'

    # sub = pd.read_csv(os.path.join(raw_data_dir, 'sample_submission.csv'))
    
    if cfg.train:
        train_fold_path = os.path.join(features_path, f'train_fold_v{cfg.feature_version}.csv')
        print(f'Loading Train df with Fold: {train_fold_path}')
        train = pd.read_csv(train_fold_path)

        # train 
        oof_df = pd.DataFrame()
        for fold in range(cfg.n_fold):
            if fold in cfg.trn_fold:
                _oof_df = train_loop(LOGGER, train, fold, OUTPUT_DIR, device)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df, LOGGER)
        # CV result
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df, LOGGER)
        # save result
        oof_df.to_csv(os.path.join(OUTPUT_DIR, 'oof_df.csv'), index=False)
    
    if cfg.inference:
        test_fe_path = os.path.join(features_path, f'test_fe_v{cfg.feature_version}.csv')
        print(f'Loading Test df with Fold: {test_fe_path}')
        test = pd.read_csv(test_fe_path)
        # X_test = np.float32(test.drop(["id", "breath_id"], axis=1)).reshape(-1, 80, len(cfg.cont_seq_cols))
        test_dataset = TestDataset(test)
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(X_test, batch_size=512, shuffle=False, pin_memory=True)
        for fold in cfg.trn_fold:
            model = CustomModel(cfg)
            path = os.path.join(OUTPUT_DIR, f"fold{fold}_best.pth")
            state = torch.load(path, map_location=torch.device('cpu'))
            model.load_state_dict(state['model'])
            predictions = inference_fn(test_loader, model, device)
            test[f'fold{fold}'] = predictions
            del state, predictions; gc.collect()
            torch.cuda.empty_cache()
        # submission
        test['pressure'] = test[[f'fold{fold}' for fold in range(cfg.n_fold)]].mean(1)
        test[['id', 'pressure']+[f'fold{fold}' for fold in range(cfg.n_fold)]].to_csv(os.path.join(OUTPUT_DIR, 'raw_submission.csv'), index=False)
        test[['id', 'pressure']].to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)
    
    wandb.finish()


if __name__ == '__main__':
    main()

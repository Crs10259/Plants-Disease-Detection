import os 
import random 
import json
import torch
import numpy as np 
import pandas as pd 
import warnings
import psutil
from datetime import datetime
from torch import nn, optim
from config import config 
from collections import OrderedDict
from torch.utils.data import DataLoader
from dataset.dataloader import *
from sklearn.model_selection import train_test_split    
from timeit import default_timer as timer
from models.model import *
from utils import *
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from data_aug import DataAugmentor
from data_aug import main as data_aug_main
import timm.utils

def setup_environment():
    """设置随机种子和CUDA环境"""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings('ignore')

def get_device():
    """获取训练设备（如果可用则使用GPU，否则使用CPU）"""
    if config.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('Using GPU:', torch.cuda.get_device_name(0))
        else:
            print('CUDA requested but not available, falling back to CPU')
            device = torch.device('cpu')
            print('Using CPU')
    elif config.device == "cpu":
        device = torch.device('cpu')
        print('Using CPU (as requested)')
    else:
        # 自动模式 - 如果可用则使用GPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('Using GPU:', torch.cuda.get_device_name(0))
        else:
            device = torch.device('cpu')
            print('Using CPU (GPU is not available)')
    return device

def create_directories():
    """创建必要的检查点和日志目录"""
    directories = [
        config.submit, 
        config.weights, 
        config.best_models, 
        config.logs,
        config.test_data  # Add test data directory
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # 为每个模型创建特定文件夹
        if directory == config.best_models or directory == config.weights:
            os.makedirs(os.path.join(directory, config.model_name, "0"), exist_ok=True)

def train_epoch(model, train_dataloader, criterion, optimizer, epoch, log, device, scaler=None, model_ema=None):
    """单轮训练函数（支持混合精度）"""
    train_losses = AverageMeter()
    train_top1 = AverageMeter()
    train_top2 = AverageMeter()
    
    model.train()
    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch}')
    
    # 跟踪错误图像，以便稍后处理
    error_files = []
    
    for iter, batch in enumerate(progress_bar):
        try:
            # 确保批次是有效的
            if len(batch) != 2:
                log.write(f"Skipping error batch in iteration {iter}\n")
                continue
                
            input, target = batch
            
            # 确保数据有效
            if input is None or len(input) == 0:
                log.write(f"Skipping empty input in iteration {iter}\n")
                continue
                
            input = input.to(device)
            target = torch.tensor(target).to(device)
            
            # 应用Mixup或CutMix数据增强
            if config.use_mixup:
                # 随机选择使用Mixup或CutMix
                r = np.random.rand(1)
                if r < config.cutmix_prob:
                    # 使用CutMix
                    input, target_a, target_b, lam = cutmix_data(input, target, config.mixup_alpha)
                    use_mixup = True
                else:
                    # 使用Mixup
                    input, target_a, target_b, lam = mixup_data(input, target, config.mixup_alpha)
                    use_mixup = True
            else:
                use_mixup = False
            
            # 使用混合精度训练
            if config.use_amp and scaler is not None:
                with autocast():
                    # 前向传播
                    output = model(input)
                    
                    # 计算损失
                    if use_mixup:
                        loss = mixup_criterion(criterion, output, target_a, target_b, lam)
                    else:
                        loss = criterion(output, target)
                
                # 反向传播（使用梯度缩放）
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # 梯度裁剪
                if config.gradient_clip_val > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # 前向传播
                output = model(input)
                
                # 计算损失
                if use_mixup:
                    loss = mixup_criterion(criterion, output, target_a, target_b, lam)
                else:
                    loss = criterion(output, target)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                if config.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
                
                optimizer.step()
            
            # 更新EMA模型
            if model_ema is not None:
                update_ema(model_ema, model, iter)
            
            # 计算指标
            if use_mixup:
                # 在Mixup的情况下，使用第一个标签计算准确率（近似）
                precision1_train, precision2_train = accuracy(output, target_a, topk=(1, 2))
            else:
                precision1_train, precision2_train = accuracy(output, target, topk=(1, 2))
                
            train_losses.update(loss.item(), input.size(0))
            train_top1.update(precision1_train[0], input.size(0))
            train_top2.update(precision2_train[0], input.size(0))
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{train_losses.avg:.3f}',
                'top1': f'{train_top1.avg:.3f}',
                'top2': f'{train_top2.avg:.3f}'
            })
            
            # 释放内存
            del output
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
        except Exception as e:
            if hasattr(batch, '__getitem__') and len(batch) > 1 and isinstance(batch[1], list) and len(batch[1]) > 0:
                # 记录出错的文件
                error_files.extend(batch[1])
                log.write(f"Error in training iteration {iter}: {str(e)} - these files will be skipped in the future\n")
            else:
                log.write(f"Error in training iteration {iter}: {str(e)}\n")
            continue
            
    if error_files:
        log.write(f"Found {len(error_files)} problematic files in this epoch.\n")
    
    return train_losses.avg, train_top1.avg, train_top2.avg

def evaluate(val_loader, model, criterion, device):
    """在验证集上评估模型"""
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    
    model.eval()
    progress_bar = tqdm(val_loader, desc='Validation')
    
    with torch.no_grad():
        for input, target in progress_bar:
            try:
                input = input.to(device)
                target = torch.tensor(target).to(device)
                
                # 前向传播
                output = model(input)
                loss = criterion(output, target)

                # 计算指标
                prec1, prec2 = accuracy(output, target, topk=(1, 2))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1[0], input.size(0))
                top2.update(prec2[0], input.size(0))
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{losses.avg:.3f}',
                    'top1': f'{top1.avg:.3f}',
                    'top2': f'{top2.avg:.3f}'
                })
            
                # 释放内存
                del output
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error in validation: {str(e)}")
                continue
                
    return losses.avg, top1.avg, top2.avg

def test(test_loader, model, device):
    """测试模型并保存预测结果"""
    csv_map = OrderedDict({"filename": [], "probability": []})
    model.eval()
    
    progress_bar = tqdm(test_loader, desc='Testing')
    
    with torch.no_grad(), open("./submit/baseline.json", "w", encoding="utf-8") as f:
        submit_results = []
        for input, filepath in progress_bar:
            filepath = [os.path.basename(x) for x in filepath]
            input = input.to(device)
            
            # 前向传播
            y_pred = model(input)
            smax = nn.Softmax(1)
            smax_out = smax(y_pred)
        
            # 保存预测结果
            csv_map["filename"].extend(filepath)
            for output in smax_out:
                prob = ";".join([str(i) for i in output.data.tolist()])
                csv_map["probability"].append(prob)
                
        # 处理结果
        result = pd.DataFrame(csv_map)
        result["probability"] = result["probability"].map(lambda x: [float(i) for i in x.split(";")])
        
        # 创建提交结果
        for index, row in result.iterrows():
            pred_label = np.argmax(row['probability'])
            if pred_label > 43:
                pred_label = pred_label + 2
            submit_results.append({"image_id": row['filename'], "disease_class": pred_label})
            
        json.dump(submit_results, f, ensure_ascii=False, cls=MyEncoder)

def load_checkpoint(model, optimizer, device):
    """如果存在则加载检查点"""
    checkpoint_path = os.path.join(config.best_models, config.model_name, "0", "model_best.pth.tar")
    start_epoch = 0
    best_precision1 = 0
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 模型名称
        model_name = checkpoint.get("model_name", "")
        
        # 根据不同模型类型处理状态字典
        # if "resnet" in model_name:
        #     # 处理ResNet状态字典
        #     model_state_dict = {k.replace('fc.weight', 'fc.0.weight').replace('fc.bias', 'fc.0.bias'): v 
        #                         for k, v in checkpoint['state_dict'].items()}

        if "efficientnet" in model_name:
            # 处理EfficientNet状态字典（无需转换）
            model_state_dict = checkpoint['state_dict']
        elif "convnext" in model_name:
            # 处理ConvNeXt状态字典（无需转换）
            model_state_dict = checkpoint['state_dict']
        else:
            # 默认情况，尝试直接加载
            model_state_dict = checkpoint['state_dict']
            
        # 尝试加载状态字典
        try:
            model.load_state_dict(model_state_dict)
            print("Model weights loaded successfully")
        except Exception as e:
            print(f"Failed to load model weights: {str(e)}")
            print("Trying to load partial model weights...")
            
            # 创建当前模型的状态字典
            current_model_dict = model.state_dict()
            
            # 筛选出匹配的键
            pretrained_dict = {k: v for k, v in model_state_dict.items() if k in current_model_dict}
            
            # 更新当前模型字典
            current_model_dict.update(pretrained_dict)
            model.load_state_dict(current_model_dict)
            print(f"Loaded {len(pretrained_dict)}/{len(current_model_dict)} layer parameters")
            
        # 加载其他信息
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_precision1 = checkpoint['best_precision1']
        print(f"Loaded checkpoint from epoch {start_epoch} with precision {best_precision1:.2f}")
    else:
        print("No checkpoint found. Starting from scratch.")
        
    return start_epoch, best_precision1

def main():
    # 设置
    setup_environment()
    create_directories()

    # 初始化日志记录器
    log = Logger()
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log.open(os.path.join(config.logs, f"log_train_{now}.txt"), mode="a")
    log.write(f"PyTorch version: {torch.__version__}\n")
    device = get_device()    
    log.write(f"Device used: {device}\n")
    
    # 模型设置
    log.write(f"Using model: {config.model_name}\n")
    model = get_net()
    
    # 应用梯度检查点以节省内存
    if config.use_gradient_checkpointing and hasattr(model, 'set_grad_checkpointing'):
        model.set_grad_checkpointing(enable=True)
        log.write("Gradient checkpointing enabled\n")
    
    model = model.to(device)
    
    # 使用改进的损失函数
    criterion = get_loss_function(device)
    log.write(f"Using loss function: {criterion.__class__.__name__}\n")
    
    # 使用高级优化器
    optimizer = get_optimizer(model, config.optimizer)
    log.write(f"Using optimizer: {config.optimizer}\n")
    
    # 如果存在则加载检查点
    start_epoch, best_precision1 = load_checkpoint(model, optimizer, device)
    log.write(f"Starting from epoch {start_epoch} with best precision {best_precision1:.2f}\n")
    
    # 创建EMA模型
    model_ema = create_model_ema(model)
    if model_ema is not None:
        log.write("Using EMA model with decay rate {:.4f}\n".format(config.ema_decay))
    
    # 在数据加载之前添加数据增强
    augmentor = DataAugmentor(config)
    
    # 数据加载
    log.write("Loading dataset...\n")
    
    # 如果启用了数据增强，进行增强并获取增强后的文件列表
    if config.use_data_aug and config.use_mode == 'merge':
        log.write("Starting data augmentation...\n")
        augmented_files = augmentor.augment_directory()
          
        # 将增强后的文件添加到训练数据中
        if augmented_files:
            aug_data = pd.DataFrame({
                "filename": augmented_files,
                "label": [int(os.path.basename(os.path.dirname(f))) for f in augmented_files]
            })
            train_data = get_files(config.train_data, "train")
            train_data = pd.concat([train_data, aug_data], ignore_index=True)
            log.write(f"Added {len(augmented_files)} augmented images to training set\n")

    elif config.use_data_aug and config.use_mode == 'replace':
        if data_aug_main():
            log.write("Data augmentation completed\n")
            train_data = get_files(config.aug_target_path, "train")
        else:
            log.write("Data augmentation failed, stop augmenting.\n")
            return
    else:
        train_data = get_files(config.train_data, "train")
        
    # Check if train_data is empty or None
    if train_data is None or train_data.empty:
        print("Error: Training data could not be loaded. Please check the data path and format.")
        return  # Exit the function if data loading fails

    print("Loading training data...")
    train_data = get_files(config.train_data, "train")
    print(f"Loaded training data: {train_data.head()}")

    train_data_list, val_data_list = train_test_split(train_data, test_size=0.15, stratify=train_data["label"])
    log.write(f"Training set size: {len(train_data_list)}, Validation set size: {len(val_data_list)}\n")
    
    test_files = get_files(config.test_data, "test")
    log.write(f"Test set size: {len(test_files)}\n")

    train_dataloader = DataLoader(
        PlantDiseaseDataset(train_data_list), 
        batch_size=config.train_batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True  
    )
    
    val_dataloader = DataLoader(
        PlantDiseaseDataset(val_data_list, train=False), 
        batch_size=config.val_batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_dataloader = DataLoader(
        PlantDiseaseDataset(test_files, test=True), 
        batch_size=config.test_batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 设置学习率调度器
    if config.scheduler == 'onecycle':
        steps_per_epoch = len(train_dataloader)
        scheduler = get_scheduler(optimizer, config.epoch - start_epoch, steps_per_epoch)
        log.write(f"Using OneCycleLR scheduler with {steps_per_epoch} steps per epoch\n")
    else:
        scheduler = get_scheduler(optimizer, config.epoch - start_epoch)
        log.write(f"Using {config.scheduler} scheduler\n")
    
    # 混合精度训练初始化
    scaler = GradScaler() if config.use_amp and device.type == 'cuda' else None
    if scaler:
        log.write("Mixed precision training enabled\n")
    
    # 早停策略
    if config.use_early_stopping:
        early_stopping_counter = 0
        best_val_loss = float('inf')
        log.write(f"Early stopping enabled with patience of {config.early_stopping_patience} epochs\n")
    
    # 训练循环
    start = timer()
    log.write("Starting training...\n")
    
    for epoch in range(start_epoch + 1, config.epoch + 1):
        try:
            # 应用学习率调度（根据调度器类型）
            if config.scheduler == 'cosine':
                scheduler.step(epoch)
            elif config.scheduler == 'step':
                scheduler.step()
            # OneCycleLR每步更新一次，在训练循环中处理
            
            current_lr = get_learning_rate(optimizer)
            log.write(f"Epoch {epoch}/{config.epoch} - Current learning rate: {current_lr:.6f}\n")
            
            # 训练
            train_loss, train_top1, train_top2 = train_epoch(
                model, train_dataloader, criterion, optimizer, epoch, log, device, scaler, model_ema
            )
            
            # 评估 - 使用EMA模型（如果可用）
            if model_ema is not None:
                # 创建临时备份
                orig_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
                
                # 加载EMA参数
                model.load_state_dict({k: v.clone().detach() for k, v in model_ema.module.state_dict().items()})
                
                # 使用EMA模型评估
                valid_loss, valid_top1, valid_top2 = evaluate(val_dataloader, model, criterion, device)
                log.write("Validation with EMA model\n")
                
                # 恢复原始参数
                model.load_state_dict(orig_state)
                del orig_state
            else:
                # 正常评估
                valid_loss, valid_top1, valid_top2 = evaluate(val_dataloader, model, criterion, device)
            
            # 早停检查
            if config.use_early_stopping:
                if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    log.write(f"Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{config.early_stopping_patience}\n")
                    if early_stopping_counter >= config.early_stopping_patience:
                        log.write(f"Early stopping triggered! Validation loss did not improve for {config.early_stopping_patience} consecutive epochs.\n")
                        break
            
            # 保存检查点
            is_best = valid_top1 > best_precision1
            best_precision1 = max(valid_top1, best_precision1)
            
            # 保存EMA模型（如果可用且性能更好）
            if model_ema is not None and is_best:
                # 保存当前模型的状态
                orig_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
                
                # 临时加载EMA权重
                model.load_state_dict({k: v.clone().detach() for k, v in model_ema.module.state_dict().items()})
                
                # 保存检查点
                save_checkpoint({
                    "epoch": epoch,
                    "model_name": config.model_name,
                    "state_dict": model.state_dict(),
                    "best_precision1": best_precision1,
                    "optimizer": optimizer.state_dict(),
                    "valid_loss": [valid_loss, valid_top1, valid_top2],
                    "is_ema": True,
                }, is_best, 0)
                
                # 恢复原始参数
                model.load_state_dict(orig_state)
                del orig_state
                
                log.write("Saved checkpoint with EMA weights\n")
            else:
                # 保存普通检查点
                save_checkpoint({
                    "epoch": epoch,
                    "model_name": config.model_name,
                    "state_dict": model.state_dict(),
                    "best_precision1": best_precision1,
                    "optimizer": optimizer.state_dict(),
                    "valid_loss": [valid_loss, valid_top1, valid_top2],
                    "is_ema": False,
                }, is_best, 0)
            
            if is_best:
                log.write("【NEW BEST MODEL】\n")
            
            # 记录进度
            log.write(
                f'Epoch {epoch:3d}/{config.epoch} | '
                f'Train Loss: {train_loss:.3f} Top1: {train_top1:.3f} Top2: {train_top2:.3f} | '
                f'Valid Loss: {valid_loss:.3f} Top1: {valid_top1:.3f} Top2: {valid_top2:.3f} | '
                f'Time: {time_to_str(timer() - start)}\n'
            )
            
        except Exception as e:
            log.write(f"Error in epoch {epoch}: {str(e)}\n")
            import traceback
            log.write(traceback.format_exc())
            continue
    
    # 测试
    total_time = time_to_str(timer() - start)
    log.write(f"Training completed, total time: {total_time}\n")
    log.write("Loading best model for testing...\n")
    best_model_path = os.path.join(config.best_models, config.model_name, "0", "model_best.pth.tar")
    if os.path.exists(best_model_path):
        best_model = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_model["state_dict"])
        test(test_dataloader, model, device)
        log.write("Testing completed. Results saved to ./submit/baseline.json\n")
    else:
        log.write("No best model found for testing.\n")

if __name__ == "__main__":
    main()






import shutil
import torch
import sys
import os
import json
import numpy as np
from config import config
from torch import nn
import torch.nn.functional as F
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.utils import ModelEmaV2
from torch.optim.lr_scheduler import OneCycleLR

def save_checkpoint(state, is_best, fold):
    """保存检查点"""
    os.makedirs(os.path.join(config.weights, config.model_name, str(fold)), exist_ok=True)
    os.makedirs(os.path.join(config.best_models, config.model_name, str(fold)), exist_ok=True)
    
    filename = os.path.join(config.weights, config.model_name, str(fold), "_checkpoint.pth.tar")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(config.best_models, config.model_name, str(fold), 'model_best.pth.tar'))

class AverageMeter(object):
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """每3个轮次将初始学习率降低10倍"""
    lr = config.lr * (0.1 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def schedule(current_epoch, current_lrs, **logs):
    """学习率调度器"""
    lrs = [1e-3, 1e-4, 0.5e-4, 1e-5, 0.5e-5]
    epochs = [0, 1, 6, 8, 12]
    for lr, epoch in zip(lrs, epochs):
        if current_epoch >= epoch:
            current_lrs[5] = lr
            if current_epoch >= 2:
                current_lrs[4] = lr * 1
                current_lrs[3] = lr * 1
                current_lrs[2] = lr * 1
                current_lrs[1] = lr * 1
                current_lrs[0] = lr * 0.1
    return current_lrs

def get_optimizer(model, name='adamw'):
    """获取优化器
    
    参数：
        model: 模型
        name: 优化器名称
    
    返回：
        优化器实例
    """
    if name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay
        )
    elif name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay
        )
    elif name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=config.lr, 
            momentum=0.9, 
            weight_decay=config.weight_decay
        )
    elif name == 'ranger':
        # Ranger优化器（RAdam + Lookahead）
        try:
            from ranger_adabelief import Ranger  # 需要先安装ranger-adabelief库
            optimizer = Ranger(
                model.parameters(), 
                lr=config.lr, 
                weight_decay=config.weight_decay
            )
        except ImportError:
            print("Ranger optimizer not available, falling back to AdamW")
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=config.lr, 
                weight_decay=config.weight_decay
            )
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
    
    # 如果使用Lookahead，包装优化器
    if config.use_lookahead and name != 'ranger':  # Ranger已经包含Lookahead
        try:
            from lookahead import Lookahead  # 需要先安装lookahead-pytorch库
            optimizer = Lookahead(optimizer, k=5, alpha=0.5)
        except ImportError:
            print("Lookahead not available")
    
    return optimizer

def get_scheduler(optimizer, num_epochs, steps_per_epoch=None):
    """获取学习率调度器
    
    参数：
        optimizer: 优化器
        num_epochs: 总轮次
        steps_per_epoch: 每轮的步数（对于OneCycleLR需要）
    
    返回：
        学习率调度器
    """
    if config.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=10, 
            gamma=0.1
        )
    elif config.scheduler == 'cosine':
        # 带预热的余弦退火 - 修复参数错误
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            # 移除 t_mul 参数
            lr_min=1e-6,
            warmup_lr_init=config.lr * config.warmup_factor,
            warmup_t=config.warmup_epochs,
            cycle_limit=1,
            t_in_epochs=True
        )
    elif config.scheduler == 'onecycle':
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch must be provided for OneCycleLR")
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.lr,
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
            pct_start=0.3,
            div_factor=10.0,
            final_div_factor=1000.0
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")
    
    return scheduler

def mixup_data(x, y, alpha=1.0):
    """执行Mixup数据增强
    
    参数：
        x: 输入数据批次
        y: 标签批次
        alpha: mixup强度参数
    
    返回：
        混合后的输入和标签
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    """执行CutMix数据增强
    
    参数：
        x: 输入数据批次
        y: 标签批次
        alpha: cutmix强度参数
    
    返回：
        混合后的输入和标签
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    # 获取图像尺寸
    h, w = x.size()[-2:]
    
    # 计算裁剪区域的尺寸和位置
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    # 确保裁剪区域在图像内
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    
    # 创建混合图像
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # 调整lambda以反映实际混合比例
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup/CutMix的损失计算
    
    参数：
        criterion: 基础损失函数
        pred: 模型预测
        y_a: 第一个标签
        y_b: 第二个标签
        lam: 混合比例
    
    返回：
        混合后的损失
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def accuracy(output, target, topk=(1,)):
    """计算前k个预测的准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class Logger(object):
    """日志记录器"""
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        pass

def get_learning_rate(optimizer):
    """获取当前学习率"""
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    lr = lr[0]
    return lr

def time_to_str(t, mode='min'):
    """将时间转换为字符串"""
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hours %02d minutes'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d minutes %02d seconds'%(min,sec)
    else:
        raise NotImplementedError

class FocalLoss(nn.Module):
    """焦点损失函数"""
    def __init__(self, focusing_param=2, balance_param=0.25):
        super(FocalLoss, self).__init__()
        self.focusing_param = focusing_param
        self.balance_param = balance_param

    def forward(self, output, target):
        cross_entropy = F.cross_entropy(output, target)
        cross_entropy_log = torch.log(cross_entropy)
        logpt = - F.cross_entropy(output, target)
        pt    = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.focusing_param) * logpt
        balanced_focal_loss = self.balance_param * focal_loss

        return balanced_focal_loss

# 改进版焦点损失，支持标签平滑
class ImprovedFocalLoss(nn.Module):
    """改进版焦点损失，支持标签平滑"""
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.1):
        super(ImprovedFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        
    def forward(self, input, target):
        n_classes = input.size(1)
        target_one_hot = F.one_hot(target, n_classes).float()
        
        # 应用标签平滑
        if self.label_smoothing > 0:
            target_one_hot = (1 - self.label_smoothing) * target_one_hot + \
                             self.label_smoothing / n_classes
        
        # 计算焦点损失
        log_softmax = F.log_softmax(input, dim=1)
        softmax = torch.exp(log_softmax)
        loss = -1 * target_one_hot * log_softmax  # 交叉熵
        
        # 添加焦点权重
        weight = self.alpha * target_one_hot * (1 - softmax) ** self.gamma + \
                (1 - self.alpha) * (1 - target_one_hot) * softmax ** self.gamma
        
        weighted_loss = weight * loss
        return weighted_loss.sum(dim=1).mean()

class LabelSmoothingCrossEntropy(nn.Module):
    """带标签平滑的交叉熵损失"""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, input, target):
        n_classes = input.size(1)
        
        # 创建平滑标签
        log_softmax = F.log_softmax(input, dim=1)
        target_one_hot = F.one_hot(target, n_classes).float()
        target_smooth = (1 - self.smoothing) * target_one_hot + \
                        self.smoothing / n_classes
        
        # 计算损失
        loss = -1 * target_smooth * log_softmax
        return loss.sum(dim=1).mean()

def get_loss_function(device):
    """获取损失函数，根据配置选择合适的损失"""
    if config.use_focal_loss:
        if config.label_smoothing > 0:
            return ImprovedFocalLoss(
                gamma=2.0, 
                alpha=0.25, 
                label_smoothing=config.label_smoothing
            ).to(device)
        else:
            return FocalLoss(
                focusing_param=2.0, 
                balance_param=0.25
            ).to(device)
    else:
        if config.label_smoothing > 0:
            return LabelSmoothingCrossEntropy(
                smoothing=config.label_smoothing
            ).to(device)
        else:
            return nn.CrossEntropyLoss().to(device)

class MyEncoder(json.JSONEncoder):
    """JSON编码器，用于处理NumPy类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def evaluate(val_loader, model, criterion, device):
    """在验证集上评估模型"""
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    
    model.eval()
    
    with torch.no_grad():
        for input, target in val_loader:
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
                
                # 释放内存
                del output
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error in validation: {str(e)}")
                continue
                
    return losses.avg, top1.avg, top2.avg

def create_model_ema(model):
    """创建模型的指数移动平均版本"""
    if config.use_ema:
        return ModelEmaV2(
            model, 
            decay=config.ema_decay, 
            device='cpu' if config.device == 'cpu' else None
        )
    return None

def update_ema(ema, model, iter):
    """更新EMA模型"""
    if ema is not None:
        ema.update(model)

import torch
import torchvision
import torch.nn.functional as F 
from torch import nn
from config.config import config
import timm

def get_densenet169():
    """生成DenseNet169模型"""
    class DenseModel(nn.Module):
        def __init__(self, pretrained_model):
            super(DenseModel, self).__init__()
            self.features = pretrained_model.features
            self.classifier = nn.Sequential(
                nn.Linear(pretrained_model.classifier.in_features, config.num_classes),
                nn.Sigmoid()
            )
            
            self._initialize_weights()
            
        def _initialize_weights(self):
            """初始化模型权重"""
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

        def forward(self, x):
            """前向传播函数"""
            features = self.features(x)
            out = F.adaptive_avg_pool2d(features, (1, 1))
            out = torch.flatten(out, 1)
            out = self.classifier(out)
            return out

    return DenseModel(torchvision.models.densenet169(pretrained=True))

def get_efficientnet():
    """获取EfficientNet-B4模型，使用渐进式解冻技术"""
    model = torchvision.models.efficientnet_b4(pretrained=True)
    
    # 冻结大部分层
    for name, param in model.named_parameters():
        if 'features.8' not in name:  # 只训练最后几层
            param.requires_grad = False
    
    # 替换分类器
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, config.num_classes),
        nn.Sigmoid()
    )
    
    return model

def get_efficientnetv2():
    """获取EfficientNetV2-S模型，性能更好"""
    try:
        # 尝试加载预训练模型
        model = timm.create_model('efficientnetv2_s', pretrained=True)
        print("Successfully loaded pretrained efficientnetv2_s model")
    except (RuntimeError, ValueError) as e:
        # 如果加载失败，退回到随机初始化
        print(f"Failed to load pretrained weights: {str(e)}")
        print("Using randomly initialized efficientnetv2_s model instead")
        model = timm.create_model('efficientnetv2_s', pretrained=False)
    
    # 使用渐进式解冻，冻结前部分层
    total_layers = len(list(model.named_parameters()))
    freeze_ratio = 0.7  # 冻结前70%的层
    
    for i, (name, param) in enumerate(model.named_parameters()):
        if i < int(total_layers * freeze_ratio):
            param.requires_grad = False
    
    # 替换分类头
    feature_dim = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(feature_dim),
        nn.Dropout(0.3),
        nn.Linear(feature_dim, config.num_classes),
        nn.Sigmoid()
    )
    
    return model

def get_convnext():
    """获取ConvNeXt模型"""
    model = torchvision.models.convnext_small(pretrained=True)
    
    # 冻结早期层
    for name, param in model.named_parameters():
        if 'features.7' not in name:  # 只训练最后一个阶段
            param.requires_grad = False
    
    # 替换分类器
    in_features = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        nn.LayerNorm2d(in_features),
        nn.Flatten(1),
        nn.Linear(in_features, config.num_classes),
        nn.Sigmoid()
    )
    
    return model

def get_swin_transformer():
    """获取Swin Transformer模型，适用于细粒度分类任务"""
    model = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
    
    # 冻结早期层
    total_blocks = 4
    blocks_to_unfreeze = 1
    
    for name, param in model.named_parameters():
        if 'layers.3' in name or 'head' in name:  # 只训练最后一个block和头部
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # 替换分类头
    feature_dim = model.head.in_features
    model.head = nn.Sequential(
        nn.LayerNorm(feature_dim),
        nn.Linear(feature_dim, config.num_classes),
        nn.Sigmoid()
    )
    
    return model

def get_hybrid_model():
    """创建混合模型（CNN+Transformer），结合卷积网络的局部特征和Transformer的全局特征"""
    class HybridModel(nn.Module):
        def __init__(self):
            super(HybridModel, self).__init__()
            # CNN部分: 使用EfficientNet提取特征
            self.cnn_model = timm.create_model('efficientnet_b3', pretrained=True, features_only=True)
            
            # 冻结CNN早期层
            for name, param in self.cnn_model.named_parameters():
                if 'blocks.5' not in name and 'blocks.6' not in name:  # 只训练最后两个块
                    param.requires_grad = False
            
            # Transformer部分: 使用轻量级Transformer
            cnn_channels = self.cnn_model.feature_info.channels()[-1]  # 获取最后一层特征图的通道数
            self.transformer = timm.create_model(
                'vit_small_patch16_224', 
                pretrained=True,
                img_size=16,  # 特征图大小
                patch_size=1,  # 使用1x1的patch
                in_chans=cnn_channels,  # 输入通道数为CNN输出的通道数
                num_classes=0  # 不使用分类头
            )
            
            # 冻结Transformer早期层
            for name, param in self.transformer.named_parameters():
                if 'blocks.10' not in name and 'blocks.11' not in name and 'norm' not in name:
                    param.requires_grad = False
            
            transformer_dim = self.transformer.embed_dim
            
            # 分类头
            self.classifier = nn.Sequential(
                nn.LayerNorm(transformer_dim),
                nn.Dropout(0.3),
                nn.Linear(transformer_dim, config.num_classes),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            # 通过CNN提取特征
            features = self.cnn_model(x)[-1]  # 获取最后一层特征图
            
            # 通过Transformer处理特征
            transformer_out = self.transformer.forward_features(features)
            
            # 分类
            out = self.classifier(transformer_out)
            return out
    
    return HybridModel()

def get_ensemble_model():
    """创建模型集成，结合多个模型的优势"""
    class EnsembleModel(nn.Module):
        def __init__(self):
            super(EnsembleModel, self).__init__()
            
            # 加载多个预训练模型
            self.model1 = get_efficientnetv2()  # EfficientNetV2
            self.model2 = get_convnext()  # ConvNeXt
            
            # 确保这些模型的输出层是一致的
            # 集成层 - 使用注意力机制进行加权
            self.attention = nn.Sequential(
                nn.Linear(config.num_classes * 2, 2),
                nn.Softmax(dim=1)
            )
            
        def forward(self, x):
            # 获取各个模型的输出
            out1 = self.model1(x)
            out2 = self.model2(x)
            
            # 拼接输出
            combined = torch.cat((out1, out2), dim=1)
            
            # 计算注意力权重
            weights = self.attention(combined)
            
            # 加权合并
            out = weights[:, 0:1] * out1 + weights[:, 1:2] * out2
            
            return out
    
    return EnsembleModel()

def get_net():
    """选择并返回模型
    
    可以在此选择哪个模型用于训练
    """
    models = {
        "densenet169": get_densenet169,
        "efficientnet_b4": get_efficientnet,
        "efficientnetv2_s": get_efficientnetv2,
        "convnext_small": get_convnext,
        "swin_transformer": get_swin_transformer,
        "hybrid_model": get_hybrid_model,
        "ensemble_model": get_ensemble_model
    }
    
    if config.model_name in models:
        print(f"Using model: {config.model_name}")
        return models[config.model_name]()
    else:
        print(f"Model {config.model_name} not found, using default EfficientNetV2")
        return get_efficientnetv2()


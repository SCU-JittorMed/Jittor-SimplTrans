import jittor as jt
import jittor.nn as nn
from jittor import transform
from jittor.dataset.mnist import MNIST
from jittor.optim import Adam
from jittor.lr_scheduler import CosineAnnealingLR
from loguru import logger
from tqdm import tqdm

from jittorsimpletrans.models import ViT, SimplTrans
from jittorsimpletrans.examples.utils import get_transforms

class Config:
    epoch = 100
    batchsize = 256
    num_workers = 8
    log_name = "mnist_experiment.log"
    best_acc = 0
    best_epoch = 0
    alpha = 0.001
    mode = "vit"  # Options: "vit", "simpltrans" (with modes: "random", "diagonal", "identity")
    
    # Model parameters
    image_size = 28
    patch_size = 4
    num_classes = 10
    dim = 48
    depth = 12
    heads = 3
    mlp_dim = 192
    pool = "cls"
    channels = 3     # Jittor loads MNIST as 3 channels
    dim_head = 16
    dropout = 0.1
    emb_dropout = 0.1
    
    # Device config
    jt.flags.use_cuda = jt.has_cuda

def train_epoch(model, train_loader, criterion, optimizer, config):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # 手动计算批次数 - MNIST训练集大小为60000
    mnist_train_size = 60000
    num_batches = (mnist_train_size + config.batchsize - 1) // config.batchsize
    print(f"\n开始训练 - 共{num_batches}个批次")
    
    for batch_idx, data in enumerate(train_loader):
        # 提取输入和目标
        inputs, targets = data
        
        # 处理目标是元组的情况
        if isinstance(targets, tuple):
            targets = targets[0]
        
        # 处理one-hot编码的目标
        if hasattr(targets, 'shape') and len(targets.shape) > 1 and targets.shape[1] > 1:
            targets = targets.argmax(dim=1)
        
        # 前向传递
        outputs = model(inputs)
        
        # 处理输出是元组的情况
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # 计算损失并优化
        loss = criterion(outputs, targets)
        optimizer.step(loss)
        
        # 获取预测类别
        predicted_class = outputs.argmax(dim=1)
        
        # 处理预测类别是元组的情况
        if isinstance(predicted_class, tuple):
            predicted_class = predicted_class[0]
                
        # 确保预测类别是Jittor变量
        if not isinstance(predicted_class, jt.Var):
            predicted_class = jt.array(predicted_class)
            
        total += targets.shape[0]
        # 使用Jittor的equal方法而非==运算符
        correct += (predicted_class.equal(targets)).sum().item()
        
        total_loss += loss.item()
        
        # 每10个批次显示一次进度
        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            print(f"训练进度: {batch_idx+1}/{num_batches} 批次 | 损失: {total_loss/(batch_idx+1):.4f} | 准确率: {100.0*correct/total:.2f}%")
    
    return total_loss / len(train_loader), 100. * correct / total

def test_epoch(model, test_loader, criterion, config):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # 手动计算批次数 - MNIST测试集大小为10000
    mnist_test_size = 10000
    num_batches = (mnist_test_size + config.batchsize - 1) // config.batchsize
    print(f"\n开始评估 - 共{num_batches}个批次")
    
    with jt.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # 提取输入和目标
            inputs, targets = data
            
            # 处理目标是元组的情况
            if isinstance(targets, tuple):
                targets = targets[0]
            
            # 前向传递
            outputs = model(inputs)
            
            # 处理输出是元组的情况
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            # 计算损失
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # 计算准确率
            predicted_class = outputs.argmax(1)
            
            # 处理预测类别是元组的情况
            if isinstance(predicted_class, tuple):
                predicted_class = predicted_class[0]
                    
            # 确保预测类别是Jittor变量
            if not isinstance(predicted_class, jt.Var):
                predicted_class = jt.array(predicted_class)
                
            total += targets.shape[0]
            # 使用Jittor的equal方法而非==运算符
            correct += (predicted_class.equal(targets)).sum().item()
    
    return total_loss / len(test_loader), 100. * correct / total

def main():
    # Initialize Jittor - enable CPU mode if CUDA not available
    if not jt.has_cuda:
        jt.flags.use_cuda = 0
    
    config = Config()
    logger.add(config.log_name)
    
    # Set random seed for reproducibility
    jt.set_seed(42)
    
    # Prepare dataset
    train_transform, test_transform = get_transforms()
    train_dataset = MNIST(train=True, transform=train_transform)
    test_dataset = MNIST(train=False, transform=test_transform)
    
    train_loader = train_dataset.set_attrs(batch_size=config.batchsize, shuffle=True, num_workers=config.num_workers)
    test_loader = test_dataset.set_attrs(batch_size=config.batchsize, shuffle=False, num_workers=config.num_workers)
    
    # Create model
    if config.mode == 'vit':
        model = ViT(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_classes=config.num_classes,
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            mlp_dim=config.mlp_dim,
            pool=config.pool,
            channels=config.channels,
            dim_head=config.dim_head,
            dropout=config.dropout,
            emb_dropout=config.emb_dropout
        )
    else:
        # Use mode as part of simpltrans with "random", "diagonal", or "identity"
        model = SimplTrans(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_classes=config.num_classes,
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            mlp_dim=config.mlp_dim,
            pool=config.pool,
            channels=config.channels,
            dim_head=config.dim_head,
            dropout=config.dropout,
            emb_dropout=config.emb_dropout,
            mode=config.mode
        )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.alpha)
    scheduler = CosineAnnealingLR(optimizer, config.epoch)
    
    logger.info(f"Starting training with mode: {config.mode}")
    
    for epoch in range(config.epoch):
        print(f"\n{'='*20} Epoch: {epoch+1}/{config.epoch} {'='*20}")
        logger.info(f"\n=== Epoch: {epoch+1}/{config.epoch} ===")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config)
        test_loss, test_acc = test_epoch(model, test_loader, criterion, config)
        
        scheduler.step()
        
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        logger.info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
        if test_acc > config.best_acc:
            config.best_acc = test_acc
            config.best_epoch = epoch
            logger.info(f"New best accuracy: {test_acc:.2f}%")
            # Save best model
            jt.save(model.state_dict(), f"best_model_{config.mode}.pkl")
    
    logger.info(f"Best accuracy: {config.best_acc:.2f}% at epoch {config.best_epoch+1}")

if __name__ == "__main__":
    main()

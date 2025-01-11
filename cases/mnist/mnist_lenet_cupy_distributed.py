import os
import sys
import json
import time

# 获取项目根目录并添加到 Python 路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

import cupy as cp
import numpy as np
import swanlab

from mytorch.ops import Max as mymax
from mytorch.tensor import Tensor, no_grad
from mytorch.dataset import MNISTDataset
from mytorch.dataloader import DataLoader, prepare_mnist_data
import mytorch.module as nn
from mytorch.module import Module, Linear, Conv2D, MaxPooling2D
import mytorch.functions as F
from mytorch.functions import relu, softmax
from mytorch.optim import Adam, Adagrad
from mytorch.loss import CrossEntropyLoss, NLLLoss
from mytorch import cuda
import argparse
from mytorch.distributed import RingAllReduce


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 第一个卷积层: 1个输入通道, 6个输出通道, 5x5的卷积核
        self.conv1 = Conv2D(1, 6, (5, 5))
        # 第一个池化层: 2x2
        self.pool1 = MaxPooling2D(2, 2, 2)
        # 第二个卷积层: 6个输入通道, 16个输出通道, 5x5的卷积核
        self.conv2 = Conv2D(6, 16, (5, 5))
        # 第二个池化层: 2x2
        self.pool2 = MaxPooling2D(2, 2, 2)
        # 三个全连接层
        self.fc1 = Linear(16 * 4 * 4, 120)
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x)
        return x


def partition_dataset(dataset, world_size, rank):
    """将数据集划分为多个部分用于分布式训练
    
    Args:
        dataset: 要划分的数据集
        world_size (int): 总的进程数量
        rank (int): 当前进程的序号（从0开始）
        
    Returns:
        Dataset: 划分后的数据集子集
    """
    # 计算每个进程应该分到的数据量
    total_size = len(dataset)
    base_size = total_size // world_size
    remainder = total_size % world_size
    
    # 如果有余数，最后一个进程会多分到一些数据
    if rank == world_size - 1:
        partition_size = base_size + remainder
    else:
        partition_size = base_size
    
    # 计算当前进程的数据起始索引
    start_idx = rank * base_size
    end_idx = start_idx + partition_size
    
    # 创建新的数据集
    if hasattr(dataset, 'data') and hasattr(dataset, 'targets'):
        # 对于MNIST等标准数据集
        dataset.data = dataset.data[start_idx:end_idx]
        dataset.targets = dataset.targets[start_idx:end_idx]
    else:
        # 对于自定义数据集，可能需要其他处理方式
        raise NotImplementedError("当前只支持包含data和targets属性的数据集")
    
    return dataset

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--nodes', type=str, required=True)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    # 解析节点配置
    nodes = []
    for node in args.nodes.split(','):
        ip, port = node.split(':')
        nodes.append((ip, int(port)))

    # 定义基本配置参数
    config = {
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "device": "cuda" if cuda.is_available() else "cpu",
        "world_size": args.world_size
    }

    # 只在主进程(rank 0)初始化 SwanLab
    if args.rank == 0:
        run = swanlab.init(
            project="MNIST-LeNet-Distributed",
            experiment_name="MNIST-LeNet-cupy-distributed",
            config=config
        )

    # 初始化分布式环境
    distributed = RingAllReduce(args.rank, args.world_size, nodes)

    # 获取项目根目录
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(root_dir, 'data')

    # 加载和准备数据
    train_dataset = prepare_mnist_data(root=data_dir, backend='cupy', train=True)
    # 划分训练数据集
    train_dataset = partition_dataset(train_dataset, args.world_size, args.rank)
    
    test_dataset = prepare_mnist_data(root=data_dir, backend='cupy', train=False)

    # 修改 DataLoader 的 batch_size 参数类型
    batch_size = int(config["batch_size"])  # 确保 batch_size 是整数
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = LeNet()
    device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")
    model.to(device)

    criterion = NLLLoss()
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])

    def train(epoch):
        model.train()
        total_samples = 0
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start_time = time.time()
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            batch_start_time = time.time()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 同步梯度
            for param in model.parameters():
                if param.grad is not None:
                    grad_tensor = Tensor(param.grad)
                    synced_grad = distributed.allreduce(grad_tensor)
                    cp.copyto(param.grad, synced_grad.data / args.world_size)
            
            optimizer.step()
            
            # 计算准确率
            predicted = mymax().forward(outputs.data, axis=1)
            predicted = cp.rint(predicted)
            total += labels.array().size
            correct += (predicted == labels.array()).sum().item()
            accuracy = 100 * correct / total
            
            # 更新统计信息
            running_loss += loss.item()
            total_samples += inputs.shape[0]
            batch_time = time.time() - batch_start_time
            samples_per_second = inputs.shape[0] / batch_time
            
            if args.rank == 0 and batch_idx % 10 == 0:  # rank 0每10个批次输出一次
                avg_loss = running_loss / (batch_idx + 1)
                progress = {
                    "rank": args.rank,
                    "epoch": epoch + 1,
                    "batch": batch_idx + 1,
                    "total_batches": len(train_loader),
                    "samples_processed": total_samples,
                    "accuracy": accuracy,
                    "loss": avg_loss,
                    "samples_per_second": samples_per_second,
                    "active_nodes": args.world_size
                }
                
                # 打印训练信息到终端
                print(f"\r【Epoch {epoch + 1}】Batch [{batch_idx + 1}/{len(train_loader)}] - "
                      f"Loss: [{avg_loss:.4f}] - Accuracy: [{accuracy:.2f}%] - "
                      f"Speed: [{samples_per_second:.1f} samples/s]", end="")
                
                # 将进度信息写入状态文件
                status_file = os.path.join(root_dir, 'cases', 'web', 'static', 'training_status.json')
                with open(status_file, 'w') as f:
                    json.dump(progress, f)
            
            elif args.rank > 0 and batch_idx % 10 == 0:  # 其他rank也打印信息
                avg_loss = running_loss / (batch_idx + 1)
                print(f"\rRank {args.rank} - Epoch {epoch + 1} - "
                      f"Batch [{batch_idx + 1}/{len(train_loader)}] - "
                      f"Loss: [{avg_loss:.4f}] - Accuracy: [{accuracy:.2f}%]", end="")

        # 每个epoch结束打印汇总信息
        epoch_avg_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        epoch_time = time.time() - epoch_start_time
        avg_speed = total_samples / epoch_time
        
        print(f"\nRank {args.rank} - Epoch {epoch + 1} Summary:")
        print(f"Average Loss: {epoch_avg_loss:.4f}")
        print(f"Accuracy: {epoch_accuracy:.2f}%")
        print(f"Average Speed: {avg_speed:.1f} samples/s")
        print(f"Time Used: {epoch_time:.2f}s")
        print("-" * 50)

        # 在主进程中记录训练指标
        if args.rank == 0:
            run.log({
                "train_loss": epoch_avg_loss,
                "train_accuracy": epoch_accuracy,
                "training_speed": avg_speed
            })

    def test():
        model.eval()
        correct = 0
        total = 0
        total_loss = 0
        total_batches = len(test_loader)
        print(f"\n【Test {epoch + 1}】")
        with no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                predicted = mymax().forward(outputs.data, axis=1)
                predicted = cp.rint(predicted)

                total_loss += loss.item()
                total += labels.array().size
                correct += (predicted == labels.array()).sum().item()

        accuracy = 100 * correct / total
        avg_loss = total_loss / total_batches
        print(f'Rank {args.rank} Accuracy on test set: {accuracy:.2f}%')

        # 在主进程中记录测试指标
        if args.rank == 0:
            run.log({
                "test_loss": avg_loss,
                "test_accuracy": accuracy
            })

    # 训练循环
    try:
        for epoch in range(config["num_epochs"]):  # 使用配置的轮数
            train(epoch)
            if args.rank == 0:  # 只在主进程上测试
                test()
    except Exception as e:
        print(f"Rank {args.rank} encountered error: {str(e)}")
    finally:
        print(f"Rank {args.rank} training completed")
        if args.rank == 0:
            run.finish()

if __name__ == '__main__':
    main()

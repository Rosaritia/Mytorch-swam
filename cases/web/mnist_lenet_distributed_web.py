from flask import Flask, render_template, jsonify, request, send_file
import threading
import os
import sys
import graphviz
import time
import logging
from openai import OpenAI
import swanlab
import cupy as cp
import subprocess
import json

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

from mytorch.dataloader import prepare_mnist_data, DataLoader
from mytorch.loss import NLLLoss
from mytorch.optim import Adam, SGD, Adagrad
from mytorch.module import Conv2D, Linear, MaxPooling2D, Module
import mytorch.functions as F
from mytorch import cuda

# 添加SwanLab配置
# 初始化 SwanLab
run = swanlab.init(
    logdir='./logs',
    mode="local",
    project="MNIST-LeNet-Distributed",
    experiment_name="MNIST-LeNet-Distributed-Web",
    config={
        "optimizer": "Adam",
        "learning_rate": 0.01,
        "batch_size": 64,
        "num_epochs": 10,
        "device": "cuda" if cuda.is_available() else "cpu",
        "world_size": 2  # 默认分布式节点数
    },
)

# 初始化Flask应用
template_dir = os.path.join(current_dir, 'templates')
static_dir = os.path.join(current_dir, 'static')
os.makedirs(template_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# 全局变量
training_processes = []
is_training = False
model = None
current_epoch = 0
current_batch = 0
total_batches = 0
samples_processed = 0
latest_accuracy = 0.0
latest_loss = 0.0
training_start_time = 0
training_logs = []

# 分布式训练配置
distributed_config = {
    'world_size': 2,  # 默认2个节点
    'nodes': [
        {'ip': '127.0.0.1', 'port': 29500},
        {'ip': '127.0.0.1', 'port': 29501}
    ]
}

# 训练参数
training_params = {
    'learning_rate': 0.01,
    'batch_size': 64,
    'epochs': 10,
    'optimizer': 'Adam',
    'log_interval': 300
}

# 网络结构参数
network_params = {
    'conv1_out_channels': 6,
    'conv1_kernel_size': 5,
    'conv2_out_channels': 16,
    'conv2_kernel_size': 5,
    'fc1_out_features': 120,
    'fc2_out_features': 84,
    'pool_size': 2,
    'pool_stride': 2
}

# 设置日志级别
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

@app.route('/')
def home():
    return render_template('train_distributed.html')

@app.route('/get_distributed_config')
def get_distributed_config():
    return jsonify(distributed_config)

@app.route('/update_distributed_config', methods=['POST'])
def update_distributed_config():
    global distributed_config
    if is_training:
        return jsonify({'status': 'error', 'message': '训练进行中，无法更新配置'})
    
    data = request.get_json()
    
    # 验证节点数量
    world_size = data.get('world_size')
    if not (2 <= world_size <= 5):
        return jsonify({'status': 'error', 'message': '节点数量必须在2-5之间'})
    
    # 验证节点配置
    nodes = data.get('nodes', [])
    if len(nodes) != world_size:
        return jsonify({'status': 'error', 'message': '节点配置数量必须与world_size相匹配'})
    
    # 验证每个节点的配置
    for node in nodes:
        if 'ip' not in node or 'port' not in node:
            return jsonify({'status': 'error', 'message': '节点配置必须包含ip和port'})
        try:
            port = int(node['port'])
            if not (1024 <= port <= 65535):
                return jsonify({'status': 'error', 'message': '端口号必须在1024-65535之间'})
        except ValueError:
            return jsonify({'status': 'error', 'message': '端口号必须是有效的整数'})
    
    # 更新配置
    distributed_config.update(data)
    # 更新SwanLab配置
    run.config.update({
        "world_size": world_size
    })
    return jsonify({'status': 'success', 'message': '分布式配置更新成功'})

def start_distributed_training():
    global training_processes, is_training
    
    try:
        # 准备训练脚本路径
        script_path = os.path.join(root_dir, 'cases', 'mnist', 'mnist_lenet_cupy_distributed.py')
        
        # 构建节点配置字符串
        nodes_config = []
        for node in distributed_config['nodes']:
            nodes_config.append(f"{node['ip']}:{node['port']}")
        nodes_str = ','.join(nodes_config)
        
        # 为每个rank启动一个新的终端进程
        for rank in range(distributed_config['world_size']):
            # 获取当前节点的配置
            node = distributed_config['nodes'][rank]
            
            # Windows系统
            if os.name == 'nt':
                cmd = [
                    'start',
                    'cmd',
                    '/k',  # 保持终端窗口打开
                    sys.executable,
                    script_path,
                    '--rank', str(rank),
                    '--world-size', str(distributed_config['world_size']),
                    '--nodes', nodes_str,
                    '--learning-rate', str(training_params['learning_rate']),
                    '--batch-size', str(training_params['batch_size']),
                    '--epochs', str(training_params['epochs'])
                ]
                # 使用os.system来启动新终端
                os.system(' '.join(cmd))
            
            # Linux/Mac系统
            else:
                cmd = [
                    'gnome-terminal',  # Linux下的终端
                    '--',  # 后面的是命令
                    sys.executable,
                    script_path,
                    '--rank', str(rank),
                    '--world-size', str(distributed_config['world_size']),
                    '--nodes', nodes_str,
                    '--learning-rate', str(training_params['learning_rate']),
                    '--batch-size', str(training_params['batch_size']),
                    '--epochs', str(training_params['epochs'])
                ]
                # 使用subprocess启动新终端
                subprocess.Popen(cmd)

            print(f"已启动rank {rank}的训练进程，IP: {node['ip']}, Port: {node['port']}")
        
        return True
    except Exception as e:
        print(f"启动分布式训练失败: {str(e)}")
        return False

@app.route('/start_training')
def start_training():
    global is_training, training_processes
    
    if not is_training:
        is_training = True
        
        # 启动分布式训练
        if start_distributed_training():
            return jsonify({'status': 'success', 'message': '分布式训练已开始'})
        else:
            is_training = False
            return jsonify({'status': 'error', 'message': '启动分布式训练失败'})
    
    return jsonify({'status': 'error', 'message': '训练已在进行中'})

@app.route('/stop_training')
def stop_training():
    global is_training, training_processes
    
    is_training = False
    
    # 终止所有训练进程
    for process in training_processes:
        try:
            process.terminate()
        except Exception as e:
            print(f"终止进程时出错: {str(e)}")
    
    training_processes = []
    # 完成SwanLab实验
    run.finish()
    return jsonify({'status': 'success', 'message': '训练已停止'})

@app.route('/get_status')
def get_status():
    status_file = os.path.join(static_dir, 'training_status.json')
    training_metrics = {
        'current_epoch': 0,
        'current_batch': 0,
        'total_batches': 0,
        'samples_processed': 0,
        'accuracy': 0.0,
        'loss': 0.0,
        'samples_per_second': 0.0,
        'active_nodes': 0
    }
    
    try:
        if os.path.exists(status_file):
            # 读取状态文件
            with open(status_file, 'r') as f:
                progress = json.load(f)
                training_metrics.update({
                    'current_epoch': progress['epoch'],
                    'current_batch': progress['batch'],
                    'total_batches': progress['total_batches'],
                    'samples_processed': progress['samples_processed'],
                    'accuracy': progress['accuracy'],
                    'loss': progress['loss'],
                    'samples_per_second': progress['samples_per_second'],
                    'active_nodes': progress['active_nodes']
                })
                
                # 记录到SwanLab
                run.log({
                    "loss": progress['loss'],
                    "accuracy": progress['accuracy'],
                    "samples_per_second": progress['samples_per_second']
                })
    except Exception as e:
        print(f"读取训练状态失败: {str(e)}")
    
    # 检查进程状态
    processes_status = []
    for i in range(distributed_config['world_size']):
        # 通过检查进程是否存在来判断节点状态
        if os.name == 'nt':  # Windows
            cmd = f'tasklist /FI "WINDOWTITLE eq Python*rank {i}*" /NH'
        else:  # Linux/Mac
            cmd = f'ps aux | grep "python.*--rank {i}" | grep -v grep'
            
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.stdout.strip():
                processes_status.append({
                    'rank': i,
                    'status': 'running'
                })
            else:
                processes_status.append({
                    'rank': i,
                    'status': 'stopped'
                })
        except Exception as e:
            processes_status.append({
                'rank': i,
                'status': 'error',
                'message': str(e)
            })

    return jsonify({
        'is_training': is_training,
        'processes': processes_status,
        'world_size': distributed_config['world_size'],
        'nodes': distributed_config['nodes'],
        'current_epoch': training_metrics['current_epoch'],
        'current_batch': training_metrics['current_batch'],
        'total_batches': training_metrics['total_batches'],
        'samples_processed': training_metrics['samples_processed'],
        'accuracy': training_metrics['accuracy'],
        'loss': training_metrics['loss'],
        'samples_per_second': training_metrics['samples_per_second'],
        'active_nodes': training_metrics['active_nodes'],
        'total_epochs': training_params['epochs']
    })

@app.route('/get_training_params')
def get_training_params():
    return jsonify(training_params)

@app.route('/update_training_params', methods=['POST'])
def update_training_params():
    global training_params
    if is_training:
        return jsonify({'status': 'error', 'message': '训练进行中，无法更新参数'})
    
    data = request.get_json()
    
    # 验证参数
    if not (0 < data['learning_rate'] <= 1):
        return jsonify({'status': 'error', 'message': '学习率必须在0-1之间'})
    if not (1 <= data['batch_size'] <= 512):
        return jsonify({'status': 'error', 'message': 'Batch Size必须在1-512之间'})
    if not (1 <= data['epochs'] <= 100):
        return jsonify({'status': 'error', 'message': '训练轮数必须在1-100之间'})
    if data['optimizer'] not in ['Adam', 'SGD', 'Adagrad']:
        return jsonify({'status': 'error', 'message': '不支持的优化器类型'})
    
    training_params.update(data)
    # 更新SwanLab配置
    run.config.update({
        "optimizer": data['optimizer'],
        "learning_rate": data['learning_rate'],
        "batch_size": data['batch_size'],
        "num_epochs": data['epochs']
    })
    return jsonify({'status': 'success', 'message': '训练参数更新成功'})

if __name__ == '__main__':

    print(f"【分布式训练控制面板启动成功！】开始训练请访问: http://127.0.0.1:7700")

    app.run(host='0.0.0.0', port=7700, debug=False)

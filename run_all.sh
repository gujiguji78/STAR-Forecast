#!/bin/bash

# STAR-Forecast 完整运行脚本
# 作者：梁德隆

set -e  # 遇到错误立即退出

echo "🚀 STAR-Forecast 启动"
echo "========================"

# 检查Python版本
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python版本: $python_version"

if [[ $(echo "$python_version < 3.8" | bc) -eq 1 ]]; then
    echo "❌ 需要Python 3.8或更高版本"
    exit 1
fi

# 创建目录结构
echo "📁 创建目录结构..."
mkdir -p data logs checkpoints results cache

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "🔧 创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "📦 安装依赖包..."
pip install --upgrade pip
pip install -r requirements.txt

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 检测到NVIDIA GPU"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
    echo "⚠️  未检测到GPU，将使用CPU"
fi

# 检查数据文件
if [ ! -f "data/ETTh1.csv" ]; then
    echo "📥 下载ETTh1数据集..."
    wget -q https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv -O data/ETTh1.csv

    if [ $? -eq 0 ]; then
        echo "✅ 数据集下载完成"
        echo "   文件大小: $(du -h data/ETTh1.csv | cut -f1)"
        echo "   样本数量: $(wc -l data/ETTh1.csv | cut -d' ' -f1)"
    else
        echo "❌ 数据集下载失败"
        echo "   请手动下载: https://github.com/zhouhaoyi/ETDataset"
        exit 1
    fi
else
    echo "✅ 数据集已存在"
fi

# 检查环境变量
echo "🔧 检查环境变量..."
if [ ! -f ".env" ]; then
    echo "⚠️  .env文件不存在，创建示例配置"
    cp .env.example .env
    echo "   请编辑.env文件配置API密钥"
    exit 1
fi

# 加载环境变量
export $(grep -v '^#' .env | xargs)

# 检查API密钥
if [ -z "$DEEPSEEK_API_KEY" ] || [ "$DEEPSEEK_API_KEY" = "sk-your-deepseek-api-key-here" ]; then
    echo "⚠️  未配置DeepSeek API密钥"
fi

if [ -z "$QWEN_API_KEY" ] || [ "$QWEN_API_KEY" = "sk-your-qwen-api-key-here" ]; then
    echo "⚠️  未配置Qwen API密钥"
fi

# 运行环境检查
echo "🔍 运行环境检查..."
python3 scripts/check_env.py

# 运行预处理
echo "🔧 运行数据预处理..."
python3 scripts/preprocess.py

# 运行训练
echo "🏋️ 开始模型训练..."
python3 train.py \
    --config config.yaml \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --device cuda:0 \
    --log-dir logs \
    --checkpoint-dir checkpoints

# 运行评估
echo "🧪 运行模型评估..."
python3 evaluate.py \
    --config config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --output results/predictions.npy

# 生成报告
echo "📊 生成实验报告..."
python3 scripts/generate_report.py \
    --results results \
    --output report.html

echo "✅ 所有任务完成！"
echo "========================"
echo "📁 结果文件:"
echo "   - 日志: logs/"
echo "   - 模型: checkpoints/"
echo "   - 预测: results/predictions.npy"
echo "   - 报告: report.html"
echo "========================"
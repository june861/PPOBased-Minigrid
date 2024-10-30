#!/bin/bash

# 定义可修改的参数
ALGO="ppo"
FRAMES=20000000
PYENV="py310"
ENVS=(
    "MiniGrid-SimpleCrossingS11N5-v0" 
    "MiniGrid-DistShift2-v0"
)
MODEL_PREFIX="MiniGrid_"

# 日志文件基础路径
LOG_DIR="/home/june/minigrid_logs"

# 切换目录
PWD="/home/june/babyai-repo/rl-starter-files"
cd $PWD

# 启动虚拟环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $PYENV

# 遍历每个环境变量
for ENV in "${ENVS[@]}"; do
    # 构建模型名称
    MODEL="${ENV}_baseline"
    ACTION_DIST_MODEL="${ENV}_act_dist"

    # 构建baseline日志文件名
    BASE_LOG_FILE="${LOG_DIR}/${MODEL}_baseline.log"
    # 构建使用动作分布的日志文件名
    ACTION_DIST_LOG_FILE="${LOG_DIR}/${MODEL}_action_dist.log"
    
    # 启动第一个训练命令
    echo "Training started for environment $ENV without action distribution with the following parameters:"
    echo "Algorithm: $ALGO"
    echo "Environment: $ENV"
    echo "Frames: $FRAMES"
    echo "Model: $MODEL"
    echo "Logs will be saved to: $BASE_LOG_FILE"
    nohup python3 -m scripts.train --algo $ALGO --env $ENV --frames $FRAMES --model $MODEL >> $BASE_LOG_FILE 2>&1 &

    echo "Training started for environment $ENV with action distribution with the following parameters:"
    echo "Algorithm: $ALGO"
    echo "Environment: $ENV"
    echo "Frames: $FRAMES"
    echo "Model: $MODEL"
    echo "Action Distribution Enabled"
    echo "Logs will be saved to: $ACTION_DIST_LOG_FILE"
    # 启动第二个训练命令（包含 --use_action_dist）
    nohup python3 -m scripts.train --algo $ALGO --env $ENV --frames $FRAMES --model $ACTION_DIST_MODEL --use_action_dist >> $ACTION_DIST_LOG_FILE 2>&1 &

done
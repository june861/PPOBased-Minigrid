#!/bin/bash

# 定义可修改的参数
ALGO="ppo"
FRAMES=20000000
PYENV="py310"
ENVS=(
    # "MiniGrid-SimpleCrossingS11N5-v0" 
    # "MiniGrid-DistShift2-v0"
    # "BabyAI-BossLevel-v0"
    "BabyAI-KeyCorridorS6R3-v0"
)
MODEL_PREFIX="MiniGrid_"
NUM_RUNS=1
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
    # baseline只跑一个
    BASE_MODEL="${ENV}_baseline"
    BASE_LOG_FILE="${LOG_DIR}/${MODEL}_baseline.log"
    # 启动第一个训练命令
    echo "Training started for environment $ENV without action distribution with the following parameters:"
    echo "Algorithm: $ALGO"
    echo "Environment: $ENV"
    echo "Frames: $FRAMES"
    echo "Model: $MODEL"
    echo "Logs will be saved to: $BASE_LOG_FILE"
    nohup python3 -m scripts.train --algo $ALGO --env $ENV --frames $FRAMES --model $BASE_MODEL >> $BASE_LOG_FILE 2>&1 &

    for ((i=1; i<=NUM_RUNS; i++)); do
        # 采样全部动作的
        ALL_ACTION_MODEL_SURR4="${ENV}_seed${i}_all_act_4surr"
        ALL_ACTION_LOG_FILE_SURR4="${LOG_DIR}/${MODEL}_all_act_4surr.log"

        echo "Training started for environment $ENV with action distribution with the following parameters:"
        echo "Algorithm: $ALGO"
        echo "Environment: $ENV"
        echo "Frames: $FRAMES"
        echo "Model: $ALL_ACTION_MODEL_SURR4"
        echo "Action Distribution Enabled"
        echo "Logs will be saved to: $ALL_ACTION_LOG_FILE_SURR4"
        # 启动第二个训练命令（包含 --use_action_dist）
        nohup python3 -m scripts.train --algo $ALGO --env $ENV --frames $FRAMES --model $ALL_ACTION_MODEL_SURR4 \
         --use_action_dist --seed $i --sample_all_act  --use_surr >> $ALL_ACTION_LOG_FILE_SURR4 2>&1 &

        ALL_ACTION_MODEL="${ENV}_seed${i}_all_act"
        ALL_ACTION_LOG_FILE="${LOG_DIR}/${MODEL}_all_act.log"
        echo "Training started for environment $ENV with action distribution with the following parameters:"
        echo "Algorithm: $ALGO"
        echo "Environment: $ENV"
        echo "Frames: $FRAMES"
        echo "Model: $ALL_ACTION_MODEL"
        echo "Action Distribution Enabled"
        echo "Logs will be saved to: $ALL_ACTION_LOG_FILE"
        # 启动第二个训练命令（包含 --use_action_dist）
        nohup python3 -m scripts.train --algo $ALGO --env $ENV --frames $FRAMES --model $ALL_ACTION_MODEL \
         --use_action_dist --seed $i --sample_all_act >> $ALL_ACTION_LOG_FILE 2>&1 &


        # 只采样两个动作的
        TWO_ACTION_MODEL_SURR4="${ENV}_seed${i}_two_act_4surr"
        TWO_ACTION_LOG_FILE_SURR4="${LOG_DIR}/${MODEL}_two_act_4surr.log"

        echo "Training started for environment $ENV with action distribution with the following parameters:"
        echo "Algorithm: $ALGO"
        echo "Environment: $ENV"
        echo "Frames: $FRAMES"
        echo "Model: $TWO_ACTION_MODEL_SURR4"
        echo "Action Distribution Enabled"
        echo "Logs will be saved to: $TWO_ACTION_LOG_FILE_SURR4"
        # 启动第二个训练命令（包含 --use_action_dist）
        nohup python3 -m scripts.train --algo $ALGO --env $ENV --frames $FRAMES --model $TWO_ACTION_MODEL_SURR4 \
        --use_action_dist --seed $i --use_surr >> $TWO_ACTION_LOG_FILE_SURR4 2>&1 &

        TWO_ACTION_MODEL="${ENV}_seed${i}_two_act"
        TWO_ACTION_LOG_FILE="${LOG_DIR}/${MODEL}_two_act.log"

        echo "Training started for environment $ENV with action distribution with the following parameters:"
        echo "Algorithm: $ALGO"
        echo "Environment: $ENV"
        echo "Frames: $FRAMES"
        echo "Model: $TWO_ACTION_MODEL"
        echo "Action Distribution Enabled"
        echo "Logs will be saved to: $TWO_ACTION_LOG_FILE"
        # 启动第二个训练命令（包含 --use_action_dist）
        nohup python3 -m scripts.train --algo $ALGO --env $ENV --frames $FRAMES --model $TWO_ACTION_MODEL \
        --use_action_dist --seed $i >> $TWO_ACTION_LOG_FILE 2>&1 &


    done
done
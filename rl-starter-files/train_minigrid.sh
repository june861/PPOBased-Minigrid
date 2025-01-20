#!/bin/bash

# 定义可修改的参数
ALGO="ppo"
FRAMES=20000000
PYENV="minigrid"
ENVS=(
    # vcis7
    "MiniGrid-SimpleCrossingS11N5-v0"
    "MiniGrid-DistShift2-v0"
    "MiniGrid-DoorKey-16x16-v0"
    # Tgent  
    "MiniGrid-LavaGapS7-v0"
    "MiniGrid-UnlockPickup-v0"
    "MiniGrid-Dynamic-Obstacles-16x16-v0"
    "MiniGrid-GoToDoor-8x8-v0"
    "MiniGrid-Fetch-8x8-N3-v0"
    
    # "MiniGrid-ObstructedMaze-1Dlhb-v0"
    # "MiniGrid-Empty-16x16-v0"
    # "MiniGrid-FourRooms-v0"
    # "MiniGrid-GoToDoor-8x8-v0"
    # "MiniGrid-GoToObject-6x6-N2-v0"
    # "MiniGrid-GoToObject-8x8-N2-v0"
    # "MiniGrid-BlockedUnlockPickup-v0"
    # "MiniGrid-SimpleCrossingS9N3-v0"
    # "MiniGrid-SimpleCrossingS9N2-v0"
    # "MiniGrid-SimpleCrossingS9N1-v0"
)
MODEL_PREFIX="MiniGrid_"
MAX_SEED=3
# 日志文件基础路径
LOG_DIR="/home/weijun.luo/Minigrid-logs"

# 切换目录
PWD="/home/weijun.luo/PPOBased-Minigrid/rl-starter-files"
cd $PWD

# 启动虚拟环境
# source $(conda info --base)/etc/profile.d/conda.sh
conda activate $PYENV

# 遍历每个环境变量
for ENV in "${ENVS[@]}"; do
    for ((i=1; i<=MAX_SEED; i++)); do

        BASE_MODEL="${ENV}_baseline_seed${i}"
        BASE_LOG_FILE="${LOG_DIR}/${ENV}_baseline_seed${i}.log"
        # 启动第一个训练命令
        echo "Training started for environment $ENV without action distribution with the following parameters:"
        echo "Algorithm: $ALGO"
        echo "Environment: $ENV"
        echo "Frames: $FRAMES"
        echo "Model: $MODEL"
        echo "Logs will be saved to: $BASE_LOG_FILE"
        CUDA_VISIBLE_DEVICES=2,3 nohup python3 -m scripts.train --algo $ALGO --env $ENV --frames $FRAMES --model $BASE_MODEL --seed $i --text >> $BASE_LOG_FILE 2>&1 &


        # all_act 4surr
        # ALL_ACTION_MODEL_SURR4="${ENV}_seed${i}_all_act_4surr"
        # ALL_ACTION_LOG_FILE_SURR4="${LOG_DIR}/${ENV}_all_act_4surr_seed${i}.log"

        # echo "Training started for environment $ENV with action distribution with the following parameters:"
        # echo "Algorithm: $ALGO"
        # echo "Environment: $ENV"
        # echo "Frames: $FRAMES"
        # echo "Model: $ALL_ACTION_MODEL_SURR4"
        # echo "Action Distribution Enabled"
        # echo "Logs will be saved to: $ALL_ACTION_LOG_FILE_SURR4"
        # CUDA_VISIBLE_DEVICES=2,3 nohup python3 -m scripts.train --algo $ALGO --env $ENV --frames $FRAMES --model $ALL_ACTION_MODEL_SURR4 \
        #  --use_action_dist --seed $i --sample_all_act  --use_surr --text >> $ALL_ACTION_LOG_FILE_SURR4 2>&1 &


        # all_act & not surr
        ALL_ACTION_MODEL="${ENV}_seed${i}_all_act"
        ALL_ACTION_LOG_FILE="${LOG_DIR}/${ENV}_all_act_seed${i}.log"
        echo "Training started for environment $ENV with action distribution with the following parameters:"
        echo "Algorithm: $ALGO"
        echo "Environment: $ENV"
        echo "Frames: $FRAMES"
        echo "Model: $ALL_ACTION_MODEL"
        echo "Action Distribution Enabled"
        echo "Logs will be saved to: $ALL_ACTION_LOG_FILE"
        CUDA_VISIBLE_DEVICES=2,3 nohup python3 -m scripts.train --algo $ALGO --env $ENV --frames $FRAMES --model $ALL_ACTION_MODEL \
         --use_action_dist --seed $i --sample_all_act  --text >> $ALL_ACTION_LOG_FILE 2>&1 &


        # # two_act & 4surr
        # TWO_ACTION_MODEL_SURR4="${ENV}_seed${i}_two_act_4surr"
        # TWO_ACTION_LOG_FILE_SURR4="${LOG_DIR}/${ENV}_two_act_4surr_seed${i}.log"
        # echo "Training started for environment $ENV with action distribution with the following parameters:"
        # echo "Algorithm: $ALGO"
        # echo "Environment: $ENV"
        # echo "Frames: $FRAMES"
        # echo "Model: $TWO_ACTION_MODEL_SURR4"
        # echo "Action Distribution Enabled"
        # echo "Logs will be saved to: $TWO_ACTION_LOG_FILE_SURR4"
        # # 启动第二个训练命令（包含 --use_action_dist）
        # CUDA_VISIBLE_DEVICES=2,3 nohup python3 -m scripts.train --algo $ALGO --env $ENV --frames $FRAMES --model $TWO_ACTION_MODEL_SURR4 \
        # --use_action_dist --seed $i --use_surr --text >> $TWO_ACTION_LOG_FILE_SURR4 2>&1 &

        TWO_ACTION_MODEL="${ENV}_seed${i}_two_act"
        TWO_ACTION_LOG_FILE="${LOG_DIR}/${ENV}_two_act_seed${i}.log"

        echo "Training started for environment $ENV with action distribution with the following parameters:"
        echo "Algorithm: $ALGO"
        echo "Environment: $ENV"
        echo "Frames: $FRAMES"
        echo "Model: $TWO_ACTION_MODEL"
        echo "Action Distribution Enabled"
        echo "Logs will be saved to: $TWO_ACTION_LOG_FILE"
        # 启动第二个训练命令（包含 --use_action_dist）
        CUDA_VISIBLE_DEVICES=2,3 nohup python3 -m scripts.train --algo $ALGO --env $ENV --frames $FRAMES --model $TWO_ACTION_MODEL \
        --use_action_dist --seed $i  --text >> $TWO_ACTION_LOG_FILE 2>&1 &

        # noise
        # NOISE_MODEL="${ENV}_seed${i}_noise"
        # NOISE_LOG_FILE="${LOG_DIR}/${MODEL}_noise_ppo.log"

        # echo "Noise Contrastive Experiments"
        # echo "Algorithm: $ALGO"
        # echo "Environment: $ENV"
        # echo "Frames: $FRAMES"
        # echo "Model: $NOISE_MODEL"
        # echo "Action Distribution Enabled"
        # echo "Logs will be saved to: $NOISE_LOG_FILE"
        # CUDA_VISIBLE_DEVICES=2,3 nohup python3 -m scripts.train  --algo $ALGO --env $ENV --frames $FRAMES --model $NOISE_MODEL \
        # --use_action_dist --seed $i  --use_noise --text >> $NOISE_LOG_FILE 2>&1 &
    done
done
#!/bin/sh
env="StarCraft2"
map="corridor"
algo="rmappo"
exp="hu-corridor-5/2"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1  --num_env_steps 10000000 --ppo_epoch 5 --use_value_active_masks --use_eval --eval_episodes 32     --enable_hu \
    --episode_length 400 \
    --sub_episode_length 100 \
    --sub_ppo_epoch 2 \
    --sub_lr_scale 0.3 \
    --sub_entropy_coef 0.02 
done

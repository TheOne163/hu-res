import subprocess


env = "StarCraft2"
map="5m_vs_6m"
algo="rmappo"
exp="rmappo_5m6m-hu-res" # 标记关键改动
seed_max=2


for seed in range(1, seed_max + 1):
    print(f"seed is {seed}:")
    subprocess.run([
        "python", "../train/train_smac.py",
        "--env_name", env,
        "--algorithm_name", algo,
        "--experiment_name", exp,
        "--map_name", map,
        "--seed", str(seed),
        "--n_training_threads", "1",
        "--n_rollout_threads", "2",
        "--num_mini_batch", "1",
        "--episode_length", "400",
        "--num_env_steps", "10000000",
        "--ppo_epoch", "10",
        "--use_value_active_masks",
        "--use_eval",
        "--eval_episodes", "32",
        "--clip_param" ,"0.05",
        "--enable_hu",
        "--sub_episode_length", "100",
        "--sub_ppo_epoch", "2",
        "--sub_lr_scale", "0.4",
        "--sub_entropy_coef","0.015",
        "--enable_res_hu",
        "--res_bias_scale 0.1"
    ])
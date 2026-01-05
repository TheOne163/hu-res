import subprocess


env = "StarCraft2"
map_name = "MMM"
algo = "rmappo"
exp = "check"
seed_max = 1


for seed in range(1, seed_max + 1):
    print(f"seed is {seed}:")
    subprocess.run([
        "python", "../train/train_smac.py",
        "--env_name", env,
        "--algorithm_name", algo,
        "--experiment_name", exp,
        "--map_name", map_name,
        "--seed", str(seed),
        "--n_training_threads", "1",
        "--n_rollout_threads", "3",
        "--num_mini_batch", "1",
        "--episode_length", "400",
        "--num_env_steps", "10000000",
        "--ppo_epoch", "15",
        "--use_value_active_masks",
        "--use_eval",
        "--eval_episodes", "32"
    ])

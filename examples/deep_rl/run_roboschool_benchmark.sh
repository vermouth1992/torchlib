python -m torchlib.deep_rl.run --env_name RoboschoolHopper-v1 --num_envs 8 ppo --num_epoch 200 -b 40000 --value_coef 1 --entropy_coef 0.005 --num_updates 20 --alpha 0.5 --batch_size 1000 --lam 0.95 --policy tanh_normal
python -m torchlib.deep_rl.run --env_name RoboschoolHopper-v1 --num_envs 8 ppo --num_epoch 200 -b 40000 --value_coef 1 --entropy_coef 0.005 --num_updates 20 --alpha 0.5 --batch_size 1000 --lam 0.95 --policy normal

python -m torchlib.deep_rl.run --env_name RoboschoolHopper-v1 --num_envs 8 ppo --num_epoch 200 -b 40000 --value_coef 1 --entropy_coef 0.005 --num_updates 20 --alpha 0.5 --batch_size 100 --lam 0.95 --policy tanh_normal
python -m torchlib.deep_rl.run --env_name RoboschoolHopper-v1 --num_envs 8 ppo --num_epoch 200 -b 40000 --value_coef 1 --entropy_coef 0.005 --num_updates 20 --alpha 0.5 --batch_size 5000 --lam 0.95 --policy tanh_normal

python -m torchlib.deep_rl.run --env_name RoboschoolHopper-v1 --num_envs 8 ppo --num_epoch 200 -b 40000 --value_coef 1 --entropy_coef 0.01 --num_updates 20 --alpha 0.5 --batch_size 1000 --lam 0.95 --policy tanh_normal
python -m torchlib.deep_rl.run --env_name RoboschoolHopper-v1 --num_envs 8 ppo --num_epoch 200 -b 40000 --value_coef 1 --entropy_coef 0.0001 --num_updates 20 --alpha 0.5 --batch_size 1000 --lam 0.95 --policy tanh_normal

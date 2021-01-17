#~/bin/bash

#xvfb-run --auto-servernum --server-args='-screen 0 640x480x24' 
python run.py --env_kind unity --exp_name test_unity_game --curiosity 1  --env UnityMaze-v0  --use_news 1 --ext_coeff 1.0 --int_coeff 1.0 --layernorm 0 --lr 0.0001 --feat_learning none --nepochs 8 --nsteps_per_seg 900 --envs_per_process 16 --video_log_freq 0 --lstm 1 --lstm1_size 256 --lstm2_size 512  --depth_pred 0 --num_timesteps 100000000 --expID test_unity_game



nohup python -m transworld_exp --scenario 'traci_tls' --train_data "test100" --training_step 50 --pred_step 10 --hid_dim 50 --n_head 4 --n_layer 4 &
nohup python -m transworld_exp --scenario 'bologna_clean' --train_data "test500" --training_step 400 --pred_step 10 --hid_dim 50 --n_head 4 --n_layer 4 &
nohup python -m transworld_exp --scenario 'hangzhou' --train_data "test500" --training_step 50 --pred_step 10 --hid_dim 50 --n_head 4 --n_layer 4 --gpu 1 &
nohup python -m transworld_hd --scenario 'HighD' --train_data "highway02" --training_step 1500 --pred_step 10 --hid_dim 50 --n_head 4 --n_layer 4 &
nohup python -m transworld_hd --scenario 'HighD' --train_data "highway02" --training_step 2000 --pred_step 10 --hid_dim 50 --n_head 4 --n_layer 4 --pretrain_model_path "/mnt/workspace/wangding/Desktop/TransWorldNG/experiment/HighD/data/highway02/preTrainModel" &


#finetune process
nohup python -m transworld_finetune --scenario 'bologna_clean' --train_data "test500" --training_step 400 --pred_step 10 --hid_dim 50 --n_head 4 --n_layer 4 --pretrain_model_path "/mnt/workspace/wangding/Desktop/TransWorldNG/experiment/bologna_clean/data/test500/out_dim_50_n_heads_4_n_layer_4_pred_step_10" &

nohup python -m transworld_finetune --scenario 'hangzhou' --train_data "test500" --training_step 50 --pred_step 10 --hid_dim 50 --n_head 4 --n_layer 4 --pretrain_model_path "/mnt/workspace/wangding/Desktop/TransWorldNG/experiment/bologna_clean/data/test500/out_dim_50_n_heads_4_n_layer_4_pred_step_10" &
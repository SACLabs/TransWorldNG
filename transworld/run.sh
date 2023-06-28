
nohup python -m transworld_exp --scenario 'traci_tls' --train_data "run1" --training_step 50 --pred_step 10 --hid_dim 50 --n_head 4 --n_layer 4 &
nohup python -m transworld_exp --scenario 'bologna_clean' --train_data "test500" --training_step 400 --pred_step 10 --hid_dim 50 --n_head 4 --n_layer 4 &
python -m transworld_exp --scenario 'hangzhou' --train_data "run1" --training_step 50 --pred_step 10 --hid_dim 50 --n_head 4 --n_layer 4
python -m transworld_hd --scenario 'HighD' --train_data "highway02" --training_step 50 --pred_step 10 --hid_dim 50 --n_head 4 --n_layer 4
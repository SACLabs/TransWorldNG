
nohup python -m tsim_exp --scenario 'bologna_clean' --test_data "test100" --training_step 50 --pred_step 10 --hid_dim 50 --n_head 4 --n_layer 4 &
nohup python -m tsim_exp --scenario 'traci_tls' --test_data "test2" --training_step 400 --pred_step 10 --hid_dim 50 --n_head 4 --n_layer 4 &

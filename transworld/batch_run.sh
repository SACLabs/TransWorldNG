# nohup python -m tsim_exp --scenario 'traci_tls' --test_data "test3" --training_step 800 --pred_step 50 --hid_dim 50 --n_head 4 --n_layer 2 &
# nohup python -m tsim_exp --scenario 'traci_tls' --test_data "test3" --training_step 800 --pred_step 50 --hid_dim 100 --n_head 4 --n_layer 2 &
# nohup python -m tsim_exp --scenario 'traci_tls' --test_data "test3" --training_step 800 --pred_step 50 --hid_dim 200 --n_head 4 --n_layer 2 &
#nohup python -m tsim_exp --scenario 'traci_tls' --test_data "test3" --training_step 800 --pred_step 50 --hid_dim 50 --n_head 4 --n_layer 4 &
# nohup python -m tsim_exp --scenario 'traci_tls' --test_data "test3" --training_step 800 --pred_step 50 --hid_dim 100 --n_head 4 --n_layer 4 &
# nohup python -m tsim_exp --scenario 'traci_tls' --test_data "test3" --training_step 800 --pred_step 50 --hid_dim 200 --n_head 4 --n_layer 4 &
# nohup python -m tsim_exp --scenario 'traci_tls' --test_data "test500" --training_step 400 --pred_step 100 --hid_dim 100 --n_head 4 --n_layer 4 &
# nohup python -m tsim_exp --scenario 'traci_tls' --test_data "test500" --training_step 400 --pred_step 10 --hid_dim 50 --n_head 8 --n_layer 4 &
# #nohup python -m tsim_exp --scenario 'traci_tls' --test_data "test2" --training_step 400 --pred_step 10 --hid_dim 200 --n_head 4 --n_layer 2 &
# nohup python -m tsim_exp --scenario 'traci_tls' --test_data "test3" --training_step 800 --pred_step 10 --hid_dim 500 --n_head 24 --n_layer 16 &
# nohup python -m tsim_exp --scenario 'traci_tls' --test_data "test3" --training_step 800 --pred_step 10 --hid_dim 200 --n_head 8 --n_layer 8 &

nohup python -m tsim_exp --scenario 'traci_tls' --test_data "speed_testrun.sumocfg" --training_step 50 --pred_step 10 --hid_dim 50 --n_head 4 --n_layer 4 &




# nohup python -m sumo_exp --scenario_name 'traci_tls' --target_step '100' --rou_file 'cross.rou1.xml' --run_file 'run.sumocfg1' &
# nohup python -m sumo_exp --scenario_name 'traci_tls' --target_step '100' --rou_file 'cross.rou2.xml' --run_file 'run.sumocfg2' &
# nohup python -m sumo_exp --scenario_name 'traci_tls' --target_step '100' --rou_file 'cross.rou3.xml' --run_file 'run.sumocfg3' &
# nohup python -m sumo_exp --scenario_name 'traci_tls' --target_step '100' --rou_file 'cross.rou4.xml' --run_file 'run.sumocfg4' &
# nohup python -m sumo_exp --scenario_name 'traci_tls' --target_step '100' --rou_file 'cross.rou5.xml' --run_file 'run.sumocfg5' &
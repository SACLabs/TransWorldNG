#!/bin/bash
# This is a basic script to run some code

# generate data
python -m sumo_exp --scenario_name 'traci_tls' --target_step '100' --rou_file 'cross.rou1.xml' --run_file 'run.sumocfg1'

# examples
nohup python -m sumo_exp --scenario_name 'bologna_clean' --target_step '100' --rou_file 'joined.rou.xml' --run_file 'run.sumocfg' &
nohup python -m sumo_exp --scenario_name 'traci_tls' --target_step '300' --rou_file 'cross.rou3.xml' --run_file 'run.sumocfg3' &
nohup python -m sumo_exp --scenario_name 'traci_tls' --target_step '400' --rou_file 'cross.rou4.xml' --run_file 'run.sumocfg4' &
nohup python -m sumo_exp --scenario_name 'traci_tls' --target_step '1000' --rou_file 'cross.rou5.xml' --run_file 'run.sumocfg' &



nohup python -m sumo_exp --scenario_name 'hangzhou' --target_step '500' --rou_file 'hangzhou_4x4_gudang_18041610_1h.rou.xml' --run_file 'hangzhou_4x4_gudang_18041610_1h.sumocfg' &
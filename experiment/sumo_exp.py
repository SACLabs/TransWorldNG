from sumolib import checkBinary
import traci
from pathlib import Path
from sumo_env import run_sumo, get_veh_depart, get_veh_route
from datetime import datetime
import os
import argparse
import logging
import shutil
import sys






def run(scenario_name, target_step, rou_file, run_file):

    sumoBinary = checkBinary("sumo")
    
    time_diff = []
    

    exp_dir = Path(__file__).parent
    
    dir_name = "run1"
    target_step = int(target_step)
    exp_setting = exp_dir / scenario_name

    train_data_dir = exp_setting / "data" / dir_name / "train_data"
    test_data_dir = exp_setting / "data" / dir_name / "test_data"
    
    name = f"scenario_{scenario_name}_test_data_{target_step}"
    log_folder_path = exp_setting / "data" / dir_name / "Log"
    logger = setup_logger(name, log_folder_path)
    if scenario_name == "traci_tls":
        rou_file_name = "cross.rou.xml"
    elif scenario_name == "bologna_clean":
        rou_file_name ="joined.rou.xml" #"cross.rou.xml" 
    #rou_file_name = rou_file
    
    isExist = os.path.exists(train_data_dir)
    if not isExist:
        os.makedirs(train_data_dir)
    
    isExist = os.path.exists(test_data_dir)
    if not isExist:
        os.makedirs(test_data_dir)
    
    before = datetime.now()
    
    logger.info(f"========== traci_start=======")

    traci.start(
        [
            sumoBinary,
            "-c",
            (exp_setting / run_file ).absolute(),
            "--tripinfo-output",
            (exp_setting / "tripinfo.xml").absolute(),
            "--step-length", "1.0",
            "--end", "1"
            
        ])
    

    
    run_sumo(target_step, train_data_dir, test_data_dir)
    
        
    veh_depart = get_veh_depart(target_step,
        exp_setting / rou_file_name, train_data_dir
    )
    veh_route = get_veh_route(
        target_step, exp_setting / rou_file_name, train_data_dir
    )
    
    after = datetime.now()
    
    logger.info(f"========== runtime_{scenario_name}_{target_step}_{(after - before).total_seconds()} =======")
    
def create_folder(folder_path, delete_origin=False):
    if not os.path.exists(folder_path):
        # shutil.rmtree(folder_path)
        os.makedirs(folder_path, exist_ok=True)
    else:
        if delete_origin:
            shutil.rmtree(folder_path)
        os.makedirs(folder_path, exist_ok=True)

def setup_logger(name, log_folder_path, level=logging.DEBUG):
    create_folder(log_folder_path)
    log_file = log_folder_path /f"{name}_log"
    handler = logging.FileHandler(log_file,encoding="utf-8",mode="a")
    formatter = logging.Formatter("%(asctime)s,%(msecs)d,%(levelname)s,%(name)s::%(message)s")
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario_name", type=str, default='traci_tls')
    parser.add_argument("--target_step", type=str, default='100')
    parser.add_argument("--rou_file", type=str, default='cross.rou.xml')
    parser.add_argument("--run_file", type=str, default='run.sumocfg')
    args = parser.parse_args()
    run(args.scenario_name,args.target_step,args.rou_file,args.run_file)
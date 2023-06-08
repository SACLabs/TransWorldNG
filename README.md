
# TransWorldNG

TransWorldNG: Empower Traffic Simulation via Foundation Model

TransWorldNG is a cutting-edge traffic simulation model that utilizes a foundation model to accurately model complex traffic behavior and relationships from real-world data.

Key features include:

1. A unified and adaptable data structure for modeling diverse agents and relationships within complex traffic systems.
2. A heterogeneous graph learning framework that automatically generates behavior models by learning from complex traffic data.


# Citing TransWorldNG

If you use TransWorldNG in your research, please cite the paper.

Wang, D., Wang, X., Chen, L., Yao, S., Jing, M., Li, H., Li, L., Bao, S., Wang, F.Y. and Lin, Y., 2023. TransWorldNG: Traffic Simulation via Foundation Model. arXiv preprint arXiv:2305.15743.

In BibTeX format:

```
@article{wang2023transworldng,
  title={TransWorldNG: Traffic Simulation via Foundation Model},
  author={Wang, Ding and Wang, Xuhong and Chen, Liang and Yao, Shengyue and Jing, Ming and Li, Honghai and Li, Li and Bao, Shiqiang and Wang, Fei-Yue and Lin, Yilun},
  journal={arXiv preprint arXiv:2305.15743},
  year={2023}
}
```

# Environment
Make sure you have all the necessary dependencies installed before running the above commands. You can install the dependencies by running `pip install -r requirements.txt`.


# Getting Started

## Generating example data with SUMO
1. Navigate to transworldNG/experiment/gen_data.sh.
2. Modify the gen_data.sh file to specify scenarios and parameters.
3. Run gen_data.sh to generate data:
    
    ```
    ./gen_data.sh
    ```

## Running TransWorldNG simulation
1. Navigate to transWorldNG/transworld/run.sh
2. Modify the run.sh file to specify scenarios and parameters.
3. Run run.sh file. The results will be saved in the original data folder. Alternatively, you can modify the settings in transworld_exp.py.
    ```
    ./run.sh
    ```



# TransWorldNG

TransWorldNG: Empower Traffic Simulation via Foundation Model

TransWorldNG is a data-driven and multiscale traffic simulation model that leverages the power of foundation model to simulate complex traffic behavior and relationships.

Checkout these features!

1. A unified and adaptable data description structure to model diverse agents and various relationships in complex traffic systems.
2. A heterogeneous graph learning framework that automates the generation of the behavior models by incorporating diverse and complex traffic data. 
3. Multi-scale traffic simulation framework with adaptive scaling capabilit that allows modeling the temporal and spatial complexity of traffic systems at multi-scales. 


# Citing SMAC

If you use TransWorldNG in your research, please cite the paper.

Wang, D., Wang, X., Chen, L., Yao, S., Jing, M., Li, H., Li, L., Bao, S., Wang, F.Y. and Lin, Y., 2023. TransWorldNG: Traffic Simulation via Foundation Model. arXiv preprint arXiv:2305.15743.

In BibTeX format:

@article{wang2023transworldng,
  title={TransWorldNG: Traffic Simulation via Foundation Model},
  author={Wang, Ding and Wang, Xuhong and Chen, Liang and Yao, Shengyue and Jing, Ming and Li, Honghai and Li, Li and Bao, Shiqiang and Wang, Fei-Yue and Lin, Yilun},
  journal={arXiv preprint arXiv:2305.15743},
  year={2023}
}

# Environment
Make sure you have all the necessary dependencies installed before running the above commands. You can install the dependencies by running `pip install -r requirements.txt`.


# Getting Started

## Generating example data from SUMO
1. Navigate to transworldNG/experiment/gen_data.
2. Modify the gen_data.sh file to specify scenarios and parameters.
3. Run gen_data.sh to generate data:
    
    ```
    ./gen_data.sh
    ```

You can run experiment/sumo_exp.py by using the following command:


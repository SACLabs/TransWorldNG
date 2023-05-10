import os
def main():
    hidden_dim_list = [50,100,200]
    n_head_list = [2,4,8]
    n_layer_list = [1,2,3,4]
    for hidden_dim in hidden_dim_list:
        for n_head in n_head_list:
            for n_layer in n_layer_list:
                cmd = f"nohup python -m tsim_exp --dimension {hidden_dim} --n_head {n_head} --n_layer {n_layer} &"
                os.system(cmd)
                        
    


if __name__ == '__main__':
    main()
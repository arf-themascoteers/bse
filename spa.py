import main

if __name__ == "__main__":
    algorithm = ["lr","svr","rf","mlp"]
    bands = [0,45,334,430,657,1367,1976,2876,3013,3091,3397,3614,3821,3887,4199]
    train_size = [0.01,0.11,0.21,0.31,0.41,0.51,0.61,0.71,0.81,0.91]
    for a in algorithm:
        for t in train_size:
            main.run_case(a,t, case_name=f"SPA_{t}_{a}",bands=bands,result_output="spa.csv")
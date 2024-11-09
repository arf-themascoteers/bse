import main

if __name__ == "__main__":
    algorithm = ["lr","svr","rf","mlp"]
    bands = [0,45,334,430,657,1367,1976,2876,3013,3091,3397,3614,3821,3887,4199]
    train_size = [0.1,1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1]
    for a in algorithm:
        for t in train_size:
            main.run_case(a,t, case_name=f"SPA_{t}_{a}",bands=bands,result_output="spa.csv")
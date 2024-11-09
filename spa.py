import main

if __name__ == "__main__":
    algorithm = ["lr","svr","rf","mlp"]
    bands = ['0,45,334,430,657,1367,1976,2876,3013,3091,3397,3614,3821,3887,4199']
    #train_size = [1,11,21,31,41,51,61,71,81,91]
    train_size = [75]
    for a in algorithm:
        for t in train_size:
            main.run_case(algorithm,t, case_name="SPA",bands=bands)
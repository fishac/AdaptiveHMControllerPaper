def main():
    params_without_completion = []
    filename = "./postprocessing/output/parameter_optimization/linearlinearoptimizerall.out"
    file = open(filename, "r")
    
    for line in file:
        if "[" in line and "]" in line:
            params = line.split("[")[1]
            params = params.split("]")[0]
            #print(params)
            if params in params_without_completion:
                params_without_completion.remove(params)
            else:
                params_without_completion.append(params)
                
    print(params_without_completion)
    print(len(params_without_completion))
    

if __name__ == "__main__":
    main()
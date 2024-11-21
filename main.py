import torch
import project_cpu
import project_gpu

    



if __name__ == "__main__":
    
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Currently use gpu")
        project_gpu.gpu_v.run()
        
        
    else:
        
        print("Currently use cpu")
        
        project_cpu.cpu_v.run()
        
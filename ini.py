import torch

def initial(method):
    print(torch.__version__)
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('Using device:', device)

    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        # print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        torch.cuda.empty_cache()
        torch.cuda.set_device(0)

    return device

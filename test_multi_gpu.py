import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup():
    return "some global vars", 99


def run(device, params):
    data, params = params
    data1, data2 = data
    print(device)
    print("data1: ", data1)
    print("data2: ", data2)
    print("params: ", params)


def init_process(pid, num_pid, fn, fn_param):
    dist_init_method = 'tcp://{ip}:{port}'.format(ip='10.50.9.13', port=12345)
    if torch.cuda.is_available():
        backend = 'nccl'
        device = torch.device(f'cuda:{pid}')
        torch.cuda.set_device(pid)
    else:
        backend = 'gloo'
        device = torch.device('cpu')
    dist.init_process_group(backend, init_method=dist_init_method, rank=pid, world_size=num_pid)

    # Run
    fn(device, fn_param)


if __name__ == '__main__':

    data = setup()
    params = {"x": 2.5}

    num_gpus = 2
    processes = []
    mp.set_start_method("spawn")
    for proc_id in range(num_gpus):

        p = mp.Process(target=init_process, args=(proc_id, num_gpus, run, (data, params)))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

from tensorflow.python.client import device_lib


def read_gpu_memory():
    device_list = device_lib.list_local_devices()
    gpu_list = [device for device in device_list if device.device_type == 'GPU']
    gpu_memory_list = [gpu_x.memory_limit/1024/1024/1024 for gpu_x in gpu_list]
    gpu_memory = sum(gpu_memory_list)
    number_of_gpu = len(gpu_memory_list)
    return gpu_memory, number_of_gpu
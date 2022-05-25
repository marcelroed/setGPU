import os
import gpustat
import random

bestGPUs = None
stats = None
n_gpus = None

def init(n=1):
    global bestGPUs, stats, n_gpus

    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.entry['memory.used'])/float(gpu.entry['memory.total']), stats)
    pairs = list(zip(ids, ratios))
    random.shuffle(pairs)

    bestGPUs = list(map(lambda x: x[0], sorted(pairs, key=lambda x: x[1])[0:n]))

    print("setGPU: Setting visible GPUs to to: {}".format(bestGPUs))
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{','.join(map(str, bestGPUs))}"

    _has_inited = True
    n_gpus = n

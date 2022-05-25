import os
import gpustat
import random

bestGPUs = None

def init(n=1):
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.entry['memory.used'])/float(gpu.entry['memory.total']), stats)
    pairs = list(zip(ids, ratios))
    random.shuffle(pairs)

    global bestGPUs
    bestGPUs = list(map(lambda x: x[0], sorted(pairs, key=lambda x: x[1])[0:n]))

    print("setGPU: Setting visible GPUs to to: {}".format(bestGPUs))
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(bestGPU)

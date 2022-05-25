# setGPU

A small Python library that automatically sets `CUDA_VISIBLE_DEVICES`
to the least-loaded GPU on multi-GPU systems.

This is a fork that allows for multiple GPUs to be used.
Usage is different. $n$

+ Installation: `pip install git+https://github.com/marcelroed/setGPU.git`
+ Usage:
    + `import setGPU`
    + `setGPU.init(num_gpus=n)` to use $n$ GPUs.
    + These need to be ran before any import that will use a GPU like `torch` or `tensorflow`


# Dependencies

+ [Jongwook Choi's](https://wook.kr) [gpustat](https://github.com/wookayin/gpustat) library.

# Licensing

This code is in the public domain.

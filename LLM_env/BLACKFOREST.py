# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 18:33:07 2025

@author: mathi
"""

#%%
import torch
from diffusers import FluxPipeline


#%%
# Specify the custom cache directory
custom_cache_dir = "D:/Generative Models"

# Load the model with the specified cache directory
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
    cache_dir=custom_cache_dir
)

pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power


#%%

import torch
print(torch.cuda.is_available())


#%%
prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-schnell.png")


"""
OutOfMemoryError: CUDA out of memory. 
Tried to allocate 74.00 MiB. GPU 0 has a total capacity of 4.00 GiB of which 0 bytes is free.
Of the allocated memory 2.77 GiB is allocated by PyTorch, 
and 13.76 MiB is reserved by PyTorch but unallocated. 
If reserved but unallocated memory is large try setting 
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  
See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
"""





























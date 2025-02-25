# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 20:17:33 2025

@author: mathi
"""

#%%

import numpy as np
from transformers import pipeline
import torch

#%%
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
#%%

classifier = pipeline("sentiment-analysis",  device=0) #Use device = 1 if multiple GPU's
classifier("I've been waiting for a HuggingFace course my whole life.")


#%%


generator = pipeline("text-generation", model="distilgpt2", device = 0)
generator(
    "In this course, we will teach you how to",
    max_length=300,
    num_return_sequences=2,
)
































































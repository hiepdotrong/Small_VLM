import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float16

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", 
                                             torch_dtype=torch_dtype, 
                                             trust_remote_code=True
                                             ).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", 
                                          trust_remote_code=True)

model.save_pretrained("local_model_directory")
processor.save_pretrained("local_processor_directory")
from transformers import LlamaForCausalLM
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
model = LlamaForCausalLM.from_pretrained("huggyllama/llama-13b")
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=4, num_nodes=1)
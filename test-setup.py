import torch; print(torch.cuda.is_available())

import deepspeed
deepspeed.ops.op_builder.CPUAdamBuilder().load()

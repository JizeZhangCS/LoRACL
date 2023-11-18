import torch
import torch.nn.functional as F

class Singleton_LoRAloss_Piggybank:
    def __init__(self):
        self.loss = 0
        self.scale = None
        self.type = None
        self.activated = True
    
    def init_pgbk(self, scale, type):
        self.scale = scale
        self.type = type
        if self.type == "none":
            self.activated = False
    
    def accumulate_loss(self, base_kernel, lora_kernel):
        if not self.activated:
            return

        base_kernel = base_kernel.detach()

        if self.type == "tight":
            # tight lora
            abs_diff = torch.abs(lora_kernel)-torch.abs(base_kernel)*self.scale
            self.loss += torch.sum(F.relu(abs_diff))                
        elif self.type == "sum":
            epsilon = torch.sum(torch.abs(base_kernel)) * self.scale
            lora_norm = torch.sum(torch.abs(lora_kernel))
            self.loss += F.relu(lora_norm-epsilon)
        elif self.type == "2norm":
            epsilon = torch.linalg.norm(base_kernel) * self.scale
            lora_norm = torch.linalg.norm(lora_kernel)
            self.loss += F.relu(lora_norm-epsilon)
        
    def smash(self):
        if not self.activated:
            return 0
        
        tmp = self.loss
        self.loss = 0
        return tmp
    
    def disable(self):
        self.activated = False
        
loraloss_pgbk = Singleton_LoRAloss_Piggybank()        

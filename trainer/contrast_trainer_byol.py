from utils.meters import AverageMeter
import torch.nn.functional as F

class ContrastTrainerBYOL:
    def __init__(self, model):
        self.model = model

    def run(self, q_images, k_images):
        p1, p2, z1, z2 = self.model(q_images, k_images)
        loss = byol_loss_calc(p1, z2) + byol_loss_calc(p2, z1)
        return loss.mean()
    

def byol_loss_calc(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return 2 - 2 * (x * y).sum(dim=-1)

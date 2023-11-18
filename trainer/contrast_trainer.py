from utils.meters import AverageMeter

class ContrastTrainer:
    def __init__(self, criterion, model, args):
        self.model = model
        self.criterion = criterion

    def run(self, q_images, k_images):
        p1, p2, z1, z2 = self.model(q_images, k_images)

        loss = -self.criterion(p1, z2).mean()
        if p2 != None:
            loss = (loss - self.criterion(p2, z1).mean()) * 0.5

        return loss
    
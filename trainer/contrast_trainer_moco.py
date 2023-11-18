from utils.meters import AverageMeter, accuracy

class ContrastTrainer:
    def __init__(self, criterion, model, args, top1_meter, top5_meter):
        self.model = model
        self.criterion = criterion
        self.top1_meter = top1_meter
        self.top5_meter = top5_meter

    def run(self, im_q, im_k, sample_idx=None):
        if sample_idx == None:
            output, target = self.model(im_q, im_k)
        else:
            output, target = self.model(im_q, im_k, sample_idx=sample_idx)
        loss = self.criterion(output, target)
        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        self.top1_meter.update(acc1[0], im_q.size(0))
        self.top5_meter.update(acc5[0], im_q.size(0))
        return loss
        
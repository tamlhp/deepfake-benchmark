
from torch import nn

class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # if gpu_id >= 0:
        #     self.cross_entropy_loss.cuda(gpu_id)

    def forward(self, classes, labels):
        loss_t = self.cross_entropy_loss(classes[:, 0, :], labels)

        for i in range(classes.size(1) - 1):
            loss_t = loss_t + self.cross_entropy_loss(classes[:, i + 1, :], labels)

        return loss_t

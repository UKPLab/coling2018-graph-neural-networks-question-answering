import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

DEFAULT_MARGIN = 0.5
MIN_TARGET_VALUE = 0.25


class VariableMarginLoss(nn.Module):

    def forward(self, predictions, target):
        loss = Variable(torch.zeros(1))
        target_index_var = Variable(torch.LongTensor([0]))
        if torch.cuda.is_available():
            loss = loss.cuda()
            target_index_var = target_index_var.cuda()

        target_sorted, target_indices = torch.sort(target, dim=-1, descending=True)
        predictions = predictions.gather(1, target_indices)
        margins = DEFAULT_MARGIN * target_sorted.data
        # margins = margins.clamp(max=1.0, min=0.5)
        for sample_index in range(target_indices.size(0)):
            target_index = 0

            while target_index < min(target_indices.size(1), 10) and \
                    (target_sorted[sample_index, target_index].data[0] > MIN_TARGET_VALUE):
                loss += F.multi_margin_loss(predictions[sample_index, target_index:],
                                            target_index_var,
                                            margin=margins[sample_index, target_index],
                                            size_average=False)
                target_index += 1
        return loss



from enum import Enum


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, batch_time, data_time, losses, top1, top5, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.batch_time = batch_time
        self.data_time = data_time
        self.losses = losses
        self.top1 = top1
        self.top5 = top5
        self.prefix = prefix

    def reset(self):
        self.batch_time.reset()
        self.data_time.reset()
        self.losses.reset()
        self.top1.reset()
        self.top5.reset()

    def display(self, batch):
        entries = [self.prefix, self.batch_fmtstr.format(batch), str(self.batch_time), str(self.data_time), str(
            self.losses), str(self.top1), str(self.top5)]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [' *', self.batch_time.summary(), self.data_time.summary(), self.losses.summary(),
                   self.top1.summary(), self.top5.summary()]
        print(' '.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

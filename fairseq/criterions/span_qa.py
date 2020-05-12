
from fairseq.criterions import FairseqCriterion, register_criterion
import math
from fairseq import utils
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from fairseq import metrics, utils

@register_criterion('span_qa')
class SQuAD2Criterion(FairseqCriterion):
    def __init__(self, task):
        super().__init__(task)

    @classmethod
    def build_criterion(cls, args, task):
        return cls(task)

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--na-loss-weight', default=1.0, type=float)
        
        # fmt: on

    def forward(self, model, sample, reduce=True):
        # compute loss and accuracy
        tokens = sample['tokens']
        start_positions = sample['starts']
        end_positions = sample['ends']
        unanswerable = sample['unanswerables']
        
        if model.args.no_pooler:
            (start_logits, end_logits), extra = model(tokens) 
        else:
            (start_logits, end_logits, cls_logits), extra = model(tokens) 


        start_logits = start_logits.squeeze(-1)   # -> B T 1
        end_logits = end_logits.squeeze(-1)
        
        for x in (start_positions, end_positions, unanswerable):
            if x is not None and x.dim() > 1:
                x.squeeze_(-1)

        loss_fct = CrossEntropyLoss()
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2


        if not model.args.no_pooler:
            loss_fct_cls = nn.BCEWithLogitsLoss()
            cls_loss = loss_fct_cls(cls_logits, unanswerable) * 0.5
            
            total_loss += cls_loss * self.args.na_loss_weight


        sample_size = tokens.size(0) 
        logging_output = {
            'loss': utils.item(total_loss.data) if reduce else total_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        logging_output['ncorrect'] = ((start_logits.argmax(-1) == start_positions).sum() + (end_logits.argmax(-1) == end_positions).sum()) / 2
        if not model.args.no_pooler:
            logging_output['ncorrect-n'] = (((cls_logits > 0.5) == (unanswerable > 0.5)).sum())
        return total_loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy', 100.0 * ncorrect / nsentences, nsentences, round=1)
            ncorrect_n = sum(log.get('ncorrect-n', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy-n', 100.0 * ncorrect_n / nsentences, nsentences, round=1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

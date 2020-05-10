
from fairseq.criterions import FairseqCriterion, register_criterion
import math
from fairseq import utils
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from fairseq import metrics, utils

@register_criterion('span_qa')
class SQuAD2Criterion(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        if self.args.save_predictions is not None:
            self.prediction_h = open(self.args.save_predictions, 'w')
        else:
            self.prediction_h = None

    @staticmethod
    def build_criterion(cls, args, task):
        return cls(args, args, task)

    def __del__(self):
        if self.prediction_h is not None:
            self.prediction_h.close()

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--save-predictions', metavar='FILE',
                            help='file to save predictions to')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        # compute loss and accuracy
        print(sample['id'])
        tokens = sample['tokens']
        start_positions = sample['starts']
        end_positions = sample['ends']
        unanswerable = sample['unanswerables']
        
        (start_logits, end_logits, cls_logits), extra = model(tokens)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        
        
        for x in (start_positions, end_positions, unanswerable):
            if x is not None and x.dim() > 1:
                x.squeeze_(-1)

        loss_fct = CrossEntropyLoss()
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2


        loss_fct_cls = nn.BCEWithLogitsLoss()
        cls_loss = loss_fct_cls(cls_logits, unanswerable)
        
        total_loss += cls_loss * 0.5


        sample_size = tokens.size(0) 
        logging_output = {
            'loss': utils.item(total_loss.data) if reduce else total_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        logging_output['ncorrect'] = (start_logits.argmax(-1) == start_positions).sum() + (end_logits.argmax(-1) == end_positions).sum()
        return total_loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

        return agg_output
        
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

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

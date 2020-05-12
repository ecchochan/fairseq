
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq import utils
from fairseq.models.roberta.hub_interface import RobertaHubInterface
from fairseq.modules import (
    LayerNorm,
    TransformerSentenceEncoder,
)

from fairseq.models import (
    FairseqDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)

@register_model('roberta_qa')
class RobertaQAModel(FairseqLanguageModel):

    @classmethod
    def hub_models(cls):
        return {
            'roberta.base': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz',
            'roberta.large': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz',
            'roberta.large.mnli': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz',
            'roberta.large.wsc': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.wsc.tar.gz',
        }

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-layers', type=int, metavar='L',
                            help='num encoder layers')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='H',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='F',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='A',
                            help='num encoder attention heads')
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--pooler-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use for pooler layer')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN')
        parser.add_argument('--pooler-dropout', type=float, metavar='D',
                            help='dropout probability in the masked_lm pooler layers')
        parser.add_argument('--max-positions', type=int,
                            help='number of positional embeddings to learn')
        parser.add_argument('--load-checkpoint-heads', action='store_true',
                            help='(re-)register and load heads when loading checkpoints')
        parser.add_argument('--mixout', type=float, metavar='D',
                            help='mixout probability')
        parser.add_argument('--pooler-mixout', type=float, metavar='D',
                            help='dropout probability in the masked_lm pooler layers')
        parser.add_argument('--no-pooler', default=False, type=bool)


                            
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample
        if hasattr(args, 'mixout') and args.mixout > 0:
            args.dropout = 0
            args.attention_dropout = 0
            args.activation_dropout = 0
            
        if hasattr(args, 'pooler_mixout') and args.pooler_mixout > 0:
            args.pooler_dropout = 0

        encoder = RobertaQAEncoder(args, task.source_dictionary)
        return cls(args, encoder)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)
        
    def apply_mixout(self):
        from fairseq.optim.mixout import MixoutWrapper
        from functools import partial
        if hasattr(self.args, 'pooler_mixout') and self.args.pooler_mixout > 0:
            self.decoder.span_logits.apply(partial(MixoutWrapper, p=self.args.pooler_mixout))
            self.decoder.answer_class.apply(partial(MixoutWrapper, p=self.args.pooler_mixout))

        if hasattr(self.args, 'mixout') and self.args.mixout > 0:
            self.decoder.sentence_encoder.apply(partial(MixoutWrapper, p=self.args.mixout))

    @property
    def supported_targets(self):
        return {'self'}

    @classmethod
    def from_pretrained(cls, model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', bpe='gpt2', **kwargs):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return RobertaHubInterface(x['args'], x['task'], x['models'][0])

    def load_pretrained(self, checkpoint_file='model.pt'):
        state = torch.load(checkpoint_file)
        states = state["model"]
        for k, v in list(states.items()):
            new_k = k.replace('module.','')
            if '._params_learned' in k:
                new_k = new_k.replace('._params_learned','')
            del states[k]
            states[new_k] = v

        self.load_state_dict(states, strict=True)

                    
                    


class PoolerAnswerClass(nn.Module):
    """ Compute SQuAD 2.0 answer class from classification and start tokens hidden states. """
    def __init__(self, hidden_size, dropout=0.1):
        super(PoolerAnswerClass, self).__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh() #Mish() # nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
        self.dense_1 = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

    def forward(self, hidden_states, cls_index=None):

        if cls_index is not None:
            cls_index = cls_index[:, None, None].expand(-1, -1, self.hidden_size) # shape (bsz, 1, hsz)
            cls_token_state = hidden_states.gather(-2, cls_index).squeeze(-2) # shape (bsz, hsz)
        else:
            cls_token_state = hidden_states[:, 0, :] # shape (bsz, hsz)


        x = self.dense_0(cls_token_state)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense_1(x).squeeze(-1)

        return x


class RobertaQAEncoder(FairseqDecoder):
    """RoBERTa encoder.
    Implements the :class:`~fairseq.models.FairseqDecoder` interface required
    by :class:`~fairseq.models.FairseqLanguageModel`.
    """

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args
        self.sentence_encoder = TransformerSentenceEncoder(
            padding_idx=dictionary.pad(),
            vocab_size=len(dictionary),
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_positions,
            num_segments=0,
            encoder_normalize_before=True,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
        )
        self.span_logits =  nn.Linear(args.encoder_embed_dim, 2)
        
        if not args.no_pooler:
            self.answer_class = PoolerAnswerClass(args.encoder_embed_dim, args.pooler_dropout)

    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, cls_index=None, **unused):
        x, extra = self.extract_features(src_tokens, return_all_hiddens)
        x = x.transpose(0,1)
        if not features_only:
            start_logits, end_logits = self.span_logits(x).split(1, dim=-1)
            if not self.args.no_pooler:
                cls_logits = self.answer_class(x, cls_index=cls_index)
                x = (start_logits, end_logits, cls_logits)
            else:
                x = (start_logits, end_logits)
            
        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **unused):
        inner_states, _ = self.sentence_encoder(
            src_tokens, last_state_only=not return_all_hiddens,
        )
        features = inner_states[-1]
        return features, {'inner_states': inner_states if return_all_hiddens else None}

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions



@register_model_architecture('roberta_qa', 'roberta_qa')
def base_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.0)
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)


@register_model_architecture('roberta_qa', 'roberta_qa_large')
def roberta_large_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 24)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    base_architecture(args)


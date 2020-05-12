# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np

from fairseq.data import (
    ConcatSentencesDataset,
    data_utils,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    OffsetTokensDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    RollDataset,
    SortDataset,
    StripTokenDataset,
    TruncateDataset,
)
from fairseq.tasks import FairseqTask, register_task

import torch

##############################################################################
##############################################################################
####
####   Data Utilities
####
##############################################################################
##############################################################################

import marshal
def read(dat):
    uid, inp, start, end, unanswerable = marshal.loads(dat)
    inp = np.frombuffer(inp, dtype=np.uint32).astype(np.int32)
    return uid, inp, start, end, unanswerable

def fread(f):
    uid, inp, start, end, unanswerable = marshal.load(f)
    inp = np.frombuffer(inp, dtype=np.uint32).astype(np.int32)
    return uid, inp, start, end, unanswerable
            
         
def pad(list_of_tokens, 
        max_seq_length,
        dtype=np.long,
        torch_tensor=None,
        pad_idx=1):
    k = np.empty((len(list_of_tokens),max_seq_length), dtype=dtype)
    k.fill(pad_idx)
    i = 0
    for tokens in list_of_tokens:
        k[i,:len(tokens)] = tokens
        i += 1
    return k if torch_tensor is None else torch_tensor(k)

from torch.utils.data.dataset import Dataset


def chunks(l, n):
    if type(l) == type((e for e in range(1))):
        it = iter(l)
        while True:
            out = []
            try:
                for _ in range(n):
                    out.append(next(it))
            except StopIteration:
                if out:
                    yield out
                break

            yield out
    else:
    
        for i in range(0, len(l), n):
            yield l[i:i + n]

def from_records(records, max_seq_length):
    fn_style = isinstance(records,str)
    if fn_style:
      def from_file(fn):
        with open(fn, 'rb') as f:
            while True:
                try:
                    record = fread(f)
                    yield record
                except EOFError:
                    break
      records = from_file(records)

    records = list(records)
      
    prepared_records = []
    for record_samples in chunks(records,48):
        uid, inp, start, end, unanswerable = zip(*record_samples) if fn_style else zip(*(read(record) for record in record_samples))
        start = start
        end = end
        unanswerable = unanswerable
        inp = pad(inp, max_seq_length,dtype=np.long, torch_tensor=torch.LongTensor)

        for e in zip(inp, start, end, unanswerable):
            yield e


##############################################################################
##############################################################################
####
####   task
####
##############################################################################
##############################################################################



logger = logging.getLogger(__name__)

from fairseq.data import BaseWrapperDataset
@register_task('span_qa')
class SQuAD2Task(FairseqTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--do_shuffle', action='store_true', default=False)
        #max_positions

    def __init__(self, args, dictionary):
        super().__init__(args)

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = cls.dictionary = Dictionary.load(os.path.join(os.path.dirname(args.restore_file), 'dict.txt'))
        dictionary.add_symbol('<mask>')
        print('| dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        path = self.args.data + '.' + split

        tokens = []
        starts = []
        ends = []
        unanswerables = []
        
        lengths = []

        try:
            data = from_records(path, self.args.max_seq_length)
        except:
            data = []
        
        for inp, start, end, unanswerable in data:
            tokens.append(inp)
            lengths.append(len(inp))
            starts.append(start)
            ends.append(end)
            unanswerables.append(unanswerable)
            
        
        tokens = BaseWrapperDataset(tokens)
        starts = BaseWrapperDataset(np.array(starts, dtype=np.long))
        ends = BaseWrapperDataset(np.array(ends, dtype=np.long))
        lengths = np.array(lengths, dtype=np.long)
        unanswerables = BaseWrapperDataset(np.array(unanswerables, dtype=np.float32))


        print('| loaded {} batches from: {}'.format(len(lengths), path))


        dataset = NestedDictionaryDataset(
                {
                    'id': IdDataset(),
                    'tokens': tokens,
                    'starts': starts,
                    'ends': ends,
                    'unanswerables': unanswerables,
                    'nsentences': NumSamplesDataset(),
                    'ntokens': NumelDataset(tokens, reduce=True),
                },
                sizes=[lengths],
            )

        self.datasets[split] = SortDataset(dataset ,
            sort_order=[
                np.random.permutation(len(lengths)),
            ],
        ) if self.args.do_shuffle else dataset

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary


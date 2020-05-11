#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

#distutils: language = c++


"""

This file is very messy and I do not have time to document it.


"""


from functools import lru_cache
import json
import re

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder:

    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.errors = errors # how to handle errors in decoding
        byte_encoder = bytes_to_unicode()
        self.byte_encoder = [byte_encoder[i] for i in range(256)]
        self.byte_decoder = {v:k for k, v in byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.cache2 = {}
        self.cache3 = {}

        try:
            import regex as re
            self.re = re
        except ImportError:
            raise ImportError('Please install regex with: pip install regex')
            
        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = self.re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def pair_key(self, pair):
        return self.bpe_ranks.get(pair, float('inf'))
        
    def bpe(self, token):
        cdef int i
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key = self.pair_key)
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        self.cache[token] = word
        return word

    def handle_token(self, token):
        if token in self.cache2:
            return self.cache2[token]
        token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
        ret = [self.encoder[bpe_token] for bpe_token in self.bpe(token)]
        
        self.cache2[token] = ret
        return ret

    def encode(self, 
               text):
        handle = self.handle_token
        bpe_tokens = []
        for token in self.re.findall(self.pat, text):
            bpe_tokens.extend(handle(token))
            
        return bpe_tokens
        

    def handle_token_raw(self, token):
        if token in self.cache3:
            return self.cache3[token]
        token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
        ret = [bpe_token for bpe_token in self.bpe(token)]
        
        self.cache3[token] = ret
        return ret

    def encode_raw(self, 
                   text):
        cdef int i=0, j, k
        handle = self.handle_token
        orig_tokens = []
        bpe_tokens = []
        offsets = []
        char_offsets = []
        orig_tokens_append = orig_tokens.append
        bpe_tokens_append = bpe_tokens.append
        offsets_append = offsets.append
        char_offsets_append = char_offsets.append
        for e in self.re.finditer(self.pat, text):
            token = e.group()
            offset = e.span()[0]
            for k in range(len(token)):
                char_offsets_append(i)
            orig_tokens_append(token)
            j = 0
            for sub_token in self.handle_token_raw(token):
                bpe_tokens_append(sub_token)
                offsets_append(offset)
                if j > 0:
                    orig_tokens_append('')
                j += 1
                i += 1
        return bpe_tokens, orig_tokens, offsets, char_offsets
        
        

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text


def get_encoder(encoder_json_path, vocab_bpe_path):
    with open(encoder_json_path, 'r') as f:
        encoder = json.load(f)
    with open(vocab_bpe_path, 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    splitted = bpe_data.split('\n')
    bpe_merges = [tuple(merge_str.split()) for merge_str in splitted[1:len(splitted)-1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )

import sys
import os

    
class RobertaEncoder():
    def __init__(self, 
                 directory, 
                 encoder_json='encoder.json',
                 vocab_bpe='vocab.bpe',
                 dict_txt='dict.txt',
                 bos='<s>',
                 pad='<pad>',
                 eos='</s>',
                 unk='<unk>',
                 mask='<mask>'):
        with open(os.path.join(directory, dict_txt)) as f:
            text = f.read()

        self.indices = indices = {
            bos: 0,
            pad: 1,
            eos: 2,
            unk: 3,
        }
        i = len(indices)
        for e in text.split('\n'):
            if not e:
                continue
            a,b = e.split(' ')
            indices[a] = i
            i += 1
        indices[mask] = i

        encoder_json = os.path.join(directory, encoder_json)
        vocab_bpe = os.path.join(directory, vocab_bpe)
        self.bpe = get_encoder(encoder_json, vocab_bpe)
        
        
        for k, v in self.bpe.encoder.items():
            self.bpe.encoder[k] = indices[str(v)]
                
        self.bpe.decoder = {v:k for k,v in self.bpe.encoder.items()}

    def encode(self, text):
        return self.bpe.encode(text)
        
    def encode_raw(self, text):
        return self.bpe.encode_raw(text)
        
        
        
        
        
        
from libcpp.utility cimport pair
from libcpp.vector cimport vector
import sys
import unicodedata
import collections

import string




def char_anchors_to_tok_pos(r):
    if len(r.char_anchors) == 2:
        a,b = r.char_anchors
    else:
        return 0,0
    a = r.char_to_tok_offset[a]
    b = r.char_to_tok_offset[b]
    while b+1 < len(r.all_doc_tokens) and r.all_text_tokens[b+1] == '':
        b += 1
        
    return a, b


import marshal

def compress_json(data):
    return marshal.dumps(data)

def decompress_json(data):
    return marshal.loads(data)
    
    
    
    
class DocTokens():
    __slots__ = ["text",
                 "all_doc_tokens", 
                 "all_text_tokens", 
                 "tok_to_char_offset",
                 "char_to_tok_offset",
                 "char_anchors",
                 "tokenizer",
                 "segments",
                 "p_mask", #can be answer
                 "is_max_context",
                 "unique_index",
                 "original_text_span",
                 "original_text_id",
                 "original_text",
                 "qid"

               ]
    
    def serialize(self):
        return compress_json(( self.text,
                               self.all_doc_tokens, 
                               self.all_text_tokens, 
                               self.tok_to_char_offset,
                               self.char_to_tok_offset,
                              
                               self.segments,
                               self.p_mask,
                               self.is_max_context,
                               self.char_anchors,
                              
                               self.unique_index,
                               self.original_text_span,
                               self.original_text_id,
                               self.original_text,
                               self.qid))
    
    '''
    g = []
    print('"DocTokens(' + ', '.join(e+'=%r' for e in g) + ')"%%(%s)'%(','.join('self.'+e for e in g)))
    '''
    def __repr__(self):
        return "DocTokens(" \
                   "unique_index=%r," \
                   "original_text=%r," \
                   "original_text_id=%r," \
                   "original_text_span=%r," \
                   "text=%r," \
                   "all_doc_tokens=%r," \
                   "tok_to_char_offset=%r," \
                   "char_to_tok_offset=%r," \
                   "char_anchors=%r," \
                   "segments=%r," \
                   "p_mask=%r," \
                   "is_max_context=%r)" % \
              (self.unique_index,
               self.original_text,
               self.original_text_id,
               self.original_text_span,
               
               self.text,
               self.all_doc_tokens,
               self.tok_to_char_offset,
               self.char_to_tok_offset,
               
               self.char_anchors,
               self.segments,
               self.p_mask,
               self.is_max_context)

    def __init__(self, text,
                       all_doc_tokens, 
                       all_text_tokens, 
                       tok_to_char_offset,
                       char_to_tok_offset,
                 
                       segments=None,
                       p_mask=None,
                       is_max_context=None,
                       char_anchors=None,
                 
                       unique_index=None,
                       original_text_span=None,
                       original_text_id=None,
                       original_text=None,
                       qid=None):
        self.text=text
        self.all_doc_tokens=all_doc_tokens
        self.all_text_tokens=all_text_tokens
        self.tok_to_char_offset=tok_to_char_offset
        self.char_to_tok_offset=char_to_tok_offset
        self.char_anchors=[] if char_anchors is None else char_anchors 
        self.segments=[0 for _ in range(len(all_doc_tokens))] if segments is None else segments
        self.p_mask=[0 for _ in range(len(all_doc_tokens))] if p_mask is None else p_mask
        self.is_max_context=[1 for _ in range(len(all_doc_tokens))] if is_max_context is None else is_max_context
        self.tokenizer = None
        self.unique_index = unique_index
        self.original_text_span = original_text_span
        self.original_text_id = original_text_id
        self.original_text = original_text
        self.qid = qid
    
    def __getitem__(self, key):
        tok_s   = key.start
        tok_e   = key.stop # exclusive
        char_s  = self.tok_to_char_offset[tok_s]
        char_e  = self.tok_to_char_offset[tok_e-1]  # exclusive
        char_e += len(self.all_text_tokens[self.char_to_tok_offset[char_e]])
        
        all_doc_tokens            = self.all_doc_tokens[tok_s:tok_e]
        all_text_tokens           = self.all_text_tokens[tok_s:tok_e]
        
        tok_to_char_offset        = [max(e-char_s,0) for e in self.tok_to_char_offset[tok_s:tok_e]]
        
        char_to_tok_offset        = [max(e-tok_s,0) for e in self.char_to_tok_offset[char_s:char_e]]
        
        
        text                      = self.text[char_s:char_e]
        
        segments                  = self.segments[tok_s:tok_e]
        p_mask                    = self.p_mask[tok_s:tok_e]
        is_max_context            = self.is_max_context[tok_s:tok_e]
        
        char_anchors              = [c-char_s for c in self.char_anchors if c>=char_s and c<=char_e ]
        
        r = DocTokens(
            text,
            all_doc_tokens, 
            all_text_tokens, 
            tok_to_char_offset,
            char_to_tok_offset,
            
            segments,
            p_mask,
            is_max_context,
            char_anchors
            )
        
        r.tokenizer = self.tokenizer
        
        return r
    
    def extend(a,b,int segment=-1, masked=-1, max_context=-1):
        cdef int offset_char, offset_tokens, e
        cdef list tok_to_char_offset, char_to_tok_offset, char_anchors
            
        
        offset_char = len(a.text)
        offset_tokens = len(a.all_doc_tokens)
        
        a.text += b.text
        
        a.all_doc_tokens.extend(b.all_doc_tokens)
        a.all_text_tokens.extend(b.all_text_tokens)
        
        
        tok_to_char_offset = a.tok_to_char_offset
        tok_to_char_offset_append = tok_to_char_offset.append
        for e in b.tok_to_char_offset:
            tok_to_char_offset_append(e+offset_char)
                                                       
        char_to_tok_offset = a.char_to_tok_offset
        char_to_tok_offset_append = char_to_tok_offset.append
        for e in b.char_to_tok_offset:
            char_to_tok_offset_append(e+offset_tokens)       
                                                 

        
        
        # segments
        segments = a.segments
        p_mask = a.p_mask
        is_max_context = a.is_max_context
        if segment >= 0:
            segments_append = segments.append
            for _ in range(len(b.all_doc_tokens)):
                segments_append(segment)
        else:
            segments.extend(b.segments)
        
        
        if masked >= 0:
            p_mask_append = p_mask.append
            for _ in range(len(b.all_doc_tokens)):
                p_mask_append(masked)
        else:
            p_mask.extend(b.p_mask)
            
        if max_context >= 0:
            is_max_context_append = is_max_context.append
            for _ in range(len(b.all_doc_tokens)):
                is_max_context_append(max_context)
        else:
            is_max_context.extend(b.is_max_context)
        
        
        char_anchors = a.char_anchors
        char_anchors_append = char_anchors.append
        for e in b.char_anchors:
            char_anchors_append(e+offset_char)
        
        return a
    
    
    def combine(a,b):
        cdef int offset_char, offset_tokens, e
        cdef list doc_tokens, all_doc_tokens, all_text_tokens, tok_to_char_offset, char_to_tok_offset, char_anchors
            
        
        offset_char = len(a.text)
        offset_tokens = len(a.all_doc_tokens)
        
        text = a.text + b.text
        
        all_doc_tokens = a.all_doc_tokens + b.all_doc_tokens
        
        all_text_tokens = a.all_text_tokens + b.all_text_tokens
        
        
        tok_to_char_offset = []
        tok_to_char_offset.extend(a.tok_to_char_offset)
        tok_to_char_offset_append = tok_to_char_offset.append
        for e in b.tok_to_char_offset:
            tok_to_char_offset_append(e+offset_char)
                                                       
        char_to_tok_offset = []
        char_to_tok_offset.extend(a.char_to_tok_offset)
        char_to_tok_offset_append = char_to_tok_offset.append
        for e in b.char_to_tok_offset:
            char_to_tok_offset_append(e+offset_tokens)       
                                                 

        
        # segments
        
        segments = a.segments + b.segments
        p_mask = a.p_mask + b.p_mask
        is_max_context = a.is_max_context + b.is_max_context
        
        char_anchors = []
        char_anchors_append = char_anchors.append
        char_anchors.extend(a.char_anchors)
        for e in b.char_anchors:
            char_anchors_append(e+offset_char)
        
        r = DocTokens(
            text,
            all_doc_tokens, 
            all_text_tokens, 
            tok_to_char_offset,
            char_to_tok_offset,
            
            segments,
            p_mask,
            is_max_context,
            char_anchors
            )
        
        r.tokenizer = a.tokenizer
        return r
        
    def __radd__(b, a):
        if isinstance(a,str):
            a = b.tokenizer.createTokens(a)
        #assert isinstance(b,(DocTokens)) and isinstance(a,(DocTokens)), "must be DocTokens or str"
        return DocTokens.combine(a,b)
    
    def __add__(a, b):
        if isinstance(b,str):
            b = a.tokenizer.createTokens(b)
        
        #assert isinstance(b,(DocTokens)) and isinstance(a,(DocTokens)), "must be DocTokens or str"
        return DocTokens.combine(a,b)





import sentencepiece as spm

try:
    from .sentencepiece_proto cimport SentencePieceText, SentencePieceTextSentencePiece
    from .pyrobuf_list cimport *
    from .pyrobuf_util cimport *
except:
    from .fstokenizers cimport SentencePieceText, SentencePieceTextSentencePiece
    from .fstokenizers.pyrobuf_list import *
    from .fstokenizers.pyrobuf_util import *

SPIECE_UNDERLINE = '▁'
from libc.math cimport ceil
ctypedef pair[int, int] int_pair
ctypedef vector[int_pair] int_pairs
from random import sample as random_sample
from random import randint

MASKED = 1
NOT_MASKED = 0

IS_MAX_CONTEXT = 1
NOT_IS_MAX_CONTEXT = 0

PUNCTUATIONS = set('?!.')

DEFAULT_CHOICES = {'yes','no'}

class BaseTokenizer(object):
  __slots__ = ["vocab",
               "inv_vocab",
               "unk_token",
               "encoder",
               "cache",
               "do_lower_case",
               "BOS",
               "EOS",
               "PAD",
               "SEP",
               "PIPE",
               "include_num",
               "MASK",
               "NEWLINE",
               ]
  def __init__(self):
        pass


  def createTokens(self,text,segment=None, masked=None,max_context=None,char_anchors=None):
    raise Exception('Not Yet Implemented')

  def merge_context(self,
                    context, 
                    query, 
                    ):
    cdef int k, j
    bos = self.BOS
    eos = self.EOS
    ret = (self.createTokens('')
               .extend(bos)
               .extend(context,1)
               .extend(eos)
               .extend(query,2,MASKED, NOT_IS_MAX_CONTEXT)
               .extend(eos))
    return ret

  def merge_cq( self, 
                context, 
                qas,
                int max_seq_length,
                int max_seq_length_var = 10,
                int max_query_length = 368,
                int doc_stride = 128,
                int max_sliding_negative = 1,
                unique_index = None,
                context_id=None,
                str add_Q = 'Question: ',
                bint is_training = False,
                bint no_sliding = False,
                bint debug = False):
    '''
        Representation: 
        <s> <context> </s> Question: <question> yes / no / ...options / (both) </s>

    '''
    
    cdef int i = 0, j = 0, k = 0, a, b, ans_in_choice, extra_options, qlen, fixed, remaining, total_length, query_length, char_s, char_e, last_stride_j, e_s, _max_seq_length
    cdef int_pairs context_spans
    cdef int_pair p
    cdef list char_anchors
    merge_method = self.merge_context
        
    createTokens = self.createTokens
        
    results        = []
    _context       = context
    context        = createTokens(_context)
    context_length = len(context.all_doc_tokens)
    i = 0
    k = 0
    
    #cdef vector[int] adjust_char_offsets = get_adjust_char_span_table(_context, context.text)
    
    cdef bint leave = False
    
    orig_to_stripped_table = []
        
    for q in qas:
        choices       = q['choices'] if 'choices' in q else []
        answer_pos    = q['answer_pos'] if 'answer_pos' in q and q["answer_pos"] is not None and q['answer_pos'] >=0 else None
        answer_choice = q['answer_choice'] if 'answer_choice' in q else None
        answer_text   = (q['answer_text'].strip() or None) if 'answer_text' in q else None
        question_text = add_Q + q['question'].strip()
        if question_text and not question_text[len(question_text)-1] in PUNCTUATIONS:
            question_text += ' /'

        qlen = len(question_text)
            
        #question_text += ' yes / no / depends'
        question_text += ' yes / no'

        is_yes = answer_text == 'yes'
        is_no  = answer_text == 'no'
        #is_depends = answer_text == 'depends'

        if is_yes or is_no or answer_text is None: 
            answer_pos = None
        
        qid           = q['id'] if 'id' in q else None


        query_char_anchors = []

        ans_in_choice = is_yes or is_no

        for c in choices:
            if c in DEFAULT_CHOICES:
                continue
            r = re.search(r'\b%s\b'%re.escape(c), question_text)
            if r is not None:
                if answer_text == c:
                    ans_in_choice = True
                    a, b = r.span()
                    query_char_anchors = [a, b-1]
                    answer_pos = None
            else:
                if answer_text == c:
                    ans_in_choice = True
                    a = len(question_text) + 3
                    b = a + len(c)
                    query_char_anchors = [a, b-1]
                    answer_pos = None
                question_text += ' / '+ c
                extra_options += 1

        if extra_options == 2 and 'both' not in choices:
            if answer_text == 'both':
                ans_in_choice = True
                a = len(question_text) + 3
                b = a + 4
                query_char_anchors = [a, b-1]
                answer_pos = None
            question_text += ' / both'


        query         = createTokens(question_text,max_context=0)
        query.char_anchors = query_char_anchors
        
        if is_yes:
            i = qlen + 1
            query.char_anchors = [i, i+2]
        elif is_no:
            i = qlen + 7
            query.char_anchors = [i, i+1]
        #elif is_depends:
        #    i = qlen + 12
        #    query.char_anchors = [i, i+6]


        this_context  = createTokens('').extend(context)

        char_anchors = [answer_pos,
                        answer_pos+len(answer_text)-1] if answer_pos is not None else []

        if is_training and answer_pos is None and not ans_in_choice and answer_text is not None:
            print('! cannot find answer')
            continue


        this_context.char_anchors = char_anchors

        query_length  = len(query.all_doc_tokens)
        if query_length > max_query_length:
            if is_training:
                continue
            query = query[0:max_query_length]
            query_length  = max_query_length

            

        total_length = 1 + context_length + 1 + query_length + 1

        sub_result = []
        
        if total_length > max_seq_length:
            
            fixed = total_length - context_length
            if no_sliding:
                results.append([])
                continue
            
            if (is_training and query.char_anchors):
                results.append([])
                if debug:
                    print('too long for choice')
                continue


            remaining_space  = max_seq_length - fixed


            if remaining_space < doc_stride:
                print ('No space remaining for documents, please restrict the length of query and options')
                continue

            #strides          = ceil((context_length-remaining_space) / doc_stride) + 1
            #strides          = <int>(ceil(<float>(context_length-remaining_space) / <float>doc_stride)+ 1)
            #last_stride_j    = strides - 1
            
            context_spans.clear()
            #for j in range(strides):
            j = 0
            while True:
                _max_seq_length = max_seq_length + randint(-max_seq_length_var, 0)
                remaining_space  = _max_seq_length - fixed

                if False and j == last_stride_j:
                    j = context_length-_max_seq_length
                else:
                    j = j*doc_stride
                if context_length - j < doc_stride:
                    break
                context_spans.push_back(int_pair(j,min(context_length,j+remaining_space)))
                j += 1
            
            

            i = 0
            
            negs = []
            if debug:
                print('context_length:    ', context_length)
                print('doc_stride:        ', doc_stride)
                print('context_spans:     ', context_spans)
            
            for p in context_spans:
                s = p.first
                e = p.second # exclusive
                assert s-e < max_seq_length, self.merge_cq( _context, 
                                                            qas,
                                                            max_seq_length,
                                                            max_seq_length_var,
                                                            max_query_length,
                                                            doc_stride ,
                                                            max_sliding_negative,
                                                            unique_index ,
                                                            context_id,
                                                            add_Q,
                                                            is_training,
                                                            no_sliding,
                                                            True ) if not debug else s-e
                
                if debug:
                    print('context_span:', (s,e))
                    print('tok_to_char_offset: [length: %d]'%len(this_context.tok_to_char_offset)) 
                    print('all_text_tokens: [length: %d]'%len(this_context.all_text_tokens)) 
                
                
                assert s >= 0 and s-1 < context_length and e <= context_length and len(this_context.all_text_tokens) == len(this_context.all_doc_tokens) and len(this_context.all_text_tokens) == len(this_context.tok_to_char_offset), {'s':s,'e':e,'context_length':context_length,'remaining_space':remaining_space,'fixed':fixed,'context':_context,'qas':qas}
                tok_to_char_offset = this_context.tok_to_char_offset
                
                char_s = tok_to_char_offset[s]
                char_e = tok_to_char_offset[e-1] + len(this_context.all_text_tokens[e-1])
                
                if debug:
                    print('seg') 
                
                try:
                    context_segment = this_context[s:e]
                    
                except:
                    print({'s':s,'e':e,'context_length':context_length,'fixed':fixed,'context':_context,'qas':qas})
                    
                    raise
                    
                is_max_context = context_segment.is_max_context
                
                if debug:
                    print('is_max_context: [length: %d]'%len(is_max_context)) 
                #j = 0
                #for e in context_segment.all_doc_tokens:
                e_s = e-s
                for j in range(e_s):
                    is_max_context[j] = _check_is_max_context(context_spans, i, s+j)
                    #j += 1
                
                result = merge_method(context_segment,query)
                
                if debug:
                    print('a') 
                if is_training:
                    char_anchors = result.char_anchors
                    a = len(char_anchors)
                    if a == 2:
                        char_to_tok_offset = result.char_to_tok_offset
                        a,b = char_anchors

                        ans_s = char_to_tok_offset[a]
                        ans_e = char_to_tok_offset[b]
                        if debug:
                            print('char_anchors: ', (ans_s,ans_e))
                            print(_context[a:b])

                        if ans_e > e_s or ans_s < 1:
                            continue

                        if is_max_context[ans_s-1] == 0 or is_max_context[ans_e-1] == 0: # minus 1 for <s>
                            continue
                            a = 0
                    elif a != 0:
                        if debug:
                            print('a: ', a)

                        continue
                
                assert len(result.all_doc_tokens) <= max_seq_length, \
                                             self.merge_cq( _context, 
                                                            qas,
                                                            max_seq_length,
                                                            max_seq_length_var,
                                                            max_query_length,
                                                            doc_stride ,
                                                            max_sliding_negative,
                                                            unique_index ,
                                                            context_id,
                                                            add_Q,
                                                            is_training,
                                                            no_sliding,
                                                            True ) if not debug else len(result.all_doc_tokens)
                
                result.unique_index = (unique_index,k)
                result.original_text_span = (char_s,char_e)
                result.original_text_id = context_id
                result.original_text = _context
                result.qid = qid
                if is_training and a == 0:
                    negs.append(result)
                else:
                    sub_result.append(result)
                k += 1
                i += 1
                
                
            if is_training:
                if len(negs) > max_sliding_negative:
                    sub_result.extend(random_sample(negs,max_sliding_negative))
                else:
                    sub_result.extend(negs)

        else:
            result = merge_method(this_context,query)
            result.unique_index = (unique_index,k)
            result.original_text_span = (0,len(_context))
            result.original_text_id = context_id
            result.original_text = _context
            sub_result.append(result)
            result.qid = qid
            k += 1

        results.append(sub_result)

    return results

  def createControlToken(self,text,segment=None, masked=None,max_context=None,char_anchors=None):
    lstriped = text.lstrip()
    r = DocTokens(text, 
                     [text.strip()], 
                     [text.strip()], 
                     [len(text) - len(lstriped)],
                     [0 for i in range(len(text))],
                  
                     [0 if segment is None else segment],
                     [0 if masked is None else masked],
                     [1 if max_context is None else max_context],
                     char_anchors
                     )
    return r
    



  def convert_tokens_to_ids(self, tokens):
    vocab = self.vocab
    output = []
    append = output.append
    for item in tokens:
        try:
            append(vocab[item])
        except:
            append(vocab[self.unk_token])
    return output

  def convert_ids_to_tokens(self, ids):
    vocab = self.inv_vocab
    output = []
    append = output.append
    for item in ids:
        try:
            append(vocab[item])
        except:
            append(self.unk_token)
    return output


  def get_tok_span(self,int i):
    t2c = self.tok_to_char_offset
    s = t2c[i]
    if i == len(t2c) - 1:
        return s, s+len(self.all_doc_tokens[i])
    else:
        return s, t2c[i+1]
    

  def from_bytes(self,data):
    r = DocTokens(*decompress_json(data))
    r.tokenizer = self
    
    return r



class FairSeqSPTokenizer(BaseTokenizer):
  def __init__(self,
               directory, 
               spm_model='sentencepiece.bpe.model',
               dict_txt='dict.txt',
               do_lower_case=False,
               do_strip_accent=True,
               include_num=True, 
               bos='<s>',
               pad='<pad>',
               eos='</s>',
               unk_token='<unk>',
               mask='<mask>'):
    self.do_lower_case = do_lower_case
    self.do_strip_accent = do_strip_accent
    self.cache = {}
    self.include_num = include_num
    s = self


    self.sp = sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(directory, spm_model))
    
    
    with open(os.path.join(directory, dict_txt)) as f:
        text = f.read()

    self.vocab = vocab = {
        bos: 0,
        pad: 1,
        eos: 2,
        unk_token: 3,
    }
    i = len(vocab)
    for e in text.split('\n'):
        if not e:
            continue
        a,b = e.split(' ')
        vocab[a] = i
        i += 1
    vocab[mask] = i

    
    self.unk_token = unk_token
    
    
    
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
    self.EOS  = self.createControlToken('</s> ' ,  0, NOT_MASKED, NOT_IS_MAX_CONTEXT)
    self.SEP  = self.createControlToken('</s> ' ,  0, NOT_MASKED, NOT_IS_MAX_CONTEXT)
    self.CLS  = self.createControlToken('<s> '  ,  2, NOT_MASKED, NOT_IS_MAX_CONTEXT)
    self.PIPE = self.createControlToken(' | '   ,  4, NOT_MASKED, NOT_IS_MAX_CONTEXT)
    self.BOS  = self.createControlToken('<s> '  ,  2, NOT_MASKED, NOT_IS_MAX_CONTEXT)
    self.PAD  = self.createControlToken('<pad> ',  3,     MASKED, NOT_IS_MAX_CONTEXT)
    self.MASK = self.createControlToken('<mask> ', 1, NOT_MASKED,     IS_MAX_CONTEXT)

  def createTokens(self,text,segment=None, masked=None,max_context=None,char_anchors=None):
    cdef int i, j, last_e, i_offset
    all_doc_tokens = []
    all_text_tokens = []
    tok_to_char_offset = []
    char_to_tok_offset = []
    tok_length_index = []
    all_doc_tokens_append = all_doc_tokens.append
    all_text_tokens_append = all_text_tokens.append
    tok_to_char_offset_append = tok_to_char_offset.append
    char_to_tok_offset_append = char_to_tok_offset.append
    tok_length_index_append = tok_length_index.append
    
    
    cdef int cursor, s, e, size, _cursor
    cdef SentencePieceText spt = <SentencePieceText>SentencePieceText()
    cdef str o, piece
    cdef SentencePieceTextSentencePiece values
    
    
    cdef bint uncased = self.do_lower_case
    cdef bint strip_accent = self.do_strip_accent
    
    cdef str stripped_text = text
    #cdef vector[int] adjust_char_offsets
    cdef list orig_to_stripped_table
    
    if uncased:
        stripped_text = stripped_text.lower()
    if strip_accent:
        stripped_text = _run_strip_accents(stripped_text)
    
        if char_anchors:
            i = 0
            orig_to_stripped_table = get_orig_to_stripped_table(text, stripped_text)
            for j in char_anchors:
                char_anchors[i] = _orig_to_stripped(orig_to_stripped_table, j)
                i += 1
    
    spt.ParseFromString(self.sp.EncodeAsSerializedProto(stripped_text))
    
    cdef TypedList pieces = spt._pieces
    cdef TypedList cur_pieces
    cdef int offset
        
    i = 0
    i_offset = 0
    last_e = 0
    
    cdef int len_pieces = len(pieces)
    cdef int len_piece
    cdef int start
    
    if len_pieces > 0:

        values = pieces[0]
        cursor = values._begin
        #for values in pieces:
        for i in range(len_pieces):
            values = pieces[i]
            piece = values._piece
            len_piece = len(piece)
            o = values._surface
            size = len(o)
            s = cursor
            e = cursor+size
            cursor += size

            
            
            if len_piece > 1 and piece[len_piece-1] == ',' and piece[len_piece-2].isdigit():
                # note o never has space at the end
                spt.ParseFromString(self.sp.EncodeAsSerializedProto(o[:len(o)-1].replace(SPIECE_UNDERLINE, '')))
                
                cur_pieces = spt._pieces
                values = cur_pieces[0]
                start = values.begin
                
                for j in range(start):
                    char_to_tok_offset_append(i+i_offset)
                
                
                offset = 0
                
                # no SPIECE_UNDERLINE in the first sub-token if it should not be
                if piece[0] != SPIECE_UNDERLINE and values._piece[0] == SPIECE_UNDERLINE:
                    if len(values._piece) == 1:
                        #cur_pieces = cur_pieces[1:]
                        offset = 1
                    else:
                        values._piece = values._piece[1:]
                        
                        
                _cursor = start
                for j in range(offset, len(cur_pieces)):
                    values = cur_pieces[j]
                    piece = values._piece
                    o = values._surface
                    size = len(o)
                    
                    
                    all_doc_tokens_append(piece)
                    all_text_tokens_append(o)
                    
                    tok_to_char_offset_append(s+_cursor)
                    
                    for j in range(size):
                        char_to_tok_offset_append(i+i_offset)
                        
                    _cursor += size
                    i_offset += 1
                    
                        
                        
                ## add ','
                all_doc_tokens_append(',')
                all_text_tokens_append(',')
                tok_to_char_offset_append(s+_cursor)
                char_to_tok_offset_append(i+i_offset)
                
                
                
            
            else:
            
                all_doc_tokens_append(piece)
                all_text_tokens_append(o)
                tok_to_char_offset_append(s)

                for j in range(s-last_e):
                    char_to_tok_offset_append(i+i_offset)

                #tok_length_index_append(0)
                for j in range(len(o)):
                    char_to_tok_offset_append(i+i_offset)

            i += 1
            last_e = e
        
    j = i - 1
    i = len(text) - len(char_to_tok_offset)
    
    while i > 0:
        char_to_tok_offset_append(j)
        i -= 1
        
    i = 1
    
    for p in reversed(all_doc_tokens):
        if p.startswith(SPIECE_UNDERLINE):
            for j in range(i):
                tok_length_index_append(i)
            i = 1
        else:
            i += 1
            
    tok_length_index.reverse()
        
    segments = [0 for _ in range(len(all_doc_tokens))] if segment is None else [segment for _ in range(len(all_doc_tokens))]
    p_mask = [0 for _ in range(len(all_doc_tokens))] if masked is None else [masked for _ in range(len(all_doc_tokens))]
    is_max_context = [1 for _ in range(len(all_doc_tokens))] if max_context is None else [max_context for _ in range(len(all_doc_tokens))]
    r = DocTokens(stripped_text, 
                     all_doc_tokens, 
                     all_text_tokens, 
                     tok_to_char_offset,
                     char_to_tok_offset,
                  
                     segments,
                     p_mask,
                     is_max_context,
                     char_anchors
                     )
        
        
        
        
    r.tokenizer = self
    return r


class BertTokenizer(BaseTokenizer):
  def __init__(self,
               vocab_file, 
               unk_token="[UNK]",
               do_lower_case = True,
               ):
    super().__init__()
                        
    self.vocab = load_vocab(vocab_file)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.vocab['\1'] = self.vocab['[MASK]']
    self.vocab['\2'] = (self.vocab['[unused0]'] if '[unused0]' in self.vocab else self.vocab['[unused90]'])
    self.vocab['\3'] = self.vocab['[unused2]']
    self.vocab['\4'] = self.vocab['[CLS]']
    self.vocab['\1\1'] = self.vocab['[unused1]']
    self.vocab['\1\2'] = self.vocab['[unused2]']
    self.vocab['\1\3'] = self.vocab['[unused3]']
    self.vocab['\1\4'] = self.vocab['[unused4]']
    self.vocab['\1\5'] = self.vocab['[unused5]']
    self.unk_token = unk_token
    self.do_lower_case = do_lower_case
    self.cache = {}
    
    self.SEP = self.createControlToken('[SEP] ',0,     MASKED, NOT_IS_MAX_CONTEXT)
    
    self.BOS = self.createControlToken('[CLS] ',0,     MASKED, NOT_IS_MAX_CONTEXT)
    
    self.EOS = self.createControlToken('[SEP] ',0,     MASKED, NOT_IS_MAX_CONTEXT)
    
    self.PIPE = self.createControlToken(' | '   ,  0, NOT_MASKED, NOT_IS_MAX_CONTEXT)

    self.PAD = self.createControlToken('[PAD] '  ,0,     MASKED, NOT_IS_MAX_CONTEXT)
    self.MASK = self.createControlToken('[MASK] ',0,     MASKED, IS_MAX_CONTEXT)

  def createTokens(self,text,segment=None, masked=None,max_context=None,char_anchors=None):
    doc_tokens, \
    word_to_char_offset, \
    char_to_word_offset = handle_paragraph_text(text)
    
    assert len(word_to_char_offset) == len(doc_tokens)
    
    tok_to_char_offset, \
    char_to_tok_offset, \
    all_doc_tokens = handle_tokens(doc_tokens,self, word_to_char_offset, char_to_word_offset)
    
    all_text_tokens = all_doc_tokens[:] ## !!!!!!
    
    assert len(all_doc_tokens) == len(all_text_tokens) and len(all_text_tokens) == len(tok_to_char_offset) and len(char_to_tok_offset) == len(text)
        
    segments = [0 for _ in range(len(all_doc_tokens))] if segment is None else [segment for _ in range(len(all_doc_tokens))]
    p_mask = [0 for _ in range(len(all_doc_tokens))] if masked is None else [masked for _ in range(len(all_doc_tokens))]
    is_max_context = [1 for _ in range(len(all_doc_tokens))] if max_context is None else [max_context for _ in range(len(all_doc_tokens))]
    r = DocTokens(text, 
                     all_doc_tokens, 
                     all_text_tokens, 
                     tok_to_char_offset,
                     char_to_tok_offset,
                  
                     segments,
                     p_mask,
                     is_max_context,
                     char_anchors
                     )
        
    r.tokenizer = self
    return r

  def tokenize(self, text):
    cdef bint is_bad
    cdef int start, end#, len_tks
    if text in self.cache:
        return 
    
    if self.do_lower_case:
        text = _run_strip_accents(text.lower())
    output_tokens = []
    vocabs = self.vocab
    if text in vocabs:
        return [text]
    chars = text
    output_tokens_append = output_tokens.append
    output_tokens_extend = output_tokens.extend
    while True:
        is_bad = False
        start = 0
        sub_tokens = []
        sub_tokens_append = sub_tokens.append
        while start < len(chars):
          end = len(chars)
          cur_substr = None
          while start < end:
            substr = chars[start:end]
            if start > 0:
              substr = "##" + substr
            if substr in vocabs:
              cur_substr = substr
              break
            end -= 1
          if cur_substr is None:
            is_bad = True
            break
          sub_tokens_append(cur_substr)
          start = end
  
        if is_bad:
          #print(token)
          output_tokens_append(self.unk_token)
          chars = chars[1:]
          if len(chars) == 0:
            break
        else:
          output_tokens_extend(sub_tokens)
          break
            
    return output_tokens


class BertTokenizer2(BaseTokenizer):
  def __init__(self,
               vocab_file, 
               unk_token="[UNK]",
               do_lower_case = True,
               include_num=True,
               ):
    super().__init__()
                        
    self.vocab = load_vocab(vocab_file)
    self.include_num = include_num
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.vocab['\1'] = self.vocab['[MASK]']
    self.vocab['\2'] = (self.vocab['[unused0]'] if '[unused0]' in self.vocab else self.vocab['[unused90]'])
    self.vocab['\3'] = self.vocab['[unused2]']
    self.vocab['\4'] = self.vocab['[CLS]']
    self.vocab['\1\1'] = self.vocab['[unused1]']
    self.vocab['\1\2'] = self.vocab['[unused2]']
    self.vocab['\1\3'] = self.vocab['[unused3]']
    self.vocab['\1\4'] = self.vocab['[unused4]']
    self.vocab['\1\5'] = self.vocab['[unused5]']
    self.unk_token = unk_token
    self.do_lower_case = do_lower_case
    self.cache = {}
    
    self.SEP = self.createControlToken('[SEP] ',0,     MASKED, NOT_IS_MAX_CONTEXT)
    
    self.BOS = self.createControlToken('[CLS] ',0,     MASKED, NOT_IS_MAX_CONTEXT)
    
    self.EOS = self.createControlToken('[SEP] ',0,     MASKED, NOT_IS_MAX_CONTEXT)
    
    self.PIPE = self.createControlToken(' | '   ,  0, NOT_MASKED, NOT_IS_MAX_CONTEXT)

    self.PAD = self.createControlToken('[PAD] '  ,0,     MASKED, NOT_IS_MAX_CONTEXT)
    self.MASK = self.createControlToken('[MASK] ',0,     MASKED, IS_MAX_CONTEXT)

  def createTokens(self,text,segment=None, masked=None,max_context=None,char_anchors=None):
    doc_tokens, \
    word_to_char_offset, \
    char_to_word_offset = handle_paragraph_text(text,self.include_num)
    
    assert len(word_to_char_offset) == len(doc_tokens)
    
    tok_to_char_offset, \
    char_to_tok_offset, \
    all_doc_tokens = handle_tokens(doc_tokens,self, word_to_char_offset, char_to_word_offset)
    
    all_text_tokens = all_doc_tokens[:] ## !!!!!!
    
    assert len(all_doc_tokens) == len(all_text_tokens) and len(all_text_tokens) == len(tok_to_char_offset) and len(char_to_tok_offset) == len(text)
        
    segments = [0 for _ in range(len(all_doc_tokens))] if segment is None else [segment for _ in range(len(all_doc_tokens))]
    p_mask = [0 for _ in range(len(all_doc_tokens))] if masked is None else [masked for _ in range(len(all_doc_tokens))]
    is_max_context = [1 for _ in range(len(all_doc_tokens))] if max_context is None else [max_context for _ in range(len(all_doc_tokens))]
    r = DocTokens(text, 
                     all_doc_tokens, 
                     all_text_tokens, 
                     tok_to_char_offset,
                     char_to_tok_offset,
                  
                     segments,
                     p_mask,
                     is_max_context,
                     char_anchors
                     )
        
    r.tokenizer = self
    return r

  def tokenize(self, text):
    cdef bint is_bad
    cdef int start, end#, len_tks
    if text in self.cache:
        return 
    
    if self.do_lower_case:
        text = _run_strip_accents(text.lower())
    output_tokens = []
    vocabs = self.vocab
    if text in vocabs:
        return [text]
    chars = text
    output_tokens_append = output_tokens.append
    output_tokens_extend = output_tokens.extend
    while True:
        is_bad = False
        start = 0
        sub_tokens = []
        sub_tokens_append = sub_tokens.append
        while start < len(chars):
          end = len(chars)
          cur_substr = None
          while start < end:
            substr = chars[start:end]
            if start > 0:
              substr = "##" + substr
            if substr in vocabs:
              cur_substr = substr
              break
            end -= 1
          if cur_substr is None:
            is_bad = True
            break
          sub_tokens_append(cur_substr)
          start = end
  
        if is_bad:
          #print(token)
          output_tokens_append(self.unk_token)
          chars = chars[1:]
          if len(chars) == 0:
            break
        else:
          output_tokens_extend(sub_tokens)
          break
            
    return output_tokens


class RobertaTokenizer(BaseTokenizer):
  def __init__(self,
               config_dir, 
               unk_token="<unk>",
               ___equals_mask = False):
    super().__init__()
                        
    self.encoder = encoder = RobertaEncoder(config_dir)

    self.unk_token = unk_token

    self.vocab = vocab = encoder.bpe.encoder

    for k in ('<s>', '<pad>', '</s>', '<unk>', '<mask>'):
        vocab[k] = encoder.indices[k]

    self.inv_vocab = {v: k for k, v in vocab.items()}

    if ___equals_mask and 'Ġ___' in vocab:
        vocab['Ġ___'] = vocab['<mask>']


    self.BOS = self.createControlToken('<s> '    ,0,     MASKED, NOT_IS_MAX_CONTEXT)
    self.EOS = self.createControlToken('</s> '   ,0,     MASKED, NOT_IS_MAX_CONTEXT)
    self.PAD = self.createControlToken('<pad> '  ,0,     MASKED, NOT_IS_MAX_CONTEXT)
    self.MASK = self.createControlToken('<mask> ',0,     MASKED, IS_MAX_CONTEXT)
    self.PIPE = self.createControlToken(' | '    ,0, NOT_MASKED, NOT_IS_MAX_CONTEXT)

  def createTokens(self,text,segment=None, masked=None,max_context=None,char_anchors=None):
    encoder = self.encoder
    encoded = text.encode('utf-8')
    
    all_doc_tokens, all_text_tokens, tok_to_char_offset, char_to_tok_offset = encoder.encode_raw(text)
    assert len(all_doc_tokens) == len(all_text_tokens) and len(all_text_tokens) == len(tok_to_char_offset) and len(char_to_tok_offset) == len(text)
        
    segments = [0 for _ in range(len(all_doc_tokens))] if segment is None else [segment for _ in range(len(all_doc_tokens))]
    p_mask = [0 for _ in range(len(all_doc_tokens))] if masked is None else [masked for _ in range(len(all_doc_tokens))]
    is_max_context = [1 for _ in range(len(all_doc_tokens))] if max_context is None else [max_context for _ in range(len(all_doc_tokens))]
    r = DocTokens(text, 
                     all_doc_tokens, 
                     all_text_tokens, 
                     tok_to_char_offset,
                     char_to_tok_offset,
                  
                     segments,
                     p_mask,
                     is_max_context,
                     char_anchors
                     )
        
    r.tokenizer = self
    return r




control_characters = set('\t\n\r\1\2\3\4\5\6\7')
english = set(string.ascii_letters) | control_characters




cdef int _check_is_max_context(int_pairs doc_spans, 
                          int cur_span_index, 
                          int position):
  cdef int end, span_index = 0, num_left_context, num_right_context, best_span_index = -1, start, length
  cdef float score, best_score = -999 
  cdef int_pair p
    
  for p in doc_spans:
    start = p.first
    end = p.second - 1 # exclusive to inclusive
    length = end - start + 1
    if position < start:
      span_index += 1
      continue
    if position > end:
      span_index += 1
      continue
    num_left_context = position - start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * length
    if score > best_score:
      best_score = score
      best_span_index = span_index
    span_index += 1
    
  if cur_span_index == best_span_index:
    return 1
  return 0


def check_is_max_context(int_pairs doc_spans, 
                          int cur_span_index, 
                          int position):
    return _check_is_max_context(doc_spans, 
                           cur_span_index, 
                           position)















    
def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = {}
  index = 0
  with open(vocab_file, "rb") as reader:
    tokens = [e for e in reader.readlines() for e in e.split(b' ')]
    for token in tokens:
      token = token.decode('utf-8', 'ignore')
      if not token:
        continue
      token = token.strip()
      if token in vocab:
        continue
      vocab[token] = index
      index += 1
  return vocab
    
    
import unicodedata
import string

from libcpp.utility cimport pair
from libcpp.vector cimport vector
cdef vector[bint] unicodedata_whitespace_map
cdef vector[bint] unicodedata_combining_map
cdef vector[bint] unicodedata_punc_map
cdef vector[bint] unicodedata_chinese_map

unicodedata_normalize = unicodedata.normalize

cdef str _run_strip_accents(str s # .replace("``", '"').replace("''", '"')  if xlnet
                           ):  
    cdef str c, normalized = unicodedata_normalize('NFD', s)
    return ''.join(c for c in normalized if unicodedata_combining_map[ord(c)])

cdef str _run_strip_accents2(str s # .replace("``", '"').replace("''", '"')  if xlnet
                           ):  
    cdef str c, normalized = unicodedata_normalize('NFKD', s)
    return ''.join(c for c in normalized if unicodedata_combining_map[ord(c)])

def run_strip_accents(str s):
    return _run_strip_accents(s)

def run_strip_accents2(str s):
    return _run_strip_accents2(s)

import sys
for i in range(sys.maxunicode): 
    ch = chr(i)
    unicodedata_punc_map.push_back(((i >= 33 and i <= 47) or (i >= 58 and i <= 64) or
            (i >= 91 and i <= 96) or (i >= 123 and i <= 126)) or unicodedata.category(ch).startswith("P"))
    unicodedata_combining_map.push_back(unicodedata.combining(ch) == 0)

    unicodedata_whitespace_map.push_back(ch == " " or ch == "\t" or ch == "\n" or ch == "\r" or unicodedata.category(ch) == "Zs")
    
    
    unicodedata_chinese_map.push_back(((i >= 0x4E00 and i <= 0x9FFF) or  #
        (i >= 0x3400 and i <= 0x4DBF) or  #
        (i >= 0x20000 and i <= 0x2A6DF) or  #
        (i >= 0x2A700 and i <= 0x2B73F) or  #
        (i >= 0x2B740 and i <= 0x2B81F) or  #
        (i >= 0x2B820 and i <= 0x2CEAF) or
        (i >= 0xF900 and i <= 0xFAFF) or  #
        (i >= 0x2F800 and i <= 0x2FA1F)))
    

def get_stripped_to_orig_table(list orig_to_stripped_table, str orig_context):
    cdef int len_context = len(orig_context), i = 0, last = 0, change, offset
    cdef list stripped_to_orig_table = []
    
    stripped_to_orig_table_append = stripped_to_orig_table.append
    
    while i <= len_context:
        offset = orig_to_stripped_table[i]
        change = last - offset

        if change == 0:
            stripped_to_orig_table_append(offset)
        elif change > 0:
            # encoded text is expanded
            for j in range(change+1):
                stripped_to_orig_table_append(offset)
        else:
            # encoded text is shrinked
            #print(change)
            stripped_to_orig_table_append(offset)
            i -= change


        last = offset
        i += 1
    return stripped_to_orig_table
    
    
    
#cdef vector[int] get_adjust_char_span_table(str context, str stripped_context):
def get_orig_to_stripped_table(str context, 
                               str stripped_context):

    cdef int stripped_length = len(stripped_context)
    cdef int original_length = len(context)
    cdef int cursor, offset, l
    
    cdef str left, right

    #cdef vector[int] offsets
    cdef list offsets = []
    append = offsets.append
    
    if stripped_length != original_length:
        cursor = 0
        offset = 0
        l = 0
        #offsets.push_back(0)
        append(0)
        while cursor < original_length:
            for left in _run_strip_accents(context[cursor]):
                right = stripped_context[cursor-offset+l]
                
                    
                # specials
                if left == right:
                    pass
                
                
                # exception for _run_strip_accents
                elif left == '`' and cursor < original_length - 1 and context[cursor+1] == '`' and right == '"':
                    offset += 1
                    #offsets.push_back(offset-l)
                    append(offset-l)
                    cursor += 1
                    
                # exception for _run_strip_accents
                elif left == '\'' and cursor < original_length - 1 and context[cursor+1] == '\'' and right == '"':
                    offset += 1
                    #offsets.push_back(offset-l)
                    append(offset-l)
                    cursor += 1
                    
                    
                elif left != right:
                    offset += 1
                    
                l += 1
            cursor += 1
            l -= 1
            #offsets.push_back(offset-l)
            append(offset-l)
            
    return offsets
            
#cdef int _adjust_char_span(vector[int] offsets, int a):
cdef int _orig_to_stripped(list offsets, int a):
    cdef int size = len(offsets)
    #assert a < size and a >=0
    if size > 0:
        return a - offsets[a]
    
    return a



    
    
    
#for c in ['̸']:
#    unicodedata_combining_map[ord(c)] = False
    

'''
def is_chinese(c):
    return u'\u4e00' <= c <= u'\u9fff'
'''

def is_chinese(c):
    cdef int i = ord(c)
    return unicodedata_chinese_map[i]

'''
whitespaces = set(chr(0x202F) + " \t\r\n" + u"\xa0")
def is_whitespace(c):
    return c in whitespaces
'''
def is_whitespace(c):
    i = ord(c)
    return i == 32 or i == 10 or i == 9 or i == 13 or i ==160 or i == 8239 


control_characters = set('\t\n\r\1\2\3\4\5\6\7')
english = set(string.ascii_letters) | control_characters


cdef bint is_english(c,bint include_num=False):
    cdef int i = ord(c)
    return (i>64 and i <123) or (include_num and i>47 and i<58) or i==10 or i == 13 or (i>0 and i<11)

cdef bint _is_chinese_char( c):
    cdef int i = ord(c)
    return unicodedata_chinese_map[i]

def _is_punctuation(c):
    cdef int cp = ord(c)
    return unicodedata_punc_map[cp]

def is_whitespace(c):
    cdef int cp = ord(c)
    return unicodedata_whitespace_map[cp]





def handle_paragraph_text(str paragraph_text):#, debug=False):
  cdef bint prev_is_whitespace = True
  cdef int j = 0, len_doc_tokens_1 = -1

  doc_tokens = []
  word_to_char_offset = []
  char_to_word_offset = []
  doc_tokens_append = doc_tokens.append
  word_to_char_offset_append = word_to_char_offset.append
  char_to_word_offset_append = char_to_word_offset.append

    
  for c in paragraph_text:
    if is_whitespace(c):
      prev_is_whitespace = True
    elif _is_chinese_char(c) or _is_punctuation(c):
      prev_is_whitespace = True
      doc_tokens_append(c)
      len_doc_tokens_1 += 1
      char_to_word_offset_append(len_doc_tokens_1)
      word_to_char_offset_append(j)
      j += 1
      continue
    else:
      if prev_is_whitespace:
        doc_tokens_append(c)
        len_doc_tokens_1 += 1
        word_to_char_offset_append(j)
      else:
        doc_tokens[len_doc_tokens_1] += c
      prev_is_whitespace = False
    
    char_to_word_offset_append(0 if len_doc_tokens_1 < 0 else len_doc_tokens_1)
    j += 1
    
  #if debug:
  #  for d, e in zip(doc_tokens, word_to_char_offset):
  #    assert paragraph_text[e] == d[0], (d[0], paragraph_text[e])
  return doc_tokens, \
      word_to_char_offset, \
      char_to_word_offset

def handle_paragraph_text2(str paragraph_text,bint include_num=False):#, debug=False):
  cdef bint prev_is_whitespace = True
  cdef int j = 0, len_doc_tokens_1 = -1

  doc_tokens = []
  word_to_char_offset = []
  char_to_word_offset = []
  doc_tokens_append = doc_tokens.append
  word_to_char_offset_append = word_to_char_offset.append
  char_to_word_offset_append = char_to_word_offset.append

    
  for c in paragraph_text:
    if is_whitespace(c):
      prev_is_whitespace = True
    elif not is_english(c,include_num):
      prev_is_whitespace = True
      doc_tokens_append(c)
      len_doc_tokens_1 += 1
      char_to_word_offset_append(len_doc_tokens_1)
      word_to_char_offset_append(j)
      j += 1
      continue
    else:
      if prev_is_whitespace:
        doc_tokens_append(c)
        len_doc_tokens_1 += 1
        word_to_char_offset_append(j)
      else:
        doc_tokens[len_doc_tokens_1] += c
      prev_is_whitespace = False
    
    char_to_word_offset_append(0 if len_doc_tokens_1 < 0 else len_doc_tokens_1)
    j += 1
    
  #if debug:
  #  for d, e in zip(doc_tokens, word_to_char_offset):
  #    assert paragraph_text[e] == d[0], (d[0], paragraph_text[e])
  return doc_tokens, \
      word_to_char_offset, \
      char_to_word_offset

'''
def handle_tokens(list tokens, tokenizer, word_to_char_offset, char_to_word_offset):#, debug=False):
    cdef int i = 0
    tok_to_orig_index = []
    orig_to_tok_index = []
    orig_to_tok_length_index = []
    all_doc_tokens = []
    orig_to_tok_index_append = orig_to_tok_index.append
    orig_to_tok_length_index_append = orig_to_tok_length_index.append
    tok_to_orig_index_append = tok_to_orig_index.append
    all_doc_tokens_append = all_doc_tokens.append
    for token in tokens:
      orig_to_tok_index_append(len(all_doc_tokens))
      sub_tokens = tokenizer.tokenize(token)
      orig_to_tok_length_index_append(len(sub_tokens))
      for sub_token in sub_tokens:
        tok_to_orig_index_append(i)
        all_doc_tokens_append(sub_token)
      i += 1
        
    return  tok_to_orig_index, \
            orig_to_tok_index, \
            orig_to_tok_length_index, \
            all_doc_tokens
'''

def handle_tokens(list tokens, tokenizer, word_to_char_offset, char_to_word_offset):#, debug=False):
    cdef int i = 0, j
    tok_to_char_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    orig_to_tok_index_append = orig_to_tok_index.append
    tok_to_char_index_append = tok_to_char_index.append
    all_doc_tokens_append = all_doc_tokens.append
    
    
    
    for token in tokens:
      orig_to_tok_index_append(len(all_doc_tokens))
    
      sub_tokens = tokenizer.tokenize(token)
        
      for sub_token in sub_tokens:
        assert i < len(word_to_char_offset), '1'
        tok_to_char_index_append(word_to_char_offset[i])
        
        all_doc_tokens_append(sub_token)
      i += 1
    
    
    i = 0
    for j in char_to_word_offset:
        assert j < len(orig_to_tok_index), orig_to_tok_index
        assert i < len(char_to_word_offset), char_to_word_offset
        char_to_word_offset[i] = orig_to_tok_index[j]
        i += 1
    #char_to_tok_index = char_to_word_offset
        
        
        
    return  tok_to_char_index, \
            char_to_word_offset, \
            all_doc_tokens






'''
python3 cythonize.py build_ext --inplace && python3.6 cythonize.py build_ext --inplace
'''
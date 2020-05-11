from libc.stdint cimport *
from libc.string cimport *
from cpython.ref cimport PyObject

from fairseq.fstokenizers.pyrobuf_list import *
from fairseq.fstokenizers.pyrobuf_util import *

import json

cdef class SentencePieceText:


    cdef public str _text
    cdef public TypedList _pieces
    cdef public float _score


    cdef uint64_t __field_bitmap0

    cdef public bint _is_present_in_parent
    cdef bytes _cached_serialization

    cdef PyObject *_listener

    cpdef void reset(self)

    cpdef void Clear(self)
    cpdef void ClearField(self, field_name)
    cpdef void CopyFrom(self, SentencePieceText other_msg)
    cpdef bint HasField(self, field_name) except -1
    cpdef bint IsInitialized(self)
    cpdef void MergeFrom(self, SentencePieceText other_msg)
    cpdef int MergeFromString(self, data, size=*) except -1
    cpdef int ParseFromString(self, data, size=*, bint reset=*, bint cache=*) except -1
    cpdef bytes SerializeToString(self, bint cache=*)
    cpdef bytes SerializePartialToString(self)

    cdef int _protobuf_deserialize(self, const unsigned char *memory, int size, bint cache)

    cdef void _protobuf_serialize(self, bytearray buf, bint cache)

    cpdef void _Modified(self)

cdef class SentencePieceTextSentencePiece:


    cdef public str _piece
    cdef public uint32_t _id
    cdef public str _surface
    cdef public uint32_t _begin
    cdef public uint32_t _end


    cdef uint64_t __field_bitmap0

    cdef public bint _is_present_in_parent
    cdef bytes _cached_serialization

    cdef PyObject *_listener

    cpdef void reset(self)

    cpdef void Clear(self)
    cpdef void ClearField(self, field_name)
    cpdef void CopyFrom(self, SentencePieceTextSentencePiece other_msg)
    cpdef bint HasField(self, field_name) except -1
    cpdef bint IsInitialized(self)
    cpdef void MergeFrom(self, SentencePieceTextSentencePiece other_msg)
    cpdef int MergeFromString(self, data, size=*) except -1
    cpdef int ParseFromString(self, data, size=*, bint reset=*, bint cache=*) except -1
    cpdef bytes SerializeToString(self, bint cache=*)
    cpdef bytes SerializePartialToString(self)

    cdef int _protobuf_deserialize(self, const unsigned char *memory, int size, bint cache)

    cdef void _protobuf_serialize(self, bytearray buf, bint cache)

    cpdef void _Modified(self)

cdef class NBestSentencePieceText:


    cdef TypedList _nbests


    cdef uint64_t __field_bitmap0

    cdef public bint _is_present_in_parent
    cdef bytes _cached_serialization

    cdef PyObject *_listener

    cpdef void reset(self)

    cpdef void Clear(self)
    cpdef void ClearField(self, field_name)
    cpdef void CopyFrom(self, NBestSentencePieceText other_msg)
    cpdef bint HasField(self, field_name) except -1
    cpdef bint IsInitialized(self)
    cpdef void MergeFrom(self, NBestSentencePieceText other_msg)
    cpdef int MergeFromString(self, data, size=*) except -1
    cpdef int ParseFromString(self, data, size=*, bint reset=*, bint cache=*) except -1
    cpdef bytes SerializeToString(self, bint cache=*)
    cpdef bytes SerializePartialToString(self)

    cdef int _protobuf_deserialize(self, const unsigned char *memory, int size, bint cache)

    cdef void _protobuf_serialize(self, bytearray buf, bint cache)

    cpdef void _Modified(self)
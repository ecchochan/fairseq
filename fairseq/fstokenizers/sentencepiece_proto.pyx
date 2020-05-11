#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
from libc.stdint cimport *
from libc.string cimport *
from cpython.ref cimport PyObject

from fairseq.fstokenizers.pyrobuf_list import *
from fairseq.fstokenizers.pyrobuf_util import *


import base64
import json
import warnings

class DecodeError(Exception):
    pass

cdef class SentencePieceText:

    def __cinit__(self):
        self._listener = <PyObject *>null_listener

    def __dealloc__(self):
        # Remove any references to self from child messages or repeated fields
        if self._pieces is not None:
            self._pieces._listener = <PyObject *>null_listener

    def __init__(self, **kwargs):
        self.reset()
        if kwargs:
            for field_name, field_value in kwargs.items():
                try:
                    if field_name in ('pieces',):
                        getattr(self, field_name).extend(field_value)
                    else:
                        setattr(self, field_name, field_value)
                except AttributeError:
                    raise ValueError('Protocol message has no "%s" field.' % (field_name,))
        return

    def __str__(self):
        fields = [
                          'text',
                          'score',]
        components = ['{0}: {1}'.format(field, getattr(self, field)) for field in fields]
        messages = [
                            'pieces',]
        for message in messages:
            components.append('{0}: {{'.format(message))
            for line in str(getattr(self, message)).split('\n'):
                components.append('  {0}'.format(line))
            components.append('}')
        return '\n'.join(components)

    cpdef void reset(self):
        # reset values and populate defaults
    
        self.__field_bitmap0 = 0
    
        self._text = ""
        if self._pieces is not None:
            self._pieces._listener = <PyObject *>null_listener
        self._pieces = TypedList.__new__(TypedList)
        self._pieces._list_type = SentencePieceTextSentencePiece
        self._pieces._listener = <PyObject *>self
        self._score = 0
        return

    
    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self.__field_bitmap0 |= 1
        if isinstance(value, bytes):
            self._text = value.decode('utf-8')
        elif isinstance(value, str):
            self._text = value
        else:
            raise TypeError("%r has type %s, but expected one of: (%s, %s)" % (value, type(value), bytes, str))
        self._Modified()
    
    @property 
    def pieces(self):
        # lazy init sub messages
        if self._pieces is None:
            self._pieces = SentencePieceTextSentencePiece.__new__(SentencePieceTextSentencePiece)
            self._pieces.reset()
            self._pieces._listener = <PyObject *>self
        return self._pieces

    @pieces.setter
    def pieces(self, value):
        if self._pieces is not None:
            self._pieces._listener = <PyObject *>null_listener
        self._pieces = TypedList.__new__(TypedList)
        self._pieces._list_type = SentencePieceTextSentencePiece
        self._pieces._listener = <PyObject *>self
        for val in value:
            list.append(self._pieces, val)
        self._Modified()
    
    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self.__field_bitmap0 |= 4
        self._score = value
        self._Modified()
    

    cdef int _protobuf_deserialize(self, const unsigned char *memory, int size, bint cache):
        cdef int current_offset = 0
        cdef int64_t key
        cdef int64_t field_size
        cdef SentencePieceTextSentencePiece pieces_elt
        while current_offset < size:
            key = get_varint64(memory, &current_offset)
            # text
            if key == 10:
                self.__field_bitmap0 |= 1
                field_size = get_varint64(memory, &current_offset)
                self._text = str(memory[current_offset:current_offset + field_size], 'utf-8')
                current_offset += <int>field_size
            # pieces
            elif key == 18:
                pieces_elt = SentencePieceTextSentencePiece.__new__(SentencePieceTextSentencePiece)
                pieces_elt.reset()
                field_size = get_varint64(memory, &current_offset)
                if cache:
                    pieces_elt._cached_serialization = bytes(memory[current_offset:current_offset+field_size])
                current_offset += pieces_elt._protobuf_deserialize(memory+current_offset, <int>field_size, cache)
                list.append(self._pieces, pieces_elt)
            # score
            elif key == 29:
                self.__field_bitmap0 |= 4
                self._score = (<float *>&memory[current_offset])[0]
                current_offset += sizeof(float)
            # Unknown field - need to skip proper number of bytes
            else:
                assert skip_generic(memory, &current_offset, size, key & 0x7)

        self._is_present_in_parent = True

        return current_offset

    cpdef void Clear(self):
        """Clears all data that was set in the message."""
        self.reset()
        self._Modified()

    cpdef void ClearField(self, field_name):
        """Clears the contents of a given field."""
        if field_name == 'text':
            self.__field_bitmap0 &= ~1
            self._text = ""
        elif field_name == 'pieces':
            if self._pieces is not None:
                self._pieces._listener = <PyObject *>null_listener
            self._pieces = TypedList.__new__(TypedList)
            self._pieces._list_type = SentencePieceTextSentencePiece
            self._pieces._listener = <PyObject *>self
        elif field_name == 'score':
            self.__field_bitmap0 &= ~4
            self._score = 0
        else:
            raise ValueError('Protocol message has no "%s" field.' % field_name)

        self._Modified()

    cpdef void CopyFrom(self, SentencePieceText other_msg):
        """
        Copies the content of the specified message into the current message.

        Params:
            other_msg (SentencePieceText): Message to copy into the current one.
        """
        if self is other_msg:
            return
        self.reset()
        self.MergeFrom(other_msg)

    cpdef bint HasField(self, field_name) except -1:
        """
        Checks if a certain field is set for the message.

        Params:
            field_name (str): The name of the field to check.
        """
        if field_name == 'text':
            return self.__field_bitmap0 & 1 == 1
        if field_name == 'score':
            return self.__field_bitmap0 & 4 == 4
        raise ValueError('Protocol message has no singular "%s" field.' % field_name)

    cpdef bint IsInitialized(self):
        """
        Checks if the message is initialized.

        Returns:
            bool: True if the message is initialized (i.e. all of its required
                fields are set).
        """
        cdef int i
        cdef SentencePieceTextSentencePiece pieces_msg

    
        for i in range(len(self._pieces)):
            pieces_msg = <SentencePieceTextSentencePiece>self._pieces[i]
            if not pieces_msg.IsInitialized():
                return False

        return True

    cpdef void MergeFrom(self, SentencePieceText other_msg):
        """
        Merges the contents of the specified message into the current message.

        Params:
            other_msg: Message to merge into the current message.
        """
        cdef int i
        cdef SentencePieceTextSentencePiece pieces_elt

        if self is other_msg:
            return

    
        if other_msg.__field_bitmap0 & 1 == 1:
            self._text = other_msg._text
            self.__field_bitmap0 |= 1
        for i in range(len(other_msg._pieces)):
            pieces_elt = SentencePieceTextSentencePiece()
            pieces_elt.MergeFrom(other_msg._pieces[i])
            list.append(self._pieces, pieces_elt)
        if other_msg.__field_bitmap0 & 4 == 4:
            self._score = other_msg._score
            self.__field_bitmap0 |= 4

        self._Modified()

    cpdef int MergeFromString(self, data, size=None) except -1:
        """
        Merges serialized protocol buffer data into this message.

        Params:
            data (bytes): a string of binary data.
            size (int): optional - the length of the data string

        Returns:
            int: the number of bytes processed during serialization
        """
        cdef int buf
        cdef int length

        length = size if size is not None else len(data)

        buf = self._protobuf_deserialize(data, length, False)

        if buf != length:
            raise DecodeError("Truncated message: got %s expected %s" % (buf, size))

        self._Modified()

        return buf

    cpdef int ParseFromString(self, data, size=None, bint reset=True, bint cache=False) except -1:
        """
        Populate the message class from a string of protobuf encoded binary data.

        Params:
            data (bytes): a string of binary data
            size (int): optional - the length of the data string
            reset (bool): optional - whether to reset to default values before serializing
            cache (bool): optional - whether to cache serialized data

        Returns:
            int: the number of bytes processed during serialization
        """
        cdef int buf
        cdef int length

        length = size if size is not None else len(data)

        if reset:
            self.reset()

        buf = self._protobuf_deserialize(data, length, cache)

        if buf != length:
            raise DecodeError("Truncated message")

        self._Modified()

        if cache:
            self._cached_serialization = data

        return buf

    @classmethod
    def FromString(cls, s):
        message = cls()
        message.MergeFromString(s)
        return message

    cdef void _protobuf_serialize(self, bytearray buf, bint cache):
        cdef ssize_t length
        # text
        cdef bytes text_bytes
        if self.__field_bitmap0 & 1 == 1:
            set_varint64(10, buf)
            text_bytes = self._text.encode('utf-8')
            set_varint64(len(text_bytes), buf)
            buf += text_bytes
        # pieces
        cdef SentencePieceTextSentencePiece pieces_elt
        cdef bytearray pieces_elt_buf
        for pieces_elt in self._pieces:
            set_varint64(18, buf)
            if pieces_elt._cached_serialization is not None:
                set_varint64(len(pieces_elt._cached_serialization), buf)
                buf += pieces_elt._cached_serialization
            else:
                pieces_elt_buf = bytearray()
                pieces_elt._protobuf_serialize(pieces_elt_buf, cache)
                set_varint64(len(pieces_elt_buf), buf)
                buf += pieces_elt_buf
                if cache:
                    pieces_elt._cached_serialization = bytes(pieces_elt_buf)
        # score
        if self.__field_bitmap0 & 4 == 4:
            set_varint64(29, buf)
            buf += (<unsigned char *>&self._score)[:sizeof(float)]

    cpdef void _Modified(self):
        self._is_present_in_parent = True
        (<object> self._listener)._Modified()
        self._cached_serialization = None

    cpdef bytes SerializeToString(self, bint cache=False):
        """
        Serialize the message class into a string of protobuf encoded binary data.

        Returns:
            bytes: a byte string of binary data
        """
        cdef int i
        cdef SentencePieceTextSentencePiece pieces_msg

    
        for i in range(len(self._pieces)):
            pieces_msg = <SentencePieceTextSentencePiece>self._pieces[i]
            if not pieces_msg.IsInitialized():
                raise Exception("Message SentencePieceText is missing required field: pieces[%d]" % i)

        if self._cached_serialization is not None:
            return self._cached_serialization

        cdef bytearray buf = bytearray()
        self._protobuf_serialize(buf, cache)
        cdef bytes out = bytes(buf)

        if cache:
            self._cached_serialization = out

        return out

    cpdef bytes SerializePartialToString(self):
        """
        Serialize the message class into a string of protobuf encoded binary data.

        Returns:
            bytes: a byte string of binary data
        """
        if self._cached_serialization is not None:
            return self._cached_serialization

        cdef bytearray buf = bytearray()
        self._protobuf_serialize(buf, False)
        return bytes(buf)

    def SetInParent(self):
        """
        Mark this an present in the parent.
        """
        self._Modified()

    def ParseFromJson(self, data, size=None, reset=True):
        """
        Populate the message class from a json string.

        Params:
            data (str): a json string
            size (int): optional - the length of the data string
            reset (bool): optional - whether to reset to default values before serializing
        """
        if size is None:
            size = len(data)
        d = json.loads(data[:size])
        self.ParseFromDict(d, reset)

    def SerializeToJson(self, **kwargs):
        """
        Serialize the message class into a json string.

        Returns:
            str: a json formatted string
        """
        d = self.SerializeToDict()
        return json.dumps(d, **kwargs)

    def SerializePartialToJson(self, **kwargs):
        """
        Serialize the message class into a json string.

        Returns:
            str: a json formatted string
        """
        d = self.SerializePartialToDict()
        return json.dumps(d, **kwargs)

    def ParseFromDict(self, d, reset=True):
        """
        Populate the message class from a Python dictionary.

        Params:
            d (dict): a Python dictionary representing the message
            reset (bool): optional - whether to reset to default values before serializing
        """
        if reset:
            self.reset()

        assert type(d) == dict
        try:
            self.text = d["text"]
        except KeyError:
            pass
        try:
            for pieces_dict in d["pieces"]:
                pieces_elt = SentencePieceTextSentencePiece()
                pieces_elt.ParseFromDict(pieces_dict)
                self.pieces.append(pieces_elt)
        except KeyError:
            pass
        try:
            self.score = d["score"]
        except KeyError:
            pass

        self._Modified()

        return

    def SerializeToDict(self):
        """
        Translate the message into a Python dictionary.

        Returns:
            dict: a Python dictionary representing the message
        """
        out = {}
        if self.__field_bitmap0 & 1 == 1:
            out["text"] = self.text
        if len(self.pieces) > 0:
            out["pieces"] = [m.SerializeToDict() for m in self.pieces]
        if self.__field_bitmap0 & 4 == 4:
            out["score"] = self.score

        return out

    def SerializePartialToDict(self):
        """
        Translate the message into a Python dictionary.

        Returns:
            dict: a Python dictionary representing the message
        """
        out = {}
        if self.__field_bitmap0 & 1 == 1:
            out["text"] = self.text
        if len(self.pieces) > 0:
            out["pieces"] = [m.SerializePartialToDict() for m in self.pieces]
        if self.__field_bitmap0 & 4 == 4:
            out["score"] = self.score

        return out

    def Items(self):
        """
        Iterator over the field names and values of the message.

        Returns:
            iterator
        """
        yield 'text', self.text
        yield 'pieces', self.pieces
        yield 'score', self.score

    def Fields(self):
        """
        Iterator over the field names of the message.

        Returns:
            iterator
        """
        yield 'text'
        yield 'pieces'
        yield 'score'

    def Values(self):
        """
        Iterator over the values of the message.

        Returns:
            iterator
        """
        yield self.text
        yield self.pieces
        yield self.score

    

    def Setters(self):
        """
        Iterator over functions to set the fields in a message.

        Returns:
            iterator
        """
        def setter(value):
            self.text = value
        yield setter
        def setter(value):
            self.pieces = value
        yield setter
        def setter(value):
            self.score = value
        yield setter

    

cdef class SentencePieceTextSentencePiece:

    def __cinit__(self):
        self._listener = <PyObject *>null_listener

    

    def __init__(self, **kwargs):
        self.reset()
        if kwargs:
            for field_name, field_value in kwargs.items():
                try:
                    setattr(self, field_name, field_value)
                except AttributeError:
                    raise ValueError('Protocol message has no "%s" field.' % (field_name,))
        return

    def __str__(self):
        fields = [
                          'piece',
                          'id',
                          'surface',
                          'begin',
                          'end',]
        components = ['{0}: {1}'.format(field, getattr(self, field)) for field in fields]
        messages = []
        for message in messages:
            components.append('{0}: {{'.format(message))
            for line in str(getattr(self, message)).split('\n'):
                components.append('  {0}'.format(line))
            components.append('}')
        return '\n'.join(components)

    cpdef void reset(self):
        # reset values and populate defaults
    
        self.__field_bitmap0 = 0
    
        self._piece = ""
        self._id = 0
        self._surface = ""
        self._begin = 0
        self._end = 0
        return

    
    @property
    def piece(self):
        return self._piece

    @piece.setter
    def piece(self, value):
        self.__field_bitmap0 |= 1
        if isinstance(value, bytes):
            self._piece = value.decode('utf-8')
        elif isinstance(value, str):
            self._piece = value
        else:
            raise TypeError("%r has type %s, but expected one of: (%s, %s)" % (value, type(value), bytes, str))
        self._Modified()
    
    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self.__field_bitmap0 |= 2
        self._id = value
        self._Modified()
    
    @property
    def surface(self):
        return self._surface

    @surface.setter
    def surface(self, value):
        self.__field_bitmap0 |= 4
        if isinstance(value, bytes):
            self._surface = value.decode('utf-8')
        elif isinstance(value, str):
            self._surface = value
        else:
            raise TypeError("%r has type %s, but expected one of: (%s, %s)" % (value, type(value), bytes, str))
        self._Modified()
    
    @property
    def begin(self):
        return self._begin

    @begin.setter
    def begin(self, value):
        self.__field_bitmap0 |= 8
        self._begin = value
        self._Modified()
    
    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        self.__field_bitmap0 |= 16
        self._end = value
        self._Modified()
    

    cdef int _protobuf_deserialize(self, const unsigned char *memory, int size, bint cache):
        cdef int current_offset = 0
        cdef int64_t key
        cdef int64_t field_size
        while current_offset < size:
            key = get_varint64(memory, &current_offset)
            # piece
            if key == 10:
                self.__field_bitmap0 |= 1
                field_size = get_varint64(memory, &current_offset)
                self._piece = str(memory[current_offset:current_offset + field_size], 'utf-8')
                current_offset += <int>field_size
            # id
            elif key == 16:
                self.__field_bitmap0 |= 2
                self._id = get_varint32(memory, &current_offset)
            # surface
            elif key == 26:
                self.__field_bitmap0 |= 4
                field_size = get_varint64(memory, &current_offset)
                self._surface = str(memory[current_offset:current_offset + field_size], 'utf-8')
                current_offset += <int>field_size
            # begin
            elif key == 32:
                self.__field_bitmap0 |= 8
                self._begin = get_varint32(memory, &current_offset)
            # end
            elif key == 40:
                self.__field_bitmap0 |= 16
                self._end = get_varint32(memory, &current_offset)
            # Unknown field - need to skip proper number of bytes
            else:
                assert skip_generic(memory, &current_offset, size, key & 0x7)

        self._is_present_in_parent = True

        return current_offset

    cpdef void Clear(self):
        """Clears all data that was set in the message."""
        self.reset()
        self._Modified()

    cpdef void ClearField(self, field_name):
        """Clears the contents of a given field."""
        if field_name == 'piece':
            self.__field_bitmap0 &= ~1
            self._piece = ""
        elif field_name == 'id':
            self.__field_bitmap0 &= ~2
            self._id = 0
        elif field_name == 'surface':
            self.__field_bitmap0 &= ~4
            self._surface = ""
        elif field_name == 'begin':
            self.__field_bitmap0 &= ~8
            self._begin = 0
        elif field_name == 'end':
            self.__field_bitmap0 &= ~16
            self._end = 0
        else:
            raise ValueError('Protocol message has no "%s" field.' % field_name)

        self._Modified()

    cpdef void CopyFrom(self, SentencePieceTextSentencePiece other_msg):
        """
        Copies the content of the specified message into the current message.

        Params:
            other_msg (SentencePieceTextSentencePiece): Message to copy into the current one.
        """
        if self is other_msg:
            return
        self.reset()
        self.MergeFrom(other_msg)

    cpdef bint HasField(self, field_name) except -1:
        """
        Checks if a certain field is set for the message.

        Params:
            field_name (str): The name of the field to check.
        """
        if field_name == 'piece':
            return self.__field_bitmap0 & 1 == 1
        if field_name == 'id':
            return self.__field_bitmap0 & 2 == 2
        if field_name == 'surface':
            return self.__field_bitmap0 & 4 == 4
        if field_name == 'begin':
            return self.__field_bitmap0 & 8 == 8
        if field_name == 'end':
            return self.__field_bitmap0 & 16 == 16
        raise ValueError('Protocol message has no singular "%s" field.' % field_name)

    cpdef bint IsInitialized(self):
        """
        Checks if the message is initialized.

        Returns:
            bool: True if the message is initialized (i.e. all of its required
                fields are set).
        """

    

        return True

    cpdef void MergeFrom(self, SentencePieceTextSentencePiece other_msg):
        """
        Merges the contents of the specified message into the current message.

        Params:
            other_msg: Message to merge into the current message.
        """

        if self is other_msg:
            return

    
        if other_msg.__field_bitmap0 & 1 == 1:
            self._piece = other_msg._piece
            self.__field_bitmap0 |= 1
        if other_msg.__field_bitmap0 & 2 == 2:
            self._id = other_msg._id
            self.__field_bitmap0 |= 2
        if other_msg.__field_bitmap0 & 4 == 4:
            self._surface = other_msg._surface
            self.__field_bitmap0 |= 4
        if other_msg.__field_bitmap0 & 8 == 8:
            self._begin = other_msg._begin
            self.__field_bitmap0 |= 8
        if other_msg.__field_bitmap0 & 16 == 16:
            self._end = other_msg._end
            self.__field_bitmap0 |= 16

        self._Modified()

    cpdef int MergeFromString(self, data, size=None) except -1:
        """
        Merges serialized protocol buffer data into this message.

        Params:
            data (bytes): a string of binary data.
            size (int): optional - the length of the data string

        Returns:
            int: the number of bytes processed during serialization
        """
        cdef int buf
        cdef int length

        length = size if size is not None else len(data)

        buf = self._protobuf_deserialize(data, length, False)

        if buf != length:
            raise DecodeError("Truncated message: got %s expected %s" % (buf, size))

        self._Modified()

        return buf

    cpdef int ParseFromString(self, data, size=None, bint reset=True, bint cache=False) except -1:
        """
        Populate the message class from a string of protobuf encoded binary data.

        Params:
            data (bytes): a string of binary data
            size (int): optional - the length of the data string
            reset (bool): optional - whether to reset to default values before serializing
            cache (bool): optional - whether to cache serialized data

        Returns:
            int: the number of bytes processed during serialization
        """
        cdef int buf
        cdef int length

        length = size if size is not None else len(data)

        if reset:
            self.reset()

        buf = self._protobuf_deserialize(data, length, cache)

        if buf != length:
            raise DecodeError("Truncated message")

        self._Modified()

        if cache:
            self._cached_serialization = data

        return buf

    @classmethod
    def FromString(cls, s):
        message = cls()
        message.MergeFromString(s)
        return message

    cdef void _protobuf_serialize(self, bytearray buf, bint cache):
        # piece
        cdef bytes piece_bytes
        if self.__field_bitmap0 & 1 == 1:
            set_varint64(10, buf)
            piece_bytes = self._piece.encode('utf-8')
            set_varint64(len(piece_bytes), buf)
            buf += piece_bytes
        # id
        if self.__field_bitmap0 & 2 == 2:
            set_varint64(16, buf)
            set_varint32(self._id, buf)
        # surface
        cdef bytes surface_bytes
        if self.__field_bitmap0 & 4 == 4:
            set_varint64(26, buf)
            surface_bytes = self._surface.encode('utf-8')
            set_varint64(len(surface_bytes), buf)
            buf += surface_bytes
        # begin
        if self.__field_bitmap0 & 8 == 8:
            set_varint64(32, buf)
            set_varint32(self._begin, buf)
        # end
        if self.__field_bitmap0 & 16 == 16:
            set_varint64(40, buf)
            set_varint32(self._end, buf)

    cpdef void _Modified(self):
        self._is_present_in_parent = True
        (<object> self._listener)._Modified()
        self._cached_serialization = None

    cpdef bytes SerializeToString(self, bint cache=False):
        """
        Serialize the message class into a string of protobuf encoded binary data.

        Returns:
            bytes: a byte string of binary data
        """

    

        if self._cached_serialization is not None:
            return self._cached_serialization

        cdef bytearray buf = bytearray()
        self._protobuf_serialize(buf, cache)
        cdef bytes out = bytes(buf)

        if cache:
            self._cached_serialization = out

        return out

    cpdef bytes SerializePartialToString(self):
        """
        Serialize the message class into a string of protobuf encoded binary data.

        Returns:
            bytes: a byte string of binary data
        """
        if self._cached_serialization is not None:
            return self._cached_serialization

        cdef bytearray buf = bytearray()
        self._protobuf_serialize(buf, False)
        return bytes(buf)

    def SetInParent(self):
        """
        Mark this an present in the parent.
        """
        self._Modified()

    def ParseFromJson(self, data, size=None, reset=True):
        """
        Populate the message class from a json string.

        Params:
            data (str): a json string
            size (int): optional - the length of the data string
            reset (bool): optional - whether to reset to default values before serializing
        """
        if size is None:
            size = len(data)
        d = json.loads(data[:size])
        self.ParseFromDict(d, reset)

    def SerializeToJson(self, **kwargs):
        """
        Serialize the message class into a json string.

        Returns:
            str: a json formatted string
        """
        d = self.SerializeToDict()
        return json.dumps(d, **kwargs)

    def SerializePartialToJson(self, **kwargs):
        """
        Serialize the message class into a json string.

        Returns:
            str: a json formatted string
        """
        d = self.SerializePartialToDict()
        return json.dumps(d, **kwargs)

    def ParseFromDict(self, d, reset=True):
        """
        Populate the message class from a Python dictionary.

        Params:
            d (dict): a Python dictionary representing the message
            reset (bool): optional - whether to reset to default values before serializing
        """
        if reset:
            self.reset()

        assert type(d) == dict
        try:
            self.piece = d["piece"]
        except KeyError:
            pass
        try:
            self.id = d["id"]
        except KeyError:
            pass
        try:
            self.surface = d["surface"]
        except KeyError:
            pass
        try:
            self.begin = d["begin"]
        except KeyError:
            pass
        try:
            self.end = d["end"]
        except KeyError:
            pass

        self._Modified()

        return

    def SerializeToDict(self):
        """
        Translate the message into a Python dictionary.

        Returns:
            dict: a Python dictionary representing the message
        """
        out = {}
        if self.__field_bitmap0 & 1 == 1:
            out["piece"] = self.piece
        if self.__field_bitmap0 & 2 == 2:
            out["id"] = self.id
        if self.__field_bitmap0 & 4 == 4:
            out["surface"] = self.surface
        if self.__field_bitmap0 & 8 == 8:
            out["begin"] = self.begin
        if self.__field_bitmap0 & 16 == 16:
            out["end"] = self.end

        return out

    def SerializePartialToDict(self):
        """
        Translate the message into a Python dictionary.

        Returns:
            dict: a Python dictionary representing the message
        """
        out = {}
        if self.__field_bitmap0 & 1 == 1:
            out["piece"] = self.piece
        if self.__field_bitmap0 & 2 == 2:
            out["id"] = self.id
        if self.__field_bitmap0 & 4 == 4:
            out["surface"] = self.surface
        if self.__field_bitmap0 & 8 == 8:
            out["begin"] = self.begin
        if self.__field_bitmap0 & 16 == 16:
            out["end"] = self.end

        return out

    def Items(self):
        """
        Iterator over the field names and values of the message.

        Returns:
            iterator
        """
        yield 'piece', self.piece
        yield 'id', self.id
        yield 'surface', self.surface
        yield 'begin', self.begin
        yield 'end', self.end

    def Fields(self):
        """
        Iterator over the field names of the message.

        Returns:
            iterator
        """
        yield 'piece'
        yield 'id'
        yield 'surface'
        yield 'begin'
        yield 'end'

    def Values(self):
        """
        Iterator over the values of the message.

        Returns:
            iterator
        """
        yield self.piece
        yield self.id
        yield self.surface
        yield self.begin
        yield self.end

    

    def Setters(self):
        """
        Iterator over functions to set the fields in a message.

        Returns:
            iterator
        """
        def setter(value):
            self.piece = value
        yield setter
        def setter(value):
            self.id = value
        yield setter
        def setter(value):
            self.surface = value
        yield setter
        def setter(value):
            self.begin = value
        yield setter
        def setter(value):
            self.end = value
        yield setter

    

    


cdef class NBestSentencePieceText:

    def __cinit__(self):
        self._listener = <PyObject *>null_listener

    def __dealloc__(self):
        # Remove any references to self from child messages or repeated fields
        if self._nbests is not None:
            self._nbests._listener = <PyObject *>null_listener

    def __init__(self, **kwargs):
        self.reset()
        if kwargs:
            for field_name, field_value in kwargs.items():
                try:
                    if field_name in ('nbests',):
                        getattr(self, field_name).extend(field_value)
                    else:
                        setattr(self, field_name, field_value)
                except AttributeError:
                    raise ValueError('Protocol message has no "%s" field.' % (field_name,))
        return

    def __str__(self):
        fields = []
        components = ['{0}: {1}'.format(field, getattr(self, field)) for field in fields]
        messages = [
                            'nbests',]
        for message in messages:
            components.append('{0}: {{'.format(message))
            for line in str(getattr(self, message)).split('\n'):
                components.append('  {0}'.format(line))
            components.append('}')
        return '\n'.join(components)

    cpdef void reset(self):
        # reset values and populate defaults
    
        self.__field_bitmap0 = 0
    
        if self._nbests is not None:
            self._nbests._listener = <PyObject *>null_listener
        self._nbests = TypedList.__new__(TypedList)
        self._nbests._list_type = SentencePieceText
        self._nbests._listener = <PyObject *>self
        return

    
    @property
    def nbests(self):
        # lazy init sub messages
        if self._nbests is None:
            self._nbests = SentencePieceText.__new__(SentencePieceText)
            self._nbests.reset()
            self._nbests._listener = <PyObject *>self
        return self._nbests

    @nbests.setter
    def nbests(self, value):
        if self._nbests is not None:
            self._nbests._listener = <PyObject *>null_listener
        self._nbests = TypedList.__new__(TypedList)
        self._nbests._list_type = SentencePieceText
        self._nbests._listener = <PyObject *>self
        for val in value:
            list.append(self._nbests, val)
        self._Modified()
    

    cdef int _protobuf_deserialize(self, const unsigned char *memory, int size, bint cache):
        cdef int current_offset = 0
        cdef int64_t key
        cdef int64_t field_size
        cdef SentencePieceText nbests_elt
        while current_offset < size:
            key = get_varint64(memory, &current_offset)
            # nbests
            if key == 10:
                nbests_elt = SentencePieceText.__new__(SentencePieceText)
                nbests_elt.reset()
                field_size = get_varint64(memory, &current_offset)
                if cache:
                    nbests_elt._cached_serialization = bytes(memory[current_offset:current_offset+field_size])
                current_offset += nbests_elt._protobuf_deserialize(memory+current_offset, <int>field_size, cache)
                list.append(self._nbests, nbests_elt)
            # Unknown field - need to skip proper number of bytes
            else:
                assert skip_generic(memory, &current_offset, size, key & 0x7)

        self._is_present_in_parent = True

        return current_offset

    cpdef void Clear(self):
        """Clears all data that was set in the message."""
        self.reset()
        self._Modified()

    cpdef void ClearField(self, field_name):
        """Clears the contents of a given field."""
        if field_name == 'nbests':
            if self._nbests is not None:
                self._nbests._listener = <PyObject *>null_listener
            self._nbests = TypedList.__new__(TypedList)
            self._nbests._list_type = SentencePieceText
            self._nbests._listener = <PyObject *>self
        else:
            raise ValueError('Protocol message has no "%s" field.' % field_name)

        self._Modified()

    cpdef void CopyFrom(self, NBestSentencePieceText other_msg):
        """
        Copies the content of the specified message into the current message.

        Params:
            other_msg (NBestSentencePieceText): Message to copy into the current one.
        """
        if self is other_msg:
            return
        self.reset()
        self.MergeFrom(other_msg)

    cpdef bint HasField(self, field_name) except -1:
        """
        Checks if a certain field is set for the message.

        Params:
            field_name (str): The name of the field to check.
        """
        raise ValueError('Protocol message has no singular "%s" field.' % field_name)

    cpdef bint IsInitialized(self):
        """
        Checks if the message is initialized.

        Returns:
            bool: True if the message is initialized (i.e. all of its required
                fields are set).
        """
        cdef int i
        cdef SentencePieceText nbests_msg

    
        for i in range(len(self._nbests)):
            nbests_msg = <SentencePieceText>self._nbests[i]
            if not nbests_msg.IsInitialized():
                return False

        return True

    cpdef void MergeFrom(self, NBestSentencePieceText other_msg):
        """
        Merges the contents of the specified message into the current message.

        Params:
            other_msg: Message to merge into the current message.
        """
        cdef int i
        cdef SentencePieceText nbests_elt

        if self is other_msg:
            return

    
        for i in range(len(other_msg._nbests)):
            nbests_elt = SentencePieceText()
            nbests_elt.MergeFrom(other_msg._nbests[i])
            list.append(self._nbests, nbests_elt)

        self._Modified()

    cpdef int MergeFromString(self, data, size=None) except -1:
        """
        Merges serialized protocol buffer data into this message.

        Params:
            data (bytes): a string of binary data.
            size (int): optional - the length of the data string

        Returns:
            int: the number of bytes processed during serialization
        """
        cdef int buf
        cdef int length

        length = size if size is not None else len(data)

        buf = self._protobuf_deserialize(data, length, False)

        if buf != length:
            raise DecodeError("Truncated message: got %s expected %s" % (buf, size))

        self._Modified()

        return buf

    cpdef int ParseFromString(self, data, size=None, bint reset=True, bint cache=False) except -1:
        """
        Populate the message class from a string of protobuf encoded binary data.

        Params:
            data (bytes): a string of binary data
            size (int): optional - the length of the data string
            reset (bool): optional - whether to reset to default values before serializing
            cache (bool): optional - whether to cache serialized data

        Returns:
            int: the number of bytes processed during serialization
        """
        cdef int buf
        cdef int length

        length = size if size is not None else len(data)

        if reset:
            self.reset()

        buf = self._protobuf_deserialize(data, length, cache)

        if buf != length:
            raise DecodeError("Truncated message")

        self._Modified()

        if cache:
            self._cached_serialization = data

        return buf

    @classmethod
    def FromString(cls, s):
        message = cls()
        message.MergeFromString(s)
        return message

    cdef void _protobuf_serialize(self, bytearray buf, bint cache):
        cdef ssize_t length
        # nbests
        cdef SentencePieceText nbests_elt
        cdef bytearray nbests_elt_buf
        for nbests_elt in self._nbests:
            set_varint64(10, buf)
            if nbests_elt._cached_serialization is not None:
                set_varint64(len(nbests_elt._cached_serialization), buf)
                buf += nbests_elt._cached_serialization
            else:
                nbests_elt_buf = bytearray()
                nbests_elt._protobuf_serialize(nbests_elt_buf, cache)
                set_varint64(len(nbests_elt_buf), buf)
                buf += nbests_elt_buf
                if cache:
                    nbests_elt._cached_serialization = bytes(nbests_elt_buf)

    cpdef void _Modified(self):
        self._is_present_in_parent = True
        (<object> self._listener)._Modified()
        self._cached_serialization = None

    cpdef bytes SerializeToString(self, bint cache=False):
        """
        Serialize the message class into a string of protobuf encoded binary data.

        Returns:
            bytes: a byte string of binary data
        """
        cdef int i
        cdef SentencePieceText nbests_msg

    
        for i in range(len(self._nbests)):
            nbests_msg = <SentencePieceText>self._nbests[i]
            if not nbests_msg.IsInitialized():
                raise Exception("Message NBestSentencePieceText is missing required field: nbests[%d]" % i)

        if self._cached_serialization is not None:
            return self._cached_serialization

        cdef bytearray buf = bytearray()
        self._protobuf_serialize(buf, cache)
        cdef bytes out = bytes(buf)

        if cache:
            self._cached_serialization = out

        return out

    cpdef bytes SerializePartialToString(self):
        """
        Serialize the message class into a string of protobuf encoded binary data.

        Returns:
            bytes: a byte string of binary data
        """
        if self._cached_serialization is not None:
            return self._cached_serialization

        cdef bytearray buf = bytearray()
        self._protobuf_serialize(buf, False)
        return bytes(buf)

    def SetInParent(self):
        """
        Mark this an present in the parent.
        """
        self._Modified()

    def ParseFromJson(self, data, size=None, reset=True):
        """
        Populate the message class from a json string.

        Params:
            data (str): a json string
            size (int): optional - the length of the data string
            reset (bool): optional - whether to reset to default values before serializing
        """
        if size is None:
            size = len(data)
        d = json.loads(data[:size])
        self.ParseFromDict(d, reset)

    def SerializeToJson(self, **kwargs):
        """
        Serialize the message class into a json string.

        Returns:
            str: a json formatted string
        """
        d = self.SerializeToDict()
        return json.dumps(d, **kwargs)

    def SerializePartialToJson(self, **kwargs):
        """
        Serialize the message class into a json string.

        Returns:
            str: a json formatted string
        """
        d = self.SerializePartialToDict()
        return json.dumps(d, **kwargs)

    def ParseFromDict(self, d, reset=True):
        """
        Populate the message class from a Python dictionary.

        Params:
            d (dict): a Python dictionary representing the message
            reset (bool): optional - whether to reset to default values before serializing
        """
        if reset:
            self.reset()

        assert type(d) == dict
        try:
            for nbests_dict in d["nbests"]:
                nbests_elt = SentencePieceText()
                nbests_elt.ParseFromDict(nbests_dict)
                self.nbests.append(nbests_elt)
        except KeyError:
            pass

        self._Modified()

        return

    def SerializeToDict(self):
        """
        Translate the message into a Python dictionary.

        Returns:
            dict: a Python dictionary representing the message
        """
        out = {}
        if len(self.nbests) > 0:
            out["nbests"] = [m.SerializeToDict() for m in self.nbests]

        return out

    def SerializePartialToDict(self):
        """
        Translate the message into a Python dictionary.

        Returns:
            dict: a Python dictionary representing the message
        """
        out = {}
        if len(self.nbests) > 0:
            out["nbests"] = [m.SerializePartialToDict() for m in self.nbests]

        return out

    def Items(self):
        """
        Iterator over the field names and values of the message.

        Returns:
            iterator
        """
        yield 'nbests', self.nbests

    def Fields(self):
        """
        Iterator over the field names of the message.

        Returns:
            iterator
        """
        yield 'nbests'

    def Values(self):
        """
        Iterator over the values of the message.

        Returns:
            iterator
        """
        yield self.nbests

    

    def Setters(self):
        """
        Iterator over functions to set the fields in a message.

        Returns:
            iterator
        """
        def setter(value):
            self.nbests = value
        yield setter

    

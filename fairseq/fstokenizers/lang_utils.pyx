#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

cimport cython
cdef int UNICODE_LANG_MAP[1114112]
cdef int e
cdef int arabic_low = 1536-1
cdef int arabic_upp = 1791-1
cdef int thai_low = 3584-1
cdef int thai_upp = 3711+1
cdef int jap_low = 12353-1
cdef int jap_upp = 12543+1
cdef int kor_low = 12593-1
cdef int kor_upp = 12684+1
cdef int chin_low = ord('\u4e00')-1
cdef int chin_upp = ord('\u9fff')+1
cdef int kor2_low = 44032-1
cdef int kor2_upp = 55203+1
cdef int UNICODE_LANG_EN = 1
cdef int UNICODE_LANG_AR = 2
cdef int UNICODE_LANG_TH = 3
cdef int UNICODE_LANG_JP = 4
cdef int UNICODE_LANG_KO = 5
cdef int UNICODE_LANG_CH = 6

    
for e in range(1114112):
    UNICODE_LANG_MAP[e] = 0
for e in range(ord('a'),ord('z')+1):
    UNICODE_LANG_MAP[e] = UNICODE_LANG_EN
for e in range(ord('A'),ord('Z')+1):
    UNICODE_LANG_MAP[e] = UNICODE_LANG_EN
for e in range(arabic_low+1,arabic_upp):
    UNICODE_LANG_MAP[e] = UNICODE_LANG_AR
for e in range(jap_low+1,jap_upp):
    UNICODE_LANG_MAP[e] = UNICODE_LANG_JP
for e in range(kor_low+1,kor_upp):
    UNICODE_LANG_MAP[e] = UNICODE_LANG_KO
for e in range(chin_low+1,chin_upp):
    UNICODE_LANG_MAP[e] = UNICODE_LANG_CH
for e in range(kor2_low+1,kor2_upp):
    UNICODE_LANG_MAP[e] = UNICODE_LANG_KO
    
UNICODE_LANG_MAP_py = [int(e) for e in UNICODE_LANG_MAP]

def get_unicode_lang(str c):
    cdef int o = ord(c)
    return UNICODE_LANG_MAP[o]


def get_unicode_langs(str string):
    if len(string) == 0:
        return []
    cdef int o = ord(string[0]), last=UNICODE_LANG_MAP[o], h = 0, k = 0
    cdef list result = []
    
    for c in string:
        o = ord(c)
        o = UNICODE_LANG_MAP[o]
        
        if not (o == last or o == 0):
            result.append((h,k,last))
            h = k
        if o != 0:
            last = o
        k += 1
        
    if h != k-1:
        result.append((h,k,last))
        
        
    return result
    
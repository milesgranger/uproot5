# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This is an internal module for writing TTrees in the "cascading" file writer. TTrees
are more like TDirectories than they are like histograms in that they can create
objects, TBaskets, which have to be allocated through the FreeSegments.

The implementation in this module does not use the TTree infrastructure in
:doc:`uproot.models.TTree`, :doc:`uproot.models.TBranch`, and :doc:`uproot.models.TBasket`,
since the models intended for reading have to adapt to different class versions, but
a writer can always write the same class version, and because writing involves allocating
and sometimes freeing data.

See :doc:`uproot.writing._cascade` for a general overview of the cascading writer concept.
"""
from __future__ import annotations

import datetime
import math
import struct
import warnings
from collections.abc import Mapping

import numpy

import uproot.compression
import uproot.const
import uproot.reading
import uproot.serialization
from uproot.writing.writable import __setitem__

_dtype_to_char = {
    numpy.dtype("bool"): "O",
    numpy.dtype(">i1"): "B",
    numpy.dtype(">u1"): "b",
    numpy.dtype(">i2"): "S",
    numpy.dtype(">u2"): "s",
    numpy.dtype(">i4"): "I",
    numpy.dtype(">u4"): "i",
    numpy.dtype(">i8"): "L",
    numpy.dtype(">u8"): "l",
    numpy.dtype(">f4"): "F",
    numpy.dtype(">f8"): "D",
    numpy.dtype(">U"): "C",
}


class Tree:
    """
    Writes a TTree, including all TBranches, TLeaves, and (upon ``extend``) TBaskets.

    Rather than treating TBranches as a separate object, this *writable* TTree writes
    the whole metadata block in one function, so that interrelationships are easier
    to preserve.

    Writes the following class instance versions:

    - TTree: version 20
    - TBranch: version 13
    - TLeaf: version 2
    - TLeaf*: version 1
    - TBasket: version 3

    The ``write_anew`` method writes the whole tree, possibly for the first time, possibly
    because it has been moved (exceeded its initial allocation of TBasket pointers).

    The ``write_updates`` method rewrites the parts that change when new TBaskets are
    added.

    The ``extend`` method adds a TBasket to every TBranch.

    The ``write_np_basket`` and ``write_jagged_basket`` methods write one TBasket in one
    TBranch, either a rectilinear one from NumPy or a simple jagged array from Awkward Array.

    See `ROOT TTree specification <https://github.com/root-project/root/blob/master/io/doc/TFile/ttree.md>`__.
    """

    def __init__(
        self,
        source,
        # new_branch,
    ):
        # 1: are any of these actually attributes of whatever type source will end up being
        # 2: so would source already be decompressed? Does any of this work? readonlyttree?
        # Use "readonlykey" to get 
        self.source = source
        self._directory = source.__dir__ # good
        self._name = source.name
        self._title = source.title
        self._freesegments # nope
        self._counter_name
        self._field_name = source.field_name
        self._basket_capacity = source.initial_basket_capacity
        self._resize_factor = source.resize_factor

        self._num_entries = source.num_entries
        self._num_baskets = source._num_baskets

        self._metadata_start = source._metadata_start
        self._metadata = {
            "fTotBytes": source.members["fTotBytes"],
            "fZipBytes": source.members["fZipBytes"],
            "fSavedBytes": source.members["fSavedBytes"],
            "fFlushedBytes": source.members["fFlushedBytes"],
            "fWeight": source.members["fWeight"],
            "fTimerInterval": source.members["fTimerInterval"],
            "fScanField": source.members["fScanField"],
            "fUpdate": source.members["fUpdate"],
            "fDefaultEntryOffsetLen": source.members["fDefaultEntryOffsetLen"],
            "fNClusterRange": source.members["fNClusterRange"],
            "fMaxEntries": source.members["fMaxEntries"],
            "fMaxEntryLoop": source.members["fMaxEntryLoop"],
            "fMaxVirtualSize": source.members["fMaxVirtualSize"],
            "fAutoSave": source.members["fAutoSave"],
            "fAutoFlush": source.members["fAutoFlush"],
            "fEstimate": source.members["fEstimate"],
        }

        self._key = None

        self._branch_data = source.cascading._branch_data
        self._branch_lookup = source.cascading._branch_lookup


        # Add new branch (eventually!)
        # self._branch_data.append({"kind":"record, "})

        # for branch_name, branch_type in branch_types_items:
        #     branch_dict = None
        #     branch_dtype = None
        #     branch_datashape = None

        # self.__setitem__() # where?

        for datum in source._branch_data:
            if datum["kind"] == "record":
                continue

            fBasketBytes = datum["fBasketBytes"]
            fBasketEntry = datum["fBasketEntry"]
            fBasketSeek = datum["fBasketSeek"]

            datum["fBasketBytes"] = numpy.zeros(
                self._basket_capacity, uproot.models.TBranch._tbranch13_dtype1
            )
            datum["fBasketEntry"] = numpy.zeros(
                self._basket_capacity, uproot.models.TBranch._tbranch13_dtype2
            )
            datum["fBasketSeek"] = numpy.zeros(
                self._basket_capacity, uproot.models.TBranch._tbranch13_dtype3
            )
            datum["fBasketBytes"][: len(fBasketBytes)] = fBasketBytes
            datum["fBasketEntry"][: len(fBasketEntry)] = fBasketEntry
            datum["fBasketSeek"][: len(fBasketSeek)] = fBasketSeek
            datum["fBasketEntry"][len(fBasketEntry)] = source._num_entries

        
        # for _ in range(num_keys):
        #     key = ReadOnlyKey(
        #         keys_chunk, keys_cursor, {}, file, self, read_strings=True
        #     )
        #     name = key.fName
        #     if name not in self._keys_lookup:
        #         self._keys_lookup[name] = []
        #     self._keys_lookup[name].append(len(self._keys))
        #     self._keys.append(key)

        # __setitem__()
        
    # def find_branch():


    def __repr__(self):
        return "{}({}, {}, {}, {}, {}, {}, {})".format(
            type(self).__name__,
            self._directory,
            self._name,
            self._title,
            [(datum["fName"], datum["branch_type"]) for datum in self._branch_data],
            self._freesegments,
            self._basket_capacity,
            self._resize_factor,
        )

    def write_copy(self, sink):
        key_num_bytes = uproot.reading._key_format_big.size + 6
        name_asbytes = self._name.encode(errors="surrogateescape")
        title_asbytes = self._title.encode(errors="surrogateescape")
        key_num_bytes += (1 if len(name_asbytes) < 255 else 5) + len(name_asbytes)
        key_num_bytes += (1 if len(title_asbytes) < 255 else 5) + len(title_asbytes)

        out = [None]
        ttree_header_index = 0

        tobject = uproot.models.TObject.Model_TObject.empty()
        tnamed = uproot.models.TNamed.Model_TNamed.empty()
        tnamed._bases.append(tobject)
        tnamed._members["fTitle"] = self._title
        tnamed._serialize(out, True, self._name, uproot.const.kMustCleanup)

        # TAttLine v2, fLineColor: 602 fLineStyle: 1 fLineWidth: 1
        # TAttFill v2, fFillColor: 0, fFillStyle: 1001
        # TAttMarker v2, fMarkerColor: 1, fMarkerStyle: 1, fMarkerSize: 1.0
        out.append(
            b"@\x00\x00\x08\x00\x02\x02Z\x00\x01\x00\x01"
            b"@\x00\x00\x06\x00\x02\x00\x00\x03\xe9"
            b"@\x00\x00\n\x00\x02\x00\x01\x00\x01?\x80\x00\x00"
        )

        metadata_out_index = len(out)
        out.append(
            uproot.models.TTree._ttree20_format1.pack(
                self.source.members()["fEntries"],
                self._metadata["fTotBytes"],
                self._metadata["fZipBytes"],
                self._metadata["fSavedBytes"],
                self._metadata["fFlushedBytes"],
                self._metadata["fWeight"],
                self._metadata["fTimerInterval"],
                self._metadata["fScanField"],
                self._metadata["fUpdate"],
                self._metadata["fDefaultEntryOffsetLen"],
                self._metadata["fNClusterRange"],
                self._metadata["fMaxEntries"],
                self._metadata["fMaxEntryLoop"],
                self._metadata["fMaxVirtualSize"],
                self._metadata["fAutoSave"],
                self._metadata["fAutoFlush"],
                self._metadata["fEstimate"],
            )
        )

        # speedbump (0), fClusterRangeEnd (empty array),
        # speedbump (0), fClusterSize (empty array)
        # fIOFeatures (TIOFeatures)
        out.append(b"\x00\x00@\x00\x00\x07\x00\x00\x1a\xa1/\x10\x00")

        tleaf_reference_numbers = []

        tobjarray_of_branches_index = len(out)
        out.append(None)

        num_branches = sum(
            0 if datum["kind"] == "record" else 1 for datum in self._branch_data
        )

        # TObjArray header with fName: ""
        out.append(b"\x00\x01\x00\x00\x00\x00\x03\x00@\x00\x00")
        out.append(
            uproot.models.TObjArray._tobjarray_format1.pack( # still need
                num_branches,  # TObjArray fSize
                0,  # TObjArray fLowerBound
            )
        )

        for datum in self._branch_data:


            
        # out[tobjarray_of_branches_index] = uproot.serialization.numbytes_version(
        #     sum(len(x) for x in out[tobjarray_of_branches_index + 1 :]), 3  # TObjArray
        # )

        # # TObjArray of TLeaf references
        # tleaf_reference_bytes = uproot._util.tobytes(
        #     numpy.array(tleaf_reference_numbers, ">u4")
        # )
        # out.append(
        #     struct.pack(
        #         ">I13sI4s",
        #         (21 + len(tleaf_reference_bytes)) | uproot.const.kByteCountMask,
        #         b"\x00\x03\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x00",
        #         len(tleaf_reference_numbers),
        #         b"\x00\x00\x00\x00",
        #     )
        # )

        # out.append(tleaf_reference_bytes)

        # # null fAliases (b"\x00\x00\x00\x00")
        # # empty fIndexValues array (4-byte length is zero)
        # # empty fIndex array (4-byte length is zero)
        # # null fTreeIndex (b"\x00\x00\x00\x00")
        # # null fFriends (b"\x00\x00\x00\x00")
        # # null fUserInfo (b"\x00\x00\x00\x00")
        # # null fBranchRef (b"\x00\x00\x00\x00")
        # out.append(b"\x00" * 28)

        # out[ttree_header_index] = uproot.serialization.numbytes_version(
        #     sum(len(x) for x in out[ttree_header_index + 1 :]), 20  # TTree
        # )

        self._metadata_start = sum(len(x) for x in out[:metadata_out_index])

        raw_data = b"".join(out)
        self._key = self._directory.add_object(
            sink,
            "TTree",
            self._name,
            self._title,
            raw_data,
            len(raw_data),
            replaces=self._key,
            big=True,
        )

    # # def write_updates(self, sink):
    # #     base = self._key.seek_location + self._key.num_bytes

    # #     sink.write(
    # #         base + self._metadata_start,
    # #         uproot.models.TTree._ttree20_format1.pack(
    # #             self._num_entries,
    # #             self._metadata["fTotBytes"],
    # #             self._metadata["fZipBytes"],
    # #             self._metadata["fSavedBytes"],
    # #             self._metadata["fFlushedBytes"],
    # #             self._metadata["fWeight"],
    # #             self._metadata["fTimerInterval"],
    # #             self._metadata["fScanField"],
    # #             self._metadata["fUpdate"],
    # #             self._metadata["fDefaultEntryOffsetLen"],
    # #             self._metadata["fNClusterRange"],
    # #             self._metadata["fMaxEntries"],
    # #             self._metadata["fMaxEntryLoop"],
    # #             self._metadata["fMaxVirtualSize"],
    # #             self._metadata["fAutoSave"],
    # #             self._metadata["fAutoFlush"],
    # #             self._metadata["fEstimate"],
    # #         ),
    # #     )

    # #     for datum in self._branch_data:
    # #         if datum["kind"] == "record":
    # #             continue

    # #         position = base + datum["metadata_start"]

    # #         # Lie about the compression level so that ROOT checks and does the right thing.
    # #         # https://github.com/root-project/root/blob/87a998d48803bc207288d90038e60ff148827664/tree/tree/src/TBasket.cxx#L560-L578
    # #         # Without this, when small buffers are left uncompressed, ROOT complains about them not being compressed.
    # #         # (I don't know where the "no, really, this is uncompressed" bit is.)
    # #         fCompress = 0

    # #         sink.write(
    # #             position,
    # #             uproot.models.TBranch._tbranch13_format1.pack(
    # #                 fCompress,
    # #                 datum["fBasketSize"],
    # #                 datum["fEntryOffsetLen"],
    # #                 self._num_baskets,  # fWriteBasket
    # #                 self._num_entries,  # fEntryNumber
    # #             ),
    # #         )

    # #         position += uproot.models.TBranch._tbranch13_format1.size + 11
    # #         sink.write(
    # #             position,
    # #             uproot.models.TBranch._tbranch13_format2.pack(
    # #                 datum["fOffset"],
    # #                 self._basket_capacity,  # fMaxBaskets
    # #                 datum["fSplitLevel"],
    # #                 self._num_entries,  # fEntries
    # #                 datum["fFirstEntry"],
    # #                 datum["fTotBytes"],
    # #                 datum["fZipBytes"],
    # #             ),
    # #         )

    # #         start, stop = datum["arrays_write_start"], datum["arrays_write_stop"]

    # #         fBasketBytes_part = uproot._util.tobytes(datum["fBasketBytes"][start:stop])
    # #         fBasketEntry_part = uproot._util.tobytes(
    # #             datum["fBasketEntry"][start : stop + 1]
    # #         )
    # #         fBasketSeek_part = uproot._util.tobytes(datum["fBasketSeek"][start:stop])

    # #         position = base + datum["basket_metadata_start"] + 1
    # #         position += datum["fBasketBytes"][:start].nbytes
    # #         sink.write(position, fBasketBytes_part)
    # #         position += len(fBasketBytes_part)
    # #         position += datum["fBasketBytes"][stop:].nbytes

    # #         position += 1
    # #         position += datum["fBasketEntry"][:start].nbytes
    # #         sink.write(position, fBasketEntry_part)
    # #         position += len(fBasketEntry_part)
    # #         position += datum["fBasketEntry"][stop + 1 :].nbytes

    # #         position += 1
    # #         position += datum["fBasketSeek"][:start].nbytes
    # #         sink.write(position, fBasketSeek_part)
    # #         position += len(fBasketSeek_part)
    # #         position += datum["fBasketSeek"][stop:].nbytes

    # #         datum["arrays_write_start"] = datum["arrays_write_stop"]

    # #         if datum["dtype"] == ">U0":
    # #             position = (
    # #                 base
    # #                 + datum["basket_metadata_start"]
    # #                 - 25  # empty TObjArray of fBaskets (embedded)
    # #                 - 8  # specialized TLeaf* members (fMinimum, fMaximum)
    # #                 - 4  # null fLeafCount
    # #                 - 14  # generic TLeaf members
    # #             )
    # #             sink.write(
    # #                 position,
    # #                 uproot.models.TLeaf._tleaf2_format0.pack(
    # #                     self._metadata["fLen"],
    # #                     datum["dtype"].itemsize,
    # #                     0,
    # #                     datum["kind"] == "counter",
    # #                     _dtype_to_char[datum["dtype"]]
    # #                     != _dtype_to_char[datum["dtype"]].upper(),
    # #                 ),
    # #             )

    # #         if datum["kind"] == "counter":
    # #             position = (
    # #                 base
    # #                 + datum["basket_metadata_start"]
    # #                 - 25  # empty TObjArray of fBaskets (embedded)
    # #                 - datum["tleaf_special_struct"].size
    # #             )
    # #             sink.write(
    # #                 position,
    # #                 datum["tleaf_special_struct"].pack(
    # #                     0,
    # #                     datum["tleaf_maximum_value"],
    # #                 ),
    # #             )

    # #     sink.flush()


    # def copy_basket(self, sink, branch_name, compression, array):
    #     fClassName = uproot.serialization.string("TBasket")
    #     fName = uproot.serialization.string(branch_name)
    #     fTitle = uproot.serialization.string(self._name)

    #     fKeylen = (
    #         uproot.reading._key_format_big.size
    #         + len(fClassName)
    #         + len(fName)
    #         + len(fTitle)
    #         + uproot.models.TBasket._tbasket_format2.size
    #         + 1
    #     )

    #     itemsize = array.dtype.itemsize
    #     for item in array.shape[1:]:
    #         itemsize *= item

    #     uncompressed_data = uproot._util.tobytes(array)
    #     compressed_data = uproot.compression.compress(uncompressed_data, compression)

    #     fObjlen = len(uncompressed_data)
    #     fNbytes = fKeylen + len(compressed_data)

    #     parent_location = self._directory.key.location  # FIXME: is this correct?

    #     location = self._freesegments.allocate(fNbytes, dry_run=False)

    #     out = []
    #     out.append(
    #         uproot.reading._key_format_big.pack(
    #             fNbytes,
    #             1004,  # fVersion
    #             fObjlen,
    #             uproot._util.datetime_to_code(datetime.datetime.now()),  # fDatime
    #             fKeylen,
    #             0,  # fCycle
    #             location,  # fSeekKey
    #             parent_location,  # fSeekPdir
    #         )
    #     )
    #     out.append(fClassName)
    #     out.append(fName)
    #     out.append(fTitle)
    #     out.append(
    #         uproot.models.TBasket._tbasket_format2.pack(
    #             3,  # fVersion
    #             32000,  # fBufferSize
    #             itemsize,  # fNevBufSize
    #             len(array),  # fNevBuf
    #             fKeylen + len(uncompressed_data),  # fLast
    #         )
    #     )
    #     out.append(b"\x00")  # part of the Key (included in fKeylen, at least)

    #     out.append(compressed_data)

    #     sink.write(location, b"".join(out))
    #     self._freesegments.write(sink)
    #     sink.set_file_length(self._freesegments.fileheader.end)
    #     sink.flush()

    #     return fKeylen + fObjlen, fNbytes, location

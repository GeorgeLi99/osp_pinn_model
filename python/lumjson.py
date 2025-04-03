"""
Copyright (c) 2024 ANSYS, Inc.

This program is commercial software: you can use it under the terms of the
Ansys License Agreement as published by ANSYS, Inc.

Except as expressly permitted in the Ansys License Agreement, you may not
modify or redistribute this program.
"""

"""
Example Usage: 

import json
from lumjson import LumEncoder, LumDecoder
with open(fname) as f:
   data = json.load(f, cls=LumDecoder)
   json.dump(data, f, cls=LumEncoder)
"""

from functools import reduce
import json
import numpy as np


class LumEncoder(json.JSONEncoder):
    TYPE_NODE_NAME = "_type"
    DATA_NODE_NAME = "_data"
    SIZE_NODE_NAME = "_size"
    COMPLEX_NODE_NAME = "_complex"

    TYPE_NODE_NAME_MATRIX = "matrix"
    TYPE_NODE_NAME_CELL = "cell"

    def default(self, obj):  # pylint: disable=E0202
        if isinstance(obj, list) or isinstance(obj, tuple):
            return {
                self.TYPE_NODE_NAME: self.TYPE_NODE_NAME_CELL,
                self.DATA_NODE_NAME: obj
            }

        elif isinstance(obj, complex):
            return {
                self.TYPE_NODE_NAME: self.TYPE_NODE_NAME_MATRIX,
                self.SIZE_NODE_NAME: [1, 1],
                self.COMPLEX_NODE_NAME: True,
                self.DATA_NODE_NAME: [obj.real, obj.imag]
            }

        elif isinstance(obj, np.ndarray):
            is_complex = np.iscomplexobj(obj)
            dimlist = obj.shape
            length = reduce(lambda x, y: x*y, dimlist)
            data = []

            if is_complex:
                iterator = obj.flatten(order='F')
                for i in range(length):
                    data.append(iterator[i].real)
                    data.append(iterator[i].imag)
            else:
                data = list(obj.flatten(order='F'))
            return {
                self.TYPE_NODE_NAME: self.TYPE_NODE_NAME_MATRIX,
                self.SIZE_NODE_NAME: dimlist,
                self.COMPLEX_NODE_NAME: is_complex,
                self.DATA_NODE_NAME: data
            }
        return json.JSONEncoder.default(self, obj)


class LumDecoder(json.JSONDecoder):
    TYPE_NODE_NAME = "_type"
    DATA_NODE_NAME = "_data"
    SIZE_NODE_NAME = "_size"
    COMPLEX_NODE_NAME = "_complex"

    TYPE_NODE_NAME_MATRIX = "matrix"
    TYPE_NODE_NAME_CELL = "cell"

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(
            self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):  # pylint: disable=E0202
        if self.TYPE_NODE_NAME in dct and dct[self.TYPE_NODE_NAME] == self.TYPE_NODE_NAME_MATRIX:
            data = dct[self.DATA_NODE_NAME]
            is_complex = False if self.COMPLEX_NODE_NAME not in dct else dct[self.COMPLEX_NODE_NAME]
            
            if self.SIZE_NODE_NAME not in dct:
                return np.asarray(data, dtype="float64")
            
            dimlist = dct[self.SIZE_NODE_NAME]
            length = reduce(lambda x, y: x*y, dimlist)
            
            if length == 1 and is_complex:
                return complex(data[0], data[1])

            elif length == 1 and not is_complex:
                return data[0]

            elif is_complex:
                r = np.empty(length, dtype=complex, order='F')
                for i in range(length):
                    r[i] = complex(data[2*i], data[2*i+1])
                r = r.reshape(dimlist, order='F')
                return r

            else:
                r = np.empty(length, dtype="float64", order='F')
                for i in range(length):
                    r[i] = data[i]
                r = r.reshape(dimlist, order='F')
                
                return r

        elif self.TYPE_NODE_NAME in dct and dct[self.TYPE_NODE_NAME] == self.TYPE_NODE_NAME_CELL:
            return dct[self.DATA_NODE_NAME]

        return dct

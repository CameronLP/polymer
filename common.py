#!/usr/bin/env python
# -*- coding: utf-8 -*-


L2FLAGS = {
        'LAND'          : 1,
        'CLOUD_BASE'    : 2,
        'L1_INVALID'    : 4,
        'NEGATIVE_BB'   : 8,
        'OUT_OF_BOUNDS' : 16,
        'EXCEPTION'     : 32,
        'EXTERNAL_MASK' : 512,
        'CASE2'         : 1024,
        }

# no product in case of...
BITMASK_INVALID = 1+2+4+32+512


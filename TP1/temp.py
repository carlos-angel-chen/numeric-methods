# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

print("Oh Yeah Mr. Krabs")

import struct
import binascii

print(binascii.hexlify(struct.pack('<d',+0)))

print(binascii.hexlify(struct.pack('<d',float('inf'))))

print(binascii.hexlify(struct.pack('<d',float('-inf'))))

print(binascii.hexlify(struct.pack('<d',float('nan'))))

print(binascii.hexlify(struct.pack('<d',+0)))

print(binascii.hexlify(struct.pack('<d',+0.0)))

print(binascii.hexlify(struct.pack('<d',-0.0)))

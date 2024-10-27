# -*- coding: utf-8 -*-
from pwn import ELF
from collections import Counter
from itertools import islice
import base64

class FileEntry(object):
    def __init__(self, binary_path, num_bytes):
        self.binary_path = binary_path
        self.num_bytes = num_bytes
        self.key = 0xAA
        
    def get_features(self):
        
        try: 
            elf = ELF(self.binary_path)
         
            entry_point = elf.entry
             
            data = elf.read(entry_point, self.num_bytes)
            
            data = bytes([b ^ self.key for b in data])
            
            data = base64.b64encode(data).decode('utf-8')
            
            bigrams = [data[i:i+2] for i in range(len(data)-1)]
            
            return str(bigrams)
         
        except Exception as e:
             return str([])
          
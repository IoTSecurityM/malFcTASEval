# -*- coding: utf-8 -*-
from pwn import ELF
from collections import Counter
from itertools import islice


class ELFEntry(object):
    def __init__(self, binary_path, L, N):
        self.binary_path = binary_path
        self.L = L
        self.N = N
        
    def get_features(self):
        
        try: 
            elf = ELF(self.binary_path)
         
            entry_point = elf.entry
             
            data = elf.read(entry_point, self.L)
            
            return str(data)
         
        except Exception as e:
             return str([])
           

# import r2pipe
# from collections import Counter

# class ELFEntry(object):

#     def __init__(self, binary_path, L, N):
#         self.binary_path = binary_path
#         self.L = L
#         self.N = N    
        
#     def get_features(self):         
#         r2 = r2pipe.open(self.binary_path)
#         r2.cmd('aaa') 
#         entry_point = int(r2.cmd('ie~entry').split(' ')[-1], 16) 
#         print(entry_point)
#         r2.cmd(f"s {entry_point}")
#         byte_sequence = r2.cmdj(f"pxj {self.L}")
#         r2.quit()
    
#         data = bytearray(byte_sequence)
#         ngrams = [data[i:i+self.N] for i in range(len(data) - self.N + 1)]
          
#         return ngrams

    
# binary_path = "/home/yefei/WSPACE/DtComb/IoTXPOT/68K/0a0a3c31bfa75d76c23ba6eef15f8a21919b87e2585a28254a46426c17dd29c8"
# elfentry = ELFEntry(binary_path, 1024, 4)
# elfentry.get_features()
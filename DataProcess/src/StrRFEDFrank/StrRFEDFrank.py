# -*- coding: utf-8 -*-
import subprocess
import math

class StrRFEDFrank(object):
    def __init__(self, str_list, max_len):        
        self.str_list = str_list
        self.max_len = max_len
        self.bin_list = [(math.exp((bin_num-1) * math.log(self.max_len) / 50), math.exp(bin_num * math.log(self.max_len) / 50)) for bin_num in range(1,51)]
        
    def get_features(self, binary_path):      
        def extract_strings(file_path):
            result = subprocess.run(["strings", file_path], text=True, capture_output=True)
            return result.stdout.splitlines()    
        
        strings_output = extract_strings(binary_path)
        if len(strings_output) > 0:
            strings_len_list = list(map(len, strings_output))
            slf = {bin_symbol: len([str_len for str_len in strings_len_list if bin_symbol[0] < str_len < bin_symbol[1]]) for bin_symbol in self.bin_list}
            redu_str_output = set(strings_output)        
            psi = {key: (1 if key in redu_str_output else 0) for key in self.str_list}
        else:
            slf = {bin_symbol: 0 for bin_symbol in self.bin_list}
            psi = {key: 0 for key in self.str_list}
        
        features = list(slf.values()) + list(psi.values())
        
        return features


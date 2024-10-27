# -*- coding: utf-8 -*-
import r2pipe
import subprocess
import yara
import re
from collections import defaultdict

class ELFOpocde(object):
    def __init__(self, binary_path):
        self.binary_path = binary_path
        
        self.architecture = ''
        self.size = 0
        self.dependencies = 0
        self.packer = 0
        self.function_num = 0
        self.network_state = 0
        self.other_ability = 0
        self.opcode_fea = []

        self.r2 = r2pipe.open(binary_path)
        self.r2.cmd('aaa')
        self.functions = self.r2.cmdj('aflj')
        
    def get_binary_details(self):   
        arch = self.r2.cmd('iI~arch')  # Get architecture
        size = self.r2.cmd('i~size')  # Get size
        archtect = arch.split()[-1]
        if archtect == 'architecture':
            archtect = "x64"          
        return archtect, int(size.split()[-1], 16)
    
    def check_dependencies(self):
        try:
          # Run ldd and capture both stdout and stderr
          result = subprocess.run(['ldd', self.binary_path], text=True, capture_output=True, check=True)
          return 0
        except subprocess.CalledProcessError as e:
          # Return both the error message and the output that might explain the issue
          return 1
        except FileNotFoundError:
          # Handle the case where the ldd command is not found
          return "ldd command not found. Please ensure it is installed and in your PATH."
      
    def pack_check(self):
        
        rules = yara.compile(filepaths={
            'jeo': '...', # yar position
            'jj': '...',
            'packer': '...',
            'pcs': '...',
            'peid': '...',
            'tpp': '...'})

        match = rules.match(self.binary_path)
        packer  = 0 if len(match)!=0 else 1
        return packer
    
    def func_num(self):
          # 'aflj' lists functions in JSON format
        function_num = len(self.functions)
        return function_num
        
    def network_func(self):
        
        function_names = [func['name'] for func in self.functions]

        def extract_strings(file_path):
            result = subprocess.run(["strings", file_path], text=True, capture_output=True)
            return result.stdout.splitlines()

        strings_output = extract_strings(self.binary_path)

        ip_pattern = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
        domain_pattern = re.compile(r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9][a-z0-9-]{0,61}[a-z0-9]\b")

        network_functions = ['socket', 'connect', 'send', 'recv', 'http', "tcp", "udp", "https", "ftp", "dns", "smtp", "ssh", "ssl", "tls", "icmp"]

        ip_addresses = [string for string in strings_output if ip_pattern.search(string)]
        domains = [string for string in strings_output if domain_pattern.search(string)]
        network_related_functions = [fn for fn in function_names if any(nf in fn for nf in network_functions)]

        network_state = 1 if (ip_addresses or domains) and network_related_functions else 0
        return network_state
        
    def other_abilit(self):
        
        other_rules = yara.compile(filepaths={
            'antivm': '...', # yar position
            'ang': '...',
            'black': '...',
            'blife': '...',
            'crimpack': '...',
            'eleonore': '...',
            'sakura': '...',
            'phoenix': '...',
            'fragus': '...',
            'zeroaccess': '...',
            'zerox88': '...',
            'zeus': '...',
            })

        other_match = other_rules.match(self.binary_path)

        other_ability = 0 if len(other_match)!=0 else 1        
        return other_ability
        
    def opcode_features(self):
        opcode_types_by_isa = {
            "x86":{"or": 'Logic', "cmp": 'ContStatus', "mov": 'Memory', "push": 'Stack', "call": 'Procedure', "es": 'Prefixed', "in": 'SystemIO', "add": 'Arithmetic', "hlt": 'System', "jmp": 'Branch', "wait": 'ExecutTime', "cbw": 'Others'},
            "x64": {"xor": 'Logic', "test": 'ContStatus', "lea": 'Memory', "pop": 'Stack', "leave": 'Procedure', "ds": 'Prefixed', "out": 'SystemIO', "sub": 'Arithmetic', "arpl": 'System', "retn": 'Branch', "fwait": 'ExecutTime', "cdq": 'Others'},
            "mips":{"nor": 'Logic', "slti": 'ContStatus', "lw": 'Memory', "break": 'Procedure', "mult": 'Arithmetic', "bltz": 'Branch'},
            "arm": {"bic": 'Logic', "cmn": 'ContStatus', "ldc": 'Memory', "srs": 'Stack', "hvc": 'Procedure', "asr": 'Arithmetic', "sys": 'System', "bl": 'Branch', "wfe": 'ExecutTime', "dsb": 'Others'},
            "sparc": {"xnor": 'Logic', "fcmp": 'ContStatus', "restore": 'Memory', "call": 'Procedure', "fabs": 'Arithmetic', "bicc": 'Branch'},
            "ppc": {"xori": 'Logic', "dcbi": 'ContStatus', "lswi": 'Memory', "svc": 'Procedure', "abs": 'Arithmetic', "ti": 'System', "bclr": 'Branch'},
            "sh": {"not": 'Logic', "tas": 'ContStatus', "sts": 'Memory', "mac": 'Arithmetic', "bsrf": 'Branch'},
            "m68k":{"eor": 'Logic', "btst": 'ContStatus', "usp": 'Memory', "pea": 'Stack', "neg": 'Arithmetic', "trap": 'System', "rte": 'Branch'}       
        }
        
        opcode_types = opcode_types_by_isa[self.architecture]

        opcode_categories = ['Logic', 'ContStatus', 'Memory', 'Stack', 'Procedure', 'Prefixed', 'SystemIO', 'Arithmetic', 'System', 'Branch', 'ExecutTime', 'Others']
        opcode_count = {category: 0 for category in opcode_categories}             
        
        total_opcode = 0
        def is_instruction(line):
           # Typically, instructions lines start with an address and contain opcodes
           parts = line.strip().split()
           if len(parts) < 3:
               return False
           try:
               int(parts[1], 16)  # Check if the first part is a valid address
               return True
           except ValueError:
               return False
        # Disassemble each function and categorize opcodes
        for function in self.functions:
            function_offset = function['offset']
            disasm = self.r2.cmd(f'pdr @ {function_offset}')          
            for line in disasm.splitlines():
                # Regex to extract the opcode from the disassembly line
                if is_instruction(line):
                    total_opcode += 1
                    match = re.search(r'\s([a-zA-Z]+)\s', line)
                    if match:
                        opcode = match.group(1)                                               
                        if opcode.lower() in opcode_types:                           
                            opcode_count[opcode_types[opcode.lower()]] += 1                  
        
        opcode_fea = list(opcode_count.values())
        opcode_fea.append(total_opcode)
        return opcode_fea
        
    def get_features(self):    
        
        self.architecture, self.size = self.get_binary_details()
        
        if self.architecture in ["x86", "x64", "mips", "arm", "sparc", "ppc", "sh", "m68k"]:
            self.dependencies = self.check_dependencies()
            self.packer = self.pack_check()
            self.function_num = self.func_num()
            self.network_state = self.network_func()
            self.other_ability = self.other_abilit()
            self.opcode_fea = self.opcode_features()        
            return [self.architecture, self.size, self.dependencies, self.packer, self.function_num, self.network_state, self.other_ability] + self.opcode_fea
        else:
            return None
        
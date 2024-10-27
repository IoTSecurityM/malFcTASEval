# -*- coding: utf-8 -*-
import pandas as pd
import os
import csv
import json
import ast

from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

# from ELFMiner import ELFMiner

# class DtELFMiner(object):
#     def __init__(self, annotations_file, mal_dir):
#         self.mal_dir = mal_dir
#         self.annotations_file = annotations_file
        
#         self.mal_labels = pd.read_csv(self.annotations_file)
        
#         self.fea_csv = "../../Features/ELFMiner.csv"
#         if os.path.exists(self.fea_csv):
#             # Remove the file
#             os.remove(self.fea_csv)
        
#         headers = ['location', 'label', 'endianness', 'elf_class', 'version', 'elf_type', 'machine', 'entry_point', 'flags', 'elf_header_size', 'program_header_size', 
#                    'program_header_num', 'section_header_size', 'section_header_num', 'section_name_index', '.text_section_type', '.text_section_flags', 
#                    '.text_section_size', '.text_section_link', '.text_section_info', '.text_section_addr_align', '.bss_section_type', '.bss_section_flags', 
#                    '.bss_section_size', '.bss_section_link', '.bss_section_info', '.bss_section_addr_align', '.comment_section_type', '.comment_section_flags', 
#                    '.comment_section_size', '.comment_section_link', '.comment_section_info', '.comment_section_addr_align', '.data_section_type', 
#                    '.data_section_flags', '.data_section_size', '.data_section_link', '.data_section_info', '.data_section_addr_align', '.data1_section_type', 
#                    '.data1_section_flags', '.data1_section_size', '.data1_section_link', '.data1_section_info', '.data1_section_addr_align', '.debug_section_type', 
#                    '.debug_section_flags', '.debug_section_size', '.debug_section_link', '.debug_section_info', '.debug_section_addr_align', '.dynamic_section_type', 
#                    '.dynamic_section_flags', '.dynamic_section_size', '.dynamic_section_link', '.dynamic_section_info', '.dynamic_section_addr_align', 
#                    '.dynstr_section_type', '.dynstr_section_flags', '.dynstr_section_size', '.dynstr_section_link', '.dynstr_section_info', '.dynstr_section_addr_align', 
#                    '.dynsym_section_type', '.dynsym_section_flags', '.dynsym_section_size', '.dynsym_section_link', '.dynsym_section_info', '.dynsym_section_addr_align', 
#                    '.fini_section_type', '.fini_section_flags', '.fini_section_size', '.fini_section_link', '.fini_section_info', '.fini_section_addr_align', 
#                    '.hash_section_type', '.hash_section_flags', '.hash_section_size', '.hash_section_link', '.hash_section_info', '.hash_section_addr_align', 
#                    '.init_section_type', '.init_section_flags', '.init_section_size', '.init_section_link', '.init_section_info', '.init_section_addr_align', 
#                    '.got_section_type', '.got_section_flags', '.got_section_size', '.got_section_link', '.got_section_info', '.got_section_addr_align', 
#                    '.interp_section_type', '.interp_section_flags', '.interp_section_size', '.interp_section_link', '.interp_section_info', '.interp_section_addr_align', 
#                    '.line_section_type', '.line_section_flags', '.line_section_size', '.line_section_link', '.line_section_info', '.line_section_addr_align', 
#                    '.note_section_type', '.note_section_flags', '.note_section_size', '.note_section_link', '.note_section_info', '.note_section_addr_align', 
#                    '.plt_section_type', '.plt_section_flags', '.plt_section_size', '.plt_section_link', '.plt_section_info', '.plt_section_addr_align', 
#                    '.rodata_section_type', '.rodata_section_flags', '.rodata_section_size', '.rodata_section_link', '.rodata_section_info', '.rodata_section_addr_align', 
#                    '.rodata1_section_type', '.rodata1_section_flags', '.rodata1_section_size', '.rodata1_section_link', '.rodata1_section_info', 
#                    '.rodata1_section_addr_align', '.shstrtab_section_type', '.shstrtab_section_flags', '.shstrtab_section_size', '.shstrtab_section_link', 
#                    '.shstrtab_section_info', '.shstrtab_section_addr_align', '.strtab_section_type', '.strtab_section_flags', '.strtab_section_size', 
#                    '.strtab_section_link', '.strtab_section_info', '.strtab_section_addr_align', '.symtab_section_type', '.symtab_section_flags', '.symtab_section_size', 
#                    '.symtab_section_link', '.symtab_section_info', '.symtab_section_addr_align', '.sdata_section_type', '.sdata_section_flags', '.sdata_section_size', 
#                    '.sdata_section_link', '.sdata_section_info', '.sdata_section_addr_align', '.sbss_section_type', '.sbss_section_flags', '.sbss_section_size', 
#                    '.sbss_section_link', '.sbss_section_info', '.sbss_section_addr_align', '.lit8_section_type', '.lit8_section_flags', '.lit8_section_size', 
#                    '.lit8_section_link', '.lit8_section_info', '.lit8_section_addr_align', '.gptab_section_type', '.gptab_section_flags', '.gptab_section_size', 
#                    '.gptab_section_link', '.gptab_section_info', '.gptab_section_addr_align', '.conflict_section_type', '.conflict_section_flags', 
#                    '.conflict_section_size', '.conflict_section_link', '.conflict_section_info', '.conflict_section_addr_align', '.tdesc_section_type', 
#                    '.tdesc_section_flags', '.tdesc_section_size', '.tdesc_section_link', '.tdesc_section_info', '.tdesc_section_addr_align', '.lit4_section_type', 
#                    '.lit4_section_flags', '.lit4_section_size', '.lit4_section_link', '.lit4_section_info', '.lit4_section_addr_align', '.reginfo_section_type', 
#                    '.reginfo_section_flags', '.reginfo_section_size', '.reginfo_section_link', '.reginfo_section_info', '.reginfo_section_addr_align', 
#                    '.liblist_section_type', '.liblist_section_flags', '.liblist_section_size', '.liblist_section_link', '.liblist_section_info', 
#                    '.liblist_section_addr_align', '.rel.dyn_section_type', '.rel.dyn_section_flags', '.rel.dyn_section_size', '.rel.dyn_section_link', 
#                    '.rel.dyn_section_info', '.rel.dyn_section_addr_align', '.rel.plt_section_type', '.rel.plt_section_flags', '.rel.plt_section_size', 
#                    '.rel.plt_section_link', '.rel.plt_section_info', '.rel.plt_section_addr_align', '.got.plt_section_type', '.got.plt_section_flags', 
#                    '.got.plt_section_size', '.got.plt_section_link', '.got.plt_section_info', '.got.plt_section_addr_align', '0_segment_type', '0_file_size', 
#                    '0_memory_size', '0_flags', '0_alignment', '1_segment_type', '1_file_size', '1_memory_size', '1_flags', '1_alignment', '2_segment_type', 
#                    '2_file_size', '2_memory_size', '2_flags', '2_alignment', '3_segment_type', '3_file_size', '3_memory_size', '3_flags', '3_alignment', 
#                    '4_segment_type', '4_file_size', '4_memory_size', '4_flags', '4_alignment', '5_segment_type', '5_file_size', '5_memory_size', '5_flags', '5_alignment', 
#                    '6_segment_type', '6_file_size', '6_memory_size', '6_flags', '6_alignment', '7_segment_type', '7_file_size', '7_memory_size', '7_flags', '7_alignment', 
#                    '8_segment_type', '8_file_size', '8_memory_size', '8_flags', '8_alignment', 'symtab_total_symbols', 'symtab_local_symbols', 'symtab_global_symbols', 
#                    'symtab_weak_symbols', 'symtab_stb_lo_proc_symbols', 'symtab_stb_hi_proc_symbols', 'symtab_stt_notype', 'symtab_local_objects', 'symtab_global_objects', 
#                    'symtab_weak_objects', 'symtab_local_functions', 'symtab_global_functions', 'symtab_weak_functions', 'symtab_sections', 'symtab_files', 
#                    'symtab_stt_lo_proc_objects', 'symtab_stt_hi_proc_objects', 'total_entries', 'null_entries', 'needed_libraries', 'plt_relocation_size', 'hiproc_entries',
#                    'plt_got_entries', 'hash_table_entries', 'string_table_entries', 'symbol_table_entries', 'rela_entries', 'rela_size', 'rela_entry_size', 
#                    'string_table_size', 'symbol_table_entry_size', 'init_function_address', 'fini_function_address', 'soname_entries', 'rpath_entries', 
#                    'symbolic_entries', 'rel_entries', 'rel_size', 'rel_entry_size', 'plt_relocation_type', 'debug_entries', 'text_relocation_entries', 'jmprel_entries', 
#                    'loproc_entries', 'dynsym_total_symbols', 'dynsym_local_symbols', 'dynsym_global_symbols', 'dynsym_weak_symbols', 'dynsym_stb_lo_proc_symbols', 
#                    'dynsym_stb_hi_proc_symbols', 'dynsym_stt_notype', 'dynsym_local_objects', 'dynsym_global_objects', 'dynsym_weak_objects', 'dynsym_local_functions', 
#                    'dynsym_global_functions', 'dynsym_weak_functions', 'dynsym_sections', 'dynsym_files', 'dynsym_stt_lo_proc_objects', 'dynsym_stt_hi_proc_objects', 
#                    'got_size', 'hash_table_size']
        
#         with open(self.fea_csv, mode='w', newline='') as file:
#             writer = csv.writer(file)      
#             writer.writerow(headers)     
        
#     def  __len__(self):
#         return len(self.mal_labels)
    
#     def fea_gen(self):
        
#         for index, row in self.mal_labels.iterrows():
#             mal_path = self.mal_dir + row[0]   
#             elfminer = ELFMiner(mal_path)
            
#             fea = elfminer.get_features()
            
#             if fea != None:            
#                 whole_fea = [row[0], row[1]] + fea               
#                 with open(self.fea_csv, mode='a', newline='') as file:
#                     writer = csv.writer(file)
#                     writer.writerow(whole_fea)
                    
# def generate_fea():
    
#     mal_dir = '...'  # dataset position
#     annotations_file = '...' # dataset annotation position
#     mal_dataset = DtELFMiner(annotations_file=annotations_file, mal_dir=mal_dir)
#     mal_dataset.fea_gen()         

          
class REGDtELFMiner(object):
    
    def __init__(self, annotations_file, who_fea_file, dtanno_file):
        self.who_fea_file = who_fea_file
        self.annotations_file = annotations_file
        self.dtanno_file = dtanno_file
        
        self.year_list = [2020, 2021, 2022]
        self.arch_list = ['ARM', 'MIPS']
        
        self.who_fea = pd.read_csv(self.who_fea_file)
        self.train_mal_labels = pd.read_excel(self.annotations_file, sheet_name="whole")   
        self.test_mal_labels = pd.read_excel(self.annotations_file, sheet_name="test_samspa")   
               
        self.fea_csv = "../../Features/ELFMiner/SamSpa/ELFMiner"+self.dtanno_file[0:-5]+".xlsx"
        self.fea_filter = "../../Features/ELFMiner/SamSpa/FeaELFMiner"+self.dtanno_file[0:-5]+".txt"
        
        self.fea_filter_infor = {train_year:{arch: [] for arch in self.arch_list} for train_year in self.year_list}
        
    def  __len__(self):
        return len(self.train_mal_labels)
    
    def feature_type_transform(self, X, exclude_columns=None):
        
        if exclude_columns is None:
            exclude_columns = []
        
        X.fillna(-1, inplace=True)
                       
        for column in X.columns:
            if column not in exclude_columns:
                if X[column].dtype == 'object':  
                    X[column] = X[column].replace(-1, '-1')
      
        le = LabelEncoder()
        for column in X.columns:
            if column not in exclude_columns:
                if X[column].dtype == 'object':  # check if the column type is 'object' (usually strings)
                    X[column] = le.fit_transform(X[column])
                
        return X
     
    def get_dt_fea(self):
        
        for train_year in self.year_list:            
            for train_arch in self.arch_list:                
                train_result_df = self.who_fea[self.who_fea['location'].isin(self.train_mal_labels[(self.train_mal_labels['year']==train_year)&(self.train_mal_labels['arch']==train_arch)]['location'])]                         
                train_result_df = train_result_df.merge(self.train_mal_labels[['location', 'year', 'arch']], on='location', how='left')
                train_result_df.reset_index(drop=True, inplace=True)                    
                
                X = train_result_df.drop(columns=['location', 'year', 'arch', 'label'])
                y = train_result_df['label']
                        
                X = self.feature_type_transform(X)
                
                info_gain = mutual_info_classif(X, y)
                selector = SelectKBest(score_func=mutual_info_classif, k=88)
                X_new = selector.fit_transform(X, y)
                
                selected_features = X.columns[selector.get_support()]
                self.fea_filter_infor[train_year][train_arch] = list(selected_features )
                
        with open(self.fea_filter, 'w') as file:
            json.dump(self.fea_filter_infor, file, indent=4) 
            
    def fea_gen(self):
        
        with open(self.fea_filter, 'r') as file:
            self.fea_filter_infor = json.load(file)
            self.fea_filter_infor = {ast.literal_eval(k): v for k, v in self.fea_filter_infor.items()}
        
        with pd.ExcelWriter(self.fea_csv, engine='xlsxwriter') as writer:
        
            for train_year in self.year_list:  
                
                for train_arch in self.arch_list: 
                    
                    fea_names = self.fea_filter_infor[train_year][train_arch]
                    
                    train_result_df = self.who_fea[self.who_fea['location'].isin(self.train_mal_labels[(self.train_mal_labels['year']==train_year)&(self.train_mal_labels['arch']==train_arch)]['location'])]                         
                    train_result_df = train_result_df.merge(self.train_mal_labels[['location', 'year', 'arch']], on='location', how='left')
                    train_result_df.reset_index(drop=True, inplace=True) 
                    train_result_df = train_result_df[['location', 'label', 'year', 'arch']+fea_names]                                     
                    train_result_df = self.feature_type_transform(train_result_df, ['location', 'label', 'year', 'arch'])                
                    train_result_df.to_excel(writer, sheet_name=str(train_year)+train_arch, index=False)
                
                    test_result_df_1 = self.who_fea[self.who_fea['location'].isin(self.train_mal_labels[~((self.train_mal_labels['year']==train_year)&(self.train_mal_labels['arch']==train_arch))]['location'])]   
                    test_result_df_1 = test_result_df_1.merge(self.train_mal_labels[['location', 'year', 'arch']], on='location', how='left')
                    
                    test_result_df_2 = self.who_fea[self.who_fea['location'].isin(self.test_mal_labels[(self.test_mal_labels['year']==train_year)&(self.test_mal_labels['arch']==train_arch)]['location'])]                                     
                    test_result_df_2 = test_result_df_2.merge(self.test_mal_labels[['location', 'year', 'arch']], on='location', how='left')
                    
                    test_result_df = pd.concat([test_result_df_1, test_result_df_2], axis=0, ignore_index=True)
                    test_result_df.reset_index(drop=True, inplace=True)
                    test_result_df = test_result_df[['location', 'label', 'year', 'arch']+fea_names]   
                    test_result_df = self.feature_type_transform(test_result_df, ['location', 'label', 'year', 'arch'])                
                    test_result_df.to_excel(writer, sheet_name=str(train_year)+train_arch+"_test", index=False)
                    

    
def REGgenerate_fea():
    
    train_dtanno_dir = '...' # annotation files postion    
    who_fea_file = "../../Features/ELFMiner.csv" 
    
    for dtanno_file in ['dt'+str(i)+".xlsx" for i in range(0,5)]:
        train_dtanno_path = train_dtanno_dir + dtanno_file    
        mal_dataset = REGDtELFMiner(annotations_file=train_dtanno_path, who_fea_file=who_fea_file, dtanno_file=dtanno_file)
        mal_dataset.fea_gen()

if __name__ == "__main__":
    # generate_fea()   
    REGgenerate_fea()

# -*- coding: utf-8 -*-
import pandas as pd
import csv
import os

# from ELFOpcode import ELFOpocde

# class DtELFOpcode(object):
#     def __init__(self, annotations_file, mal_dir):
#         self.mal_dir = mal_dir
#         self.annotations_file = annotations_file
        
#         self.mal_labels = pd.read_csv(self.annotations_file)
        
#         self.fea_csv = "Features/ELFOpocode.csv"
#         if os.path.exists(self.fea_csv):
#             # Remove the file
#             os.remove(self.fea_csv)
        
#         headers = ['location', 'label', 'arch', 'size', 'dependencies', 'packer', 'function_num', 'network_state', 'other_ability', 'Logic', 'ContStatus', 'Memory', 'Stack',\
#                   'Procedure', 'Prefixed', 'SystemIO', 'Arithmetic', 'System', 'Branch', 'ExecutTime', 'Others', 'totalOpcode']
#         with open(self.fea_csv, mode='w', newline='') as file:
#             writer = csv.writer(file)      
#             writer.writerow(headers)     
        
#     def  __len__(self):
#         return len(self.mal_labels)
    
#     def fea_gen(self):
        
#         for index, row in self.mal_labels.iterrows():
#             mal_path = self.mal_dir + row[0]   
#             elfopcode = ELFOpocde(mal_path)
            
#             fea = elfopcode.get_features()
            
#             if fea != None:            
#                 whole_fea = [row[0], row[1]] + fea               
#                 with open(self.fea_csv, mode='a', newline='') as file:
#                     writer = csv.writer(file)
#                     writer.writerow(whole_fea)


# def generate_fea():
    
#     mal_dir = '...'  # dataset position
#     annotations_file = '...' # dataset annotation position
#     mal_dataset = DtELFOpcode(annotations_file=annotations_file, mal_dir=mal_dir)
#     mal_dataset.fea_gen()
      

class REGREGDtELFOpcode(object):
    
    def __init__(self, annotations_file, who_fea_file, dtanno_file):
        self.who_fea_file = who_fea_file
        self.annotations_file = annotations_file
        self.dtanno_file = dtanno_file
        
        self.year_list = [2020, 2021, 2022]
        self.arch_list = ['ARM', 'MIPS']
        
        self.who_fea = pd.read_csv(self.who_fea_file)
        self.train_mal_labels = pd.read_excel(self.annotations_file, sheet_name="whole")   
        self.test_mal_labels = pd.read_excel(self.annotations_file, sheet_name="test_diffspa")   
               
        self.fea_csv = "../../Features/ELFOpocode/DiffSpa/ELFOpocode"+self.dtanno_file[0:-5]+".xlsx"
        
    def  __len__(self):
        return len(self.mal_labels)
    
    def ISA_proc(self, value):
        ISA_identifier = {"arm": 1, "mips": 2, "sh": 3, "m68k":4}
        return ISA_identifier[value]
    
    def size_proc(self, value):
        value = value/1000
        if 0 < value <= 20:
            return 1
        elif 20 < value <= 40:
            return 2
        elif 40 < value <= 60:
            return 3
        elif 60 < value <= 80:
            return 4
        elif 80 < value <= 100:
            return 5
        elif 100 < value <= 120:
            return 6
        elif 120 < value <= 140:
            return 7
        elif 140 < value <= 160:
            return 8
        elif 160 < value <= 180:
            return 9
        elif 180 < value <= 200:
            return 10
        elif 200 < value <= 600:
            return 11
        elif 600 < value <= 1000:
            return 12
        elif 1000 < value <= 1500:
            return 13
        elif 1500 < value:
            return 14
       
    def exlib_proc(self, value):
        if value == 1:
            return 0
        else:
            return 1
        
    def packer_proc(self, value):
        if value == 1:
            return 0
        else:
            return 1
        
    def funcnum_proc(self, value):
        if 0 < value <= 50:
            return 1
        elif 50 < value <= 250:
            return 2
        elif 250 < value:
            return 3
    
    def network_proc(self, value):
        if value == 1:
            return 0
        else:
            return 1
        
    
    def fea_gen(self):
        
        def df_process(result_df):
            
            result_df['arch_identifies'] = result_df['arch_identifies'].map(self.ISA_proc)
            result_df['size'] = result_df['size'].map(self.size_proc)
            train_result_df['dependencies'] = result_df['dependencies'].map(self.exlib_proc)
            result_df['packer'] = result_df['packer'].map(self.packer_proc)
            result_df['function_num'] = result_df['function_num'].map(self.funcnum_proc)
            result_df['network_state'] = result_df['network_state'].map(self.network_proc)
            result_df['Logic'] = result_df['Logic']/result_df['totalOpcode']
            result_df['ContStatus'] = result_df['ContStatus']/result_df['totalOpcode']
            result_df['Memory'] = result_df['Memory']/result_df['totalOpcode']
            result_df['Stack'] = result_df['Stack']/result_df['totalOpcode']
            result_df['Procedure'] = result_df['Procedure']/result_df['totalOpcode']
            result_df['Prefixed'] = result_df['Prefixed']/result_df['totalOpcode']
            result_df['SystemIO'] = result_df['SystemIO']/result_df['totalOpcode']
            result_df['Arithmetic'] = result_df['Arithmetic']/result_df['totalOpcode']
            result_df['System'] = result_df['System']/result_df['totalOpcode']
            result_df['Branch'] = result_df['Branch']/result_df['totalOpcode']
            result_df['ExecutTime'] = result_df['ExecutTime']/result_df['totalOpcode']
            result_df['Others'] = result_df['Others']/result_df['totalOpcode']
            
            return result_df               
            
        with pd.ExcelWriter(self.fea_csv, engine='xlsxwriter') as writer:
            
            for train_year in self.year_list:
                
                for train_arch in self.arch_list:
                
                    train_result_df = self.who_fea[self.who_fea['location'].isin(self.train_mal_labels[(self.train_mal_labels['year']==train_year)&(self.train_mal_labels['arch']==train_arch)]['location'])]                         
                    train_result_df = train_result_df.rename(columns={'arch': 'arch_identifies'})                  
                    train_result_df = train_result_df.merge(self.train_mal_labels[['location', 'year', 'arch']], on='location', how='left')
                    train_result_df.reset_index(drop=True, inplace=True)                    
                    train_result_df = df_process(train_result_df)                                  
                    train_result_df.to_excel(writer, sheet_name=str(train_year)+train_arch, index=False)
                                          
                    
                    test_result_df = self.who_fea[self.who_fea['location'].isin(self.test_mal_labels['location'])]   
                    test_result_df = test_result_df.rename(columns={'arch': 'arch_identifies'})                  
                    test_result_df = test_result_df.merge(self.test_mal_labels[['location', 'year', 'arch']], on='location', how='left')
                                           
                    test_result_df.reset_index(drop=True, inplace=True)
                    test_result_df = df_process(test_result_df)      
                    test_result_df.to_excel(writer, sheet_name=str(train_year)+train_arch+"_test", index=False)
                    

def REGREGgenerate_fea():
    
    train_dtanno_dir = '...' # annotation files postion 
    who_fea_file = "../../Features/ELFOpocode.csv" 
    
    for dtanno_file in ['dt'+str(i)+".xlsx" for i in range(0,5)]:
        train_dtanno_path = train_dtanno_dir + dtanno_file    
        mal_dataset = REGREGDtELFOpcode(annotations_file=train_dtanno_path, who_fea_file=who_fea_file, dtanno_file=dtanno_file)
        mal_dataset.fea_gen()
    
    
if __name__ == "__main__":
    REGREGgenerate_fea()            

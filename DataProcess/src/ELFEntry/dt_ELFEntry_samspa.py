# -*- coding: utf-8 -*-
import pandas as pd
import os
import csv
import json
import ast
from collections import Counter
import math

# from ELFEntry import ELFEntry

# class DtELFEntry(object):
#     def __init__(self, annotations_file, mal_dir):
#         self.mal_dir = mal_dir
#         self.annotations_file = annotations_file
        
#         self.mal_labels = pd.read_csv(self.annotations_file)
        
#         self.fea_csv = "Features/ELFEntry.csv"
#         if os.path.exists(self.fea_csv):
#             # Remove the file
#             os.remove(self.fea_csv)
            
#         self.L, self.N = 1024, 4
        
#         headers = ['location', 'label', 'data']
        
#         with open(self.fea_csv, mode='w', newline='') as file:
#             writer = csv.writer(file)      
#             writer.writerow(headers)     
        
#     def  __len__(self):
#         return len(self.mal_labels)
    
#     def fea_gen(self):
        
#         for index, row in self.mal_labels.iterrows():
#             mal_path = self.mal_dir + row[0]   
#             elfentry = ELFEntry(mal_path, self.L, self.N)
            
#             fea = elfentry.get_features()
            
#             if fea != None:            
#                 whole_fea = [row[0], row[1]] + [fea]
#                 with open(self.fea_csv, mode='a', newline='') as file:
#                     writer = csv.writer(file)
#                     writer.writerow(whole_fea)
                    
# def generate_fea():
    
#     mal_dir = '...'  # dataset position
#     annotations_file = '...' # dataset annotation position
#     mal_dataset = DtELFEntry(annotations_file=annotations_file, mal_dir=mal_dir)
#     mal_dataset.fea_gen()      
    
    
class REGDtELFEntry(object):
    
    def __init__(self, annotations_file, who_fea_file, dtanno_file):
        self.who_fea_file = who_fea_file
        self.annotations_file = annotations_file
        self.dtanno_file = dtanno_file
        
        self.year_list = [2020, 2021, 2022]
        self.arch_list = ['ARM', 'MIPS']
        
        self.who_fea = pd.read_csv(self.who_fea_file)
        self.train_mal_labels = pd.read_excel(self.annotations_file, sheet_name="whole")   
        self.test_mal_labels = pd.read_excel(self.annotations_file, sheet_name="test_samspa")   
        
        self.L, self.N = 1024, 4
               
        self.fea_csv = "../../Features/ELFEntry/SamSpa/ELFEntry"+self.dtanno_file[0:-5]+".xlsx"
        self.fea_filter = "../../Features/ELFEntry/SamSpa/FeaELFEntry"+self.dtanno_file[0:-5]+".txt"
        
        self.fea_filter_infor = {train_year:{arch: [] for arch in self.arch_list} for train_year in self.year_list}
        
    def  __len__(self):
        return len(self.train_mal_labels)
    
    def get_dt_fea(self):
        
        for train_year in self.year_list:            
            for train_arch in self.arch_list:                
                train_result_df = self.who_fea[self.who_fea['location'].isin(self.train_mal_labels[(self.train_mal_labels['year']==train_year)&(self.train_mal_labels['arch']==train_arch)]['location'])]                         
                train_result_df = train_result_df.merge(self.train_mal_labels[['location', 'year', 'arch']], on='location', how='left')
                train_result_df.reset_index(drop=True, inplace=True)                    
                
                volcabulary = []
                
                for index, row in train_result_df.iterrows():
                    ngrams = [row['data'][i:i+self.N] for i in range(len(row['data']) - self.N + 1)]
                    volcabulary.extend(ngrams)
                    print(len(volcabulary))
                    volcabulary = list(set(volcabulary))
                    print(len(volcabulary))
                    print('-------------------')
                      
                self.fea_filter_infor[train_year][train_arch] = volcabulary
                
        with open(self.fea_filter, 'w') as file:
            json.dump(self.fea_filter_infor, file, indent=4) 
            
    def fea_gen(self):
        
        with open(self.fea_filter, 'r') as file:
            self.fea_filter_infor = json.load(file)
            self.fea_filter_infor = {ast.literal_eval(k): v for k, v in self.fea_filter_infor.items()}
            
        def final_feature(data, fea_names):
            
            ngrams = [data[i:i+self.N] for i in range(len(data) - self.N + 1)]          
            ngram_freq = Counter(ngrams)            
            ngram_vector = dict(ngram_freq)
            
            fea_count = []
            for fea in fea_names:
                if fea in ngram_vector:
                    fea_count.append(ngram_vector[fea])
                else:
                    fea_count.append(0)
            
            chunk_size =  math.ceil(len(fea_names)/100) 
            chunks = [fea_count[i:i + chunk_size] for i in range(0, len(fea_count), chunk_size)]
                          
            return chunks
        
        with pd.ExcelWriter(self.fea_csv, engine='xlsxwriter') as writer:
        
            for train_year in self.year_list:  
                
                for train_arch in self.arch_list: 
                    
                    fea_names = self.fea_filter_infor[train_year][train_arch]         
                    chunk_names = ["chunk"+str(i) for i in range(100)]
                    
                    train_result_df = self.who_fea[self.who_fea['location'].isin(self.train_mal_labels[(self.train_mal_labels['year']==train_year)&(self.train_mal_labels['arch']==train_arch)]['location'])]                         
                    train_result_df = train_result_df.merge(self.train_mal_labels[['location', 'year', 'arch']], on='location', how='left')
                    train_result_df.reset_index(drop=True, inplace=True)   
                    new_columns = train_result_df['data'].apply(lambda x: final_feature(x, fea_names))   
                    df_new_cols = pd.DataFrame(new_columns.tolist(), columns=chunk_names)
                    train_result_df = pd.concat([train_result_df, df_new_cols], axis=1)
                    train_result_df = train_result_df.drop(columns=['data'])            
                    train_result_df.to_excel(writer, sheet_name=str(train_year)+train_arch, index=False)
                
                    test_result_df_1 = self.who_fea[self.who_fea['location'].isin(self.train_mal_labels[~((self.train_mal_labels['year']==train_year)&(self.train_mal_labels['arch']==train_arch))]['location'])]   
                    test_result_df_1 = test_result_df_1.merge(self.train_mal_labels[['location', 'year', 'arch']], on='location', how='left')
                    
                    test_result_df_2 = self.who_fea[self.who_fea['location'].isin(self.test_mal_labels[(self.test_mal_labels['year']==train_year)&(self.test_mal_labels['arch']==train_arch)]['location'])]                                     
                    test_result_df_2 = test_result_df_2.merge(self.test_mal_labels[['location', 'year', 'arch']], on='location', how='left')
                    
                    test_result_df = pd.concat([test_result_df_1, test_result_df_2], axis=0, ignore_index=True)
                    test_result_df.reset_index(drop=True, inplace=True)
                    new_columns = test_result_df['data'].apply(lambda x: final_feature(x, fea_names))   
                    df_new_cols = pd.DataFrame(new_columns.tolist(), columns=chunk_names)
                    test_result_df = pd.concat([test_result_df, df_new_cols], axis=1)
                    test_result_df = test_result_df.drop(columns=['data'])                            
                    test_result_df.to_excel(writer, sheet_name=str(train_year)+train_arch+"_test", index=False)
                    

    
def REGgenerate_fea():
    
    train_dtanno_dir = '...' # annotation files postion
    who_fea_file = "../../Features/ELFEntry.csv" 
    
    for dtanno_file in ['dt'+str(i)+".xlsx" for i in range(0,5)]:
        train_dtanno_path = train_dtanno_dir + dtanno_file    
        mal_dataset = REGDtELFEntry(annotations_file=train_dtanno_path, who_fea_file=who_fea_file, dtanno_file=dtanno_file)
        mal_dataset.fea_gen()

if __name__ == "__main__":
    # generate_fea()   
    REGgenerate_fea()

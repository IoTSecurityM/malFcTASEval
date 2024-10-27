# -*- coding: utf-8 -*-
import os
import pandas as pd
import subprocess
import numpy as np
import re
import json
import csv
import math
import ast

from scipy.stats import entropy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

from StrRFEDFrank import StrRFEDFrank


class DtStrRFEDFrank(object):
    def __init__(self, annotations_file, mal_dir, dtanno_file):
        self.mal_dir = mal_dir
        self.annotations_file = annotations_file
        self.dtanno_file = dtanno_file
                   
        self.mal_labels = pd.read_excel(self.annotations_file, sheet_name="whole") 
        self.test_mal_labels = pd.read_excel(self.annotations_file, sheet_name="test_diffspa") 
        
        self.year_list = [2020, 2021, 2022]
        self.arch_list = ['68K', 'ARM', 'MIPS', 'SH']
        
        self.fea_filter = "../../Features/StrRFEDFrank/DiffSpa/StrRFEDFrank"+self.dtanno_file[0:-5]+".xlsx"
        self.dt_str_infor = "../../Features/StrRFEDFrank/DiffSpa/stringInforStrRFEDFrank"+self.dtanno_file[0:-5]+".txt"
        self.top50_str_perfam = "../../Features/StrRFEDFrank/DiffSpa/to50strperfam"+self.dtanno_file[0:-5]+".txt" 
        
        self.str_maxlen_infor = {train_year:{arch: {"max_len": 0, "str_list":[]} for arch in self.arch_list} for train_year in self.year_list}
        self.top50_str_infor = {}
        
    def  __len__(self):
        return len(self.mal_labels)
    
    def get_dt_fea(self):
        
        for train_year in self.year_list:
            
            for arch in self.arch_list:
                
                print(f"{train_year} - {arch}")
                
                train_df = self.mal_labels[(self.mal_labels['year'] == train_year) & (self.mal_labels['arch'] == arch)]
                
                def extract_strings(file_path):
                    result = subprocess.run(["strings", file_path], text=True, capture_output=True)
                    return result.stdout.splitlines()    
                
                documents = []     
                max_len = 0
                for index, row in train_df.iterrows():
                    mal_path = self.mal_dir + row[0]  
                    strings_output = extract_strings(mal_path)
                    
                    if len(strings_output) > 0: 
                        max_len = max(map(len, strings_output)) if max(map(len, strings_output)) > max_len else max_len
                                   
                    documents.append(" ".join(strings_output))
                
                self.str_maxlen_infor[train_year][arch]["max_len"] = max_len
                
                vectorizer = CountVectorizer()
                X = vectorizer.fit_transform(documents)
                df = np.sum(X > 0, axis=0).A1  # Document frequency of each term
                strings = np.array(vectorizer.get_feature_names_out())  # Extracted feature names
                filtered_strings = strings[df >= 10]  # Filter terms appearing in at least 10 documents
                
                def has_special_chars(s):
                    return bool(re.search(r'[^a-zA-Z0-9]', s))
                
                filtered_strings = [s for s in filtered_strings if not has_special_chars(s)]
                
                vectorizer = CountVectorizer(vocabulary=filtered_strings)
                X_filtered = vectorizer.fit_transform(documents)             
                
                # Calculate co-occurrence matrix
                co_occurrence_matrix = (X_filtered.T @ X_filtered).toarray()
                
                # Calculate correlation matrix
                correlation_matrix = np.corrcoef(co_occurrence_matrix)
                
                # Find pairs of highly correlated strings
                threshold = 0.8  # Define your correlation threshold
                to_remove = set()
                for i in range(len(filtered_strings)):
                    for j in range(i + 1, len(filtered_strings)):
                        if correlation_matrix[i, j] > threshold:
                            to_remove.add(filtered_strings[j])
                
                final_strings = [s for s in filtered_strings if s not in to_remove]        
                
                self.str_maxlen_infor[train_year][arch]["str_list"] = final_strings
        
        self.str_maxlen_infor = {str(k): v for k, v in self.str_maxlen_infor.items()}
        
        with open(self.dt_str_infor, 'w') as file:
            json.dump(self.str_maxlen_infor, file, indent=4) 
            
            
    def DFrank(self):
        
        with open(self.dt_str_infor, 'r') as file:
            self.str_maxlen_infor = json.load(file)
            self.str_maxlen_infor = {ast.literal_eval(k): v for k, v in self.str_maxlen_infor.items()}
        
        for train_year, infor in self.str_maxlen_infor.items():
            
            for arch in ['ARM', 'MIPS']:
                            
                train_df = self.mal_labels[(self.mal_labels['year'] == train_year) & (self.mal_labels['arch'] == arch)]
                
                strings = infor[arch]['str_list']
                str_arch_family = {'arch': {strs: [] for strs in strings}, 'family': {strs: [] for strs in strings}}
                
                def extract_strings(file_path):
                    result = subprocess.run(["strings", file_path], text=True, capture_output=True)
                    return result.stdout.splitlines()      
                
                arch_docs, family_docs = {}, {}
                for index, row in train_df.iterrows():
                    arch, family = row[0].split('/')[0], row[1]   
                    arch_docs.setdefault(arch, [])
                    family_docs.setdefault(family, [])
                    
                    mal_path = self.mal_dir + row[0]  
                    strings_output = extract_strings(mal_path)                           
                    arch_docs[arch].append(" ".join(strings_output))
                    family_docs[family].append(" ".join(strings_output))
                
                print(arch_docs.keys())
                for arch, documents in arch_docs.items():            
                    vectorizer = CountVectorizer(vocabulary=strings)
                    X_filtered = vectorizer.fit_transform(documents)    
                    df = np.sum(X_filtered > 0, axis=0).A1 
                    
                    for strs, df_value in zip(strings, df):
                        str_arch_family['arch'][strs].append(df_value)
                
                print(family_docs.keys())
                for family, documents in family_docs.items():
                    vectorizer = CountVectorizer(vocabulary=strings)
                    X_filtered = vectorizer.fit_transform(documents)    
                    df = np.sum(X_filtered > 0, axis=0).A1 
                    
                    for strs, df_value in zip(strings, df):
                        str_arch_family['family'][strs].append(df_value)
                
                strs_entropy ={}
                for strs, arch_df in str_arch_family['arch'].items():                     
                    value, counts = np.unique(arch_df, return_counts=True)
                    probability_distribution = counts / len(arch_df)
                    entropy_value = entropy(probability_distribution)
                    strs_entropy[strs] = entropy_value
    
                DFRank = {}
                for strs, family_df in str_arch_family['family'].items():        
                    DFRank[strs] = {family: ratio for family, ratio in zip(list(family_docs.keys()), list(1*np.array(family_df)/sum(family_df)))}
                                 
                DFRank_df = pd.DataFrame.from_dict(DFRank, orient='index')
                
                top_50_strings_per_family = DFRank_df.apply(lambda x: x.nlargest(50).index)
                
                self.top50_str_infor.setdefault(train_year, {})
                self.top50_str_infor[train_year][arch] = top_50_strings_per_family.to_dict()
       
        self.top50_str_infor = {str(k): v for k, v in self.top50_str_infor.items()}
        with open(self.top50_str_perfam, 'w') as f:
           json.dump(self.top50_str_infor, f, indent=4)            
    
    
    def fea_gen(self):
        
        with open(self.top50_str_perfam, 'r') as file:
            self.top50_str_infor = json.load(file)
            self.top50_str_infor = {ast.literal_eval(k): v for k, v in self.top50_str_infor.items()}
        
        with open(self.dt_str_infor, 'r') as file:
            self.str_maxlen_infor = json.load(file)
            self.str_maxlen_infor = {ast.literal_eval(k): v for k, v in self.str_maxlen_infor.items()}
            
        with pd.ExcelWriter(self.fea_filter, engine='xlsxwriter') as writer:
            
            for train_year, infor in self.top50_str_infor.items():
                
                print(f"{train_year}:")
                for arch, arch_infor in infor.items():
                    
                    print(f"{arch}:")
                    str_list = list(set(pd.DataFrame(arch_infor).values.flatten().tolist()))
                    max_len = self.str_maxlen_infor[train_year][arch]['max_len'] 
                    
                    print(f"{len(str_list)} and {max_len}")
                   
                    headers_df = pd.DataFrame(columns = ['location', 'label', 'year', 'arch']+['bin'+str(i) for i in range(1, 51)] + str_list)
                    strrfedfrank = StrRFEDFrank(str_list, max_len)  
                                       
                    headers_df.head(0).to_excel(writer, sheet_name=str(train_year)+str(arch), index=False)                    
                    current_row = 1         
                    train_mal_labels = self.mal_labels[(self.mal_labels['year']==train_year) & (self.mal_labels['arch']==arch)]  
                    for index, row in train_mal_labels.iterrows():
                        mal_path = self.mal_dir + row[0]              
                        fea = strrfedfrank.get_features(mal_path)                        
                        whole_fea = [row[0], row[1], row[2], row[4]] + fea 
                        pd.DataFrame([whole_fea]).to_excel(writer, sheet_name=str(train_year)+str(arch), index=False, header=False, startrow=current_row)                         
                        current_row += 1       
                        
                    headers_df.head(0).to_excel(writer, sheet_name=str(train_year)+str(arch)+"_test", index=False)                
                    current_row = 1                                                         
                    for index, row in self.test_mal_labels.iterrows():
                        mal_path = self.mal_dir + row[0]              
                        fea = strrfedfrank.get_features(mal_path)                   
                        whole_fea = [row[0], row[1], row[2], row[4]] + fea 
                        pd.DataFrame([whole_fea]).to_excel(writer, sheet_name=str(train_year)+str(arch)+"_test", index=False, header=False, startrow=current_row)                     
                        current_row += 1    

            
def generate_fea():
    
    train_dtanno_dir = '...' # annotation position    
    mal_dir = '...' # dataset position 
    
    for dtanno_file in ['dt'+str(i)+".xlsx" for i in range(0,5)]:
        train_dtanno_path = train_dtanno_dir + dtanno_file
        mal_dataset = DtStrRFEDFrank(annotations_file=train_dtanno_path, mal_dir=mal_dir, dtanno_file=dtanno_file)
        mal_dataset.fea_gen()
    
    
if __name__ == "__main__":
    generate_fea() 
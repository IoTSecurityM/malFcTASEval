# -*- coding: utf-8 -*-
import pandas as pd
import csv
import os

from ImgHaralick import ImgHaralick

class DtImgHaralick(object):
    def __init__(self, annotations_file, mal_dir, dtanno_file):
        self.mal_dir = mal_dir
        self.annotations_file = annotations_file
        self.dtanno_file = dtanno_file
        
        self.year_list = [2020, 2021, 2022]
        self.arch_list = ['ARM', 'MIPS']
        
        self.mal_labels = pd.read_excel(self.annotations_file, sheet_name="whole") 
        self.test_mal_labels = pd.read_excel(self.annotations_file, sheet_name="test_diffspa") 
        
        self.fea_csv = "../../Features/ImgHaralick/DiffSpa/ImgHaralick"+self.dtanno_file[0:-5]+".xlsx"
              
    def  __len__(self):
        return len(self.mal_labels)
    
    def fea_gen(self):
        
        if os.path.exists(self.fea_csv):
            # Remove the file
            os.remove(self.fea_csv)
        
        headers = ['location', 'label', 'year', 'arch', 'ASM_0', 'ASM_45', 'ASM_90', 'ASM_135','contra_0','contra_45','contra_90', 'contra_135', 'homoge_0',\
                   'homoge_45', 'homoge_90', 'homoge_135', 'correl_0', 'correl_45', 'correl_90', 'correl_135', 'entro_0', 'entro_45', 'entro_90', 'entro_135']
       
        with pd.ExcelWriter(self.fea_csv, engine='xlsxwriter') as writer:
            
            for train_year in self.year_list:
                
                print(f"{train_year}:")
                for arch in self.arch_list :                   
                   
                    headers_df = pd.DataFrame(columns = headers)
                    imgharalick = ImgHaralick()
                                       
                    headers_df.head(0).to_excel(writer, sheet_name=str(train_year)+str(arch), index=False)                    
                    current_row = 1         
                    train_mal_labels = self.mal_labels[(self.mal_labels['year']==train_year) & (self.mal_labels['arch']==arch)]  
                    for index, row in train_mal_labels.iterrows():
                        mal_path = self.mal_dir + row[0]              
                        fea = imgharalick.get_features(mal_path)  
                        whole_fea = [row[0], row[1], row[2], row[4]] + fea  
                        pd.DataFrame([whole_fea]).to_excel(writer, sheet_name=str(train_year)+str(arch), index=False, header=False, startrow=current_row)                         
                        current_row += 1       
                        
                    headers_df.head(0).to_excel(writer, sheet_name=str(train_year)+str(arch)+"_test", index=False)                
                    current_row = 1                  
                    for index, row in self.test_mal_labels.iterrows():
                        mal_path = self.mal_dir + row[0]              
                        fea = imgharalick.get_features(mal_path)  
                        whole_fea = [row[0], row[1], row[2], row[4]] + fea  
                        pd.DataFrame([whole_fea]).to_excel(writer, sheet_name=str(train_year)+str(arch)+"_test", index=False, header=False, startrow=current_row)                     
                        current_row += 1  
                                                     
            
def generate_fea():
    
    train_dtanno_dir = '...' # annotation position    
    mal_dir = '...' # dataset position 
    
    for dtanno_file in ['dt'+str(i)+".xlsx" for i in range(0,5)]:
        train_dtanno_path = train_dtanno_dir + dtanno_file    
        mal_dataset = DtImgHaralick(annotations_file=train_dtanno_path, mal_dir=mal_dir, dtanno_file=dtanno_file)
        mal_dataset.fea_gen()


if __name__ == "__main__":
    generate_fea()            

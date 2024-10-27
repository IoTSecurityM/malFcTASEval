# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

from figure_show_util import add_avg_std, mark_func, cell_feature_dict, plot_draw, classifier_algorithm_based

def dt_best_classifier():
    file_shown_lists = ["[25]", "[34]", "[35]", "[37]", "[39]", "[40]"]
    file_name_lists = ["ELFMiner", "ELFEntry", "ImgHaralick", "StrRFEDFrank", "ELFOpocode", "FileEntry"]
    
    ELFMiner_results_dir = "../../Results/DiffSpa/ELFMiner/"
    ELFEntry_results_dir = "../../Results/DiffSpa/ELFEntry/"
    ImgHaralick_results_dir = "../../Results/DiffSpa/ImgHaralick/"
    StrRFEDFrank_results_dir = "../../Results/DiffSpa/StrRFEDFrank/"
    ELFOpocode_results_dir = "../../Results/DiffSpa/ELFOpocode/"
    FileEntry_results_dir = "../../Results/DiffSpa/FileEntry/"
    
    # Training results
    results_dir_list = [ELFMiner_results_dir, ELFEntry_results_dir, ImgHaralick_results_dir, StrRFEDFrank_results_dir, ELFOpocode_results_dir, FileEntry_results_dir]
    
    who_results_df = pd.DataFrame(columns=["train_alg", "metric", "model", "train_arch", "train_year", "Performance"])
    
    for results_dir, file_name, file_show in zip(results_dir_list, file_name_lists, file_shown_lists): 
        for file_num in range(0,5):
            result_file_path = results_dir + file_name + "dt" + str(file_num) + '_train.txt'
            with open(result_file_path, 'r') as file:
                content = file.read()
                tcsaic_results = json.loads(content)
            for metric, metric_infor in tcsaic_results.items():               
                for model, model_infor in metric_infor.items():
                    for train_infor, train_perform in model_infor.items():                        
                        who_results_df = who_results_df.append({"train_alg": file_show, "dt_idx":file_num, "metric": metric, "model": model, "train_arch": train_infor[4:], "train_year": train_infor[0:4], "Performance": train_perform}, ignore_index=True)   
           
    who_results_df = who_results_df[who_results_df["metric"] == 'f1']
    
    train_year_order = ['2020', '2021', '2022']
    train_alg_order = file_shown_lists
    train_arch_order = ['ARM', 'MIPS']        
    train_model_order = ['RF', 'KNN', 'SVC', 'MLP']
    
    who_results_df['train_year'] = pd.Categorical(who_results_df['train_year'], categories=train_year_order, ordered=True)
    who_results_df['train_arch'] = pd.Categorical(who_results_df['train_arch'], categories=train_arch_order, ordered=True)   
    who_results_df['train_alg'] = pd.Categorical(who_results_df['train_alg'], categories=train_alg_order, ordered=True)            
    who_results_df['model'] = pd.Categorical(who_results_df['model'], categories=train_model_order, ordered=True)  
     
    who_results_df = who_results_df.sort_values(by=['train_year', 'train_arch', 'model', 'dt_idx', 'train_alg'])            
    who_results_df = who_results_df.reset_index(drop=True)
    
    who_results_df = who_results_df.loc[who_results_df.groupby(['train_year', 'train_arch', 'model', 'dt_idx'])['Performance'].idxmax()]
    
    who_results_df.to_csv("../../Results/train_dt_bset_classifier1.csv", index=False)
    
    
    

if __name__ == "__main__":
    dt_best_classifier()
    

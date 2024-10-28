# -*- coding: utf-8 -*-
import pandas as pd
import json

def results_csv():
    
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
                        who_results_df = who_results_df.append({"train_alg": file_show, "metric": metric, "model": model, "train_arch": train_infor[4:], "train_year": train_infor[0:4], "Performance": train_perform}, ignore_index=True)   
        
    who_results_df = who_results_df[["metric", "train_year", "train_alg", "train_arch", "model", "Performance"]]
    who_results_df.to_csv("../../Results/train_result.csv", index=False)
    
    ELFMiner_results_dir = "../../Results/SamSpa/ELFMiner/"
    ELFEntry_results_dir = "../../Results/SamSpa/ELFEntry/"
    ImgHaralick_results_dir = "../../Results/SamSpa/ImgHaralick/"
    StrRFEDFrank_results_dir = "../../Results/SamSpa/StrRFEDFrank/"
    ELFOpocode_results_dir = "../../Results/SamSpa/ELFOpocode/"
    FileEntry_results_dir = "../../Results/SamSpa/FileEntry/"
    
    results_dir_list = [ELFMiner_results_dir, ELFEntry_results_dir, ImgHaralick_results_dir, StrRFEDFrank_results_dir, ELFOpocode_results_dir, FileEntry_results_dir] 
    test_who_results_df = pd.DataFrame(columns=["train_alg", "metric", "model", "train_arch", "train_year", "Test Year", "test_arch", "Performance"])
    
    for results_dir, file_name, file_show in zip(results_dir_list, file_name_lists, file_shown_lists): 
        for file_num in range(0,5):
            result_file_path = results_dir + file_name + "dt" + str(file_num) + '.txt'
            with open(result_file_path, 'r') as file:
                content = file.read()
                tcsaic_results = json.loads(content)
            for metric, metric_infor in tcsaic_results.items():               
                for model, model_infor in metric_infor.items():
                    for train_infor, train_infor_perform in model_infor.items():
                        for test_year, test_infor in train_infor_perform.items():
                            for test_arch, test_perform in test_infor.items():
                                test_who_results_df = test_who_results_df.append({"train_alg": file_show, "metric": metric, "model": model, "train_arch": train_infor[4:], "train_year": train_infor[0:4], "Test Year": test_year, "test_arch": test_arch, "Performance": test_perform}, ignore_index=True)   
        
    test_who_results_df = test_who_results_df[["metric", "model", "train_year", "train_alg", "train_arch", "Test Year", "test_arch", "Performance"]]
    
    test_who_results_df.to_csv("../../Results/samespa_result.csv", index=False)
    
    ELFMiner_results_dir = "../../Results/DiffSpa/ELFMiner/"
    ELFEntry_results_dir = "../../Results/DiffSpa/ELFEntry/"
    ImgHaralick_results_dir = "../../Results/DiffSpa/ImgHaralick/"
    StrRFEDFrank_results_dir = "../../Results/DiffSpa/StrRFEDFrank/"
    ELFOpocode_results_dir = "../../Results/DiffSpa/ELFOpocode/"
    FileEntry_results_dir = "../../Results/DiffSpa/FileEntry/"
    
    results_dir_list = [ELFMiner_results_dir, ELFEntry_results_dir, ImgHaralick_results_dir, StrRFEDFrank_results_dir, ELFOpocode_results_dir, FileEntry_results_dir]    
    test_who_results_df = pd.DataFrame(columns=["train_alg", "metric", "model", "train_arch", "train_year", "Test Year", "test_arch", "Performance"])
    
    for results_dir, file_name, file_show in zip(results_dir_list, file_name_lists, file_shown_lists): 
        for file_num in range(0,5):
            result_file_path = results_dir + file_name + "dt" + str(file_num) + '.txt'
            with open(result_file_path, 'r') as file:
                content = file.read()
                tcsaic_results = json.loads(content)
            for metric, metric_infor in tcsaic_results.items():               
                for model, model_infor in metric_infor.items():
                    for train_infor, train_infor_perform in model_infor.items():
                        for test_year, test_infor in train_infor_perform.items():
                            for test_arch, test_perform in test_infor.items():
                                test_who_results_df = test_who_results_df.append({"train_alg": file_show, "metric": metric, "model": model, "train_arch": train_infor[4:], "train_year": train_infor[0:4], "Test Year": test_year, "test_arch": test_arch, "Performance": test_perform}, ignore_index=True)   
        
    test_who_results_df = test_who_results_df[["metric", "model", "train_year", "train_alg", "train_arch", "Test Year", "test_arch", "Performance"]]
    
    test_who_results_df.to_csv("../../Results/diffspa_result.csv", index=False)


def spa_results_csv():
    
    file_shown_lists = ["[25]", "[34]", "[35]", "[37]", "[39]", "[40]"]
    file_name_lists = ["ELFMiner", "ELFEntry", "ImgHaralick", "StrRFEDFrank", "ELFOpocode", "FileEntry"]
    
    ELFMiner_results_dir = "../../Results/DiffSpa/SpaResults/ELFMiner/"
    ELFEntry_results_dir = "../../Results/DiffSpa/SpaResults/ELFEntry/"
    ImgHaralick_results_dir = "../../Results/DiffSpa/SpaResults/ImgHaralick/"
    StrRFEDFrank_results_dir = "../../Results/DiffSpa/SpaResults/StrRFEDFrank/"
    ELFOpocode_results_dir = "../../Results/DiffSpa/SpaResults/ELFOpocode/"
    FileEntry_results_dir = "../../Results/DiffSpa/SpaResults/FileEntry/"
    
    # Training results
    results_dir_list = [ELFMiner_results_dir, ELFEntry_results_dir, ImgHaralick_results_dir, StrRFEDFrank_results_dir, ELFOpocode_results_dir, FileEntry_results_dir]
    
    who_results_df = pd.DataFrame(columns=["train_alg", "model", "train_arch", "train_year", "0_precision", "0_recall", "0_f1_score", "1_precision", "1_recall", "1_f1_score","2_precision", "2_recall", "2_f1_score"])
    
    for results_dir, file_name, file_show in zip(results_dir_list, file_name_lists, file_shown_lists): 
        for file_num in range(0,5):
            result_file_path = results_dir + file_name + "dt" + str(file_num) + '_train.txt'
            with open(result_file_path, 'r') as file:
                content = file.read()
                tcsaic_results = json.loads(content)                         
            for model, model_infor in tcsaic_results.items():
                for train_infor, train_perform in model_infor.items():                     
                    who_results_df = who_results_df.append({"train_alg": file_show, "model": model, "train_arch": train_infor[4:], "train_year": train_infor[0:4], "0_precision": train_perform['0']['precision'], "0_recall": train_perform['0']['recall'], "0_f1_score": train_perform['0']['f1_score'], "1_precision": train_perform['1']['precision'], "1_recall": train_perform['1']['recall'], "1_f1_score": train_perform['1']['f1_score'], "2_precision": train_perform['2']['precision'], "2_recall": train_perform['2']['recall'], "2_f1_score": train_perform['2']['f1_score']}, ignore_index=True)   
    
    who_results_df = who_results_df[["train_year", "train_alg", "train_arch", "model", "0_precision", "0_recall", "0_f1_score", "1_precision", "1_recall", "1_f1_score","2_precision", "2_recall", "2_f1_score"]]
    who_results_df.to_csv("../../Results/spatest_train_result.csv", index=False)
    
    ELFMiner_results_dir = "../../Results/SamSpa/SpaResults/ELFMiner/"
    ELFEntry_results_dir = "../../Results/SamSpa/SpaResults/ELFEntry/"
    ImgHaralick_results_dir = "../../Results/SamSpa/SpaResults/ImgHaralick/"
    StrRFEDFrank_results_dir = "../../Results/SamSpa/SpaResults/StrRFEDFrank/"
    ELFOpocode_results_dir = "../../Results/SamSpa/SpaResults/ELFOpocode/"
    FileEntry_results_dir = "../../Results/SamSpa/SpaResults/FileEntry/"
    
    results_dir_list = [ELFMiner_results_dir, ELFEntry_results_dir, ImgHaralick_results_dir, StrRFEDFrank_results_dir, ELFOpocode_results_dir, FileEntry_results_dir] 
    test_who_results_df = pd.DataFrame(columns=["train_alg", "model", "train_arch", "train_year", "Test Year", "test_arch", "0_precision", "0_recall", "0_f1_score", "1_precision", "1_recall", "1_f1_score","2_precision", "2_recall", "2_f1_score"])
    
    for results_dir, file_name, file_show in zip(results_dir_list, file_name_lists, file_shown_lists): 
        for file_num in range(0,5):
            result_file_path = results_dir + file_name + "dt" + str(file_num) + '.txt'
            with open(result_file_path, 'r') as file:
                content = file.read()
                tcsaic_results = json.loads(content)                  
            for model, model_infor in tcsaic_results.items():
                for train_infor, train_infor_perform in model_infor.items():
                    for test_year, test_infor in train_infor_perform.items():
                        for test_arch, test_perform in test_infor.items():                               
                            test_who_results_df = test_who_results_df.append({"train_alg": file_show, "model": model, "train_arch": train_infor[4:], "train_year": train_infor[0:4], "Test Year": test_year, "test_arch": test_arch, "0_precision": test_perform['0']['precision'], "0_recall": test_perform['0']['recall'], "0_f1_score": test_perform['0']['f1_score'], "1_precision": test_perform['1']['precision'], "1_recall": test_perform['1']['recall'], "1_f1_score": test_perform['1']['f1_score'], "2_precision": test_perform['2']['precision'], "2_recall": test_perform['2']['recall'], "2_f1_score": test_perform['2']['f1_score']}, ignore_index=True)   
    
    test_who_results_df = test_who_results_df[["model", "train_year", "train_alg", "train_arch", "Test Year", "test_arch", "0_precision", "0_recall", "0_f1_score", "1_precision", "1_recall", "1_f1_score","2_precision", "2_recall", "2_f1_score"]]
    
    test_who_results_df.to_csv("../../Results/spatest_samespa_result.csv", index=False)
    
    ELFMiner_results_dir = "../../Results/DiffSpa/SpaResults/ELFMiner/"
    ELFEntry_results_dir = "../../Results/DiffSpa/SpaResults/ELFEntry/"
    ImgHaralick_results_dir = "../../Results/DiffSpa/SpaResults/ImgHaralick/"
    StrRFEDFrank_results_dir = "../../Results/DiffSpa/SpaResults/StrRFEDFrank/"
    ELFOpocode_results_dir = "../../Results/DiffSpa/SpaResults/ELFOpocode/"
    FileEntry_results_dir = "../../Results/DiffSpa/SpaResults/FileEntry/"
    
    results_dir_list = [ELFMiner_results_dir, ELFEntry_results_dir, ImgHaralick_results_dir, StrRFEDFrank_results_dir, ELFOpocode_results_dir, FileEntry_results_dir]    
    test_who_results_df = pd.DataFrame(columns=["train_alg", "model", "train_arch", "train_year", "Test Year", "test_arch", "0_precision", "0_recall", "0_f1_score", "1_precision", "1_recall", "1_f1_score","2_precision", "2_recall", "2_f1_score"])
    
    for results_dir, file_name, file_show in zip(results_dir_list, file_name_lists, file_shown_lists): 
        for file_num in range(0,5):
            result_file_path = results_dir + file_name + "dt" + str(file_num) + '.txt'
            with open(result_file_path, 'r') as file:
                content = file.read()
                tcsaic_results = json.loads(content)                       
            for model, model_infor in tcsaic_results.items():
                for train_infor, train_infor_perform in model_infor.items():
                    for test_year, test_infor in train_infor_perform.items():
                        for test_arch, test_perform in test_infor.items():                                
                            test_who_results_df = test_who_results_df.append({"train_alg": file_show, "model": model, "train_arch": train_infor[4:], "train_year": train_infor[0:4], "Test Year": test_year, "test_arch": test_arch, "0_precision": test_perform['0']['precision'], "0_recall": test_perform['0']['recall'], "0_f1_score": test_perform['0']['f1_score'], "1_precision": test_perform['1']['precision'], "1_recall": test_perform['1']['recall'], "1_f1_score": test_perform['1']['f1_score'], "2_precision": test_perform['2']['precision'], "2_recall": test_perform['2']['recall'], "2_f1_score": test_perform['2']['f1_score']}, ignore_index=True)   
    
    test_who_results_df = test_who_results_df[["model", "train_year", "train_alg", "train_arch", "Test Year", "test_arch", "0_precision", "0_recall", "0_f1_score", "1_precision", "1_recall", "1_f1_score","2_precision", "2_recall", "2_f1_score"]]
    
    test_who_results_df.to_csv("../../Results/spatest_diffspa_result.csv", index=False)
    
    
    
def model_filter():
    train_who_results_df = pd.read_csv('../../Results/train_result.csv')
    train_who_results_df = train_who_results_df[train_who_results_df['metric'] == 'f1'].drop(columns=["metric"])
    train_who_results_df = train_who_results_df.groupby(["train_year", "train_alg", "train_arch", "model"], as_index=False).mean()
    train_who_results_df = train_who_results_df[train_who_results_df['Performance']>=0.9]    
    
    train_who_results_df = pd.read_csv('../../Results/samespa_result.csv')
    train_who_results_df = train_who_results_df[train_who_results_df['metric'] == 'f1'].drop(columns=["metric"])
    train_who_results_df = train_who_results_df[train_who_results_df['train_arch']==train_who_results_df['test_arch']].drop(columns=["test_arch"])
    train_who_results_df = train_who_results_df[train_who_results_df['train_year']==train_who_results_df['Test Year']].drop(columns=["Test Year"])
    train_who_results_df = train_who_results_df.groupby(["train_year", "train_alg", "train_arch", "model"], as_index=False).mean()
    train_who_results_df = train_who_results_df[train_who_results_df['Performance']>=0.9]    
    
    test_who_results_df = pd.read_csv("../../Results/samespa_result.csv")
    test_who_results_df = test_who_results_df[test_who_results_df['metric'] == 'f1'].drop(columns=["metric"]).reset_index(drop=True)
    
    filtered_test_results_df = test_who_results_df[test_who_results_df[["train_year", "train_alg", "train_arch", "model"]].apply(tuple, axis=1).isin(train_who_results_df[["train_year", "train_alg", "train_arch", "model"]].apply(tuple, axis=1))]
    filtered_test_results_df.to_csv("../../Results/goodmodel_samespa_result.csv", index=False)
    
    test_who_results_df = pd.read_csv("../../Results/diffspa_result.csv")
    test_who_results_df = test_who_results_df[test_who_results_df['metric'] == 'f1'].drop(columns=["metric"]).reset_index(drop=True)
    
    filtered_test_results_df = test_who_results_df[test_who_results_df[["train_year", "train_alg", "train_arch", "model"]].apply(tuple, axis=1).isin(train_who_results_df[["train_year", "train_alg", "train_arch", "model"]].apply(tuple, axis=1))]
    filtered_test_results_df.to_csv("../../Results/goodmodel_diffspa_result.csv", index=False)
    
    test_who_results_df = pd.read_csv("../../Results/spatest_samespa_result.csv")
   
    filtered_test_results_df = test_who_results_df[test_who_results_df[["train_year", "train_alg", "train_arch", "model"]].apply(tuple, axis=1).isin(train_who_results_df[["train_year", "train_alg", "train_arch", "model"]].apply(tuple, axis=1))]
    filtered_test_results_df.to_csv("../../Results/goodmodel_spatest_samespa_result.csv", index=False)
    
    test_who_results_df = pd.read_csv("../../Results/spatest_diffspa_result.csv")
   
    filtered_test_results_df = test_who_results_df[test_who_results_df[["train_year", "train_alg", "train_arch", "model"]].apply(tuple, axis=1).isin(train_who_results_df[["train_year", "train_alg", "train_arch", "model"]].apply(tuple, axis=1))]
    filtered_test_results_df.to_csv("../../Results/goodmodel_spatest_diffspa_result.csv", index=False)
    
    
if __name__ == "__main__":
    # spa_results_csv()
    # results_csv()
    model_filter()
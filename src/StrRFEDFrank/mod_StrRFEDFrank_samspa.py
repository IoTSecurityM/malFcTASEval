# -*- coding: utf-8 -*-
import pandas as pd
import ast
import numpy as np
import json
import itertools
import pickle

from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, f1_score

def optimal_results(dtanno_file):
    fea_filter = "../../Features/StrRFEDFrank/SamSpa/StrRFEDFrank"+dtanno_file[0:-5]+".xlsx"
    
    # Load the Excel file
    whofol_feas = pd.ExcelFile(fea_filter)
    
    # Get the sheet names    
    year_list = [2020, 2021, 2022]
    train_arch_list = ['ARM', 'MIPS']
    arch_list = ['68K', 'ARM', 'MIPS', 'SH']
    
    whofol_results = {"acc": {}, "f1":{}}
    train_results = {"acc": {}, "f1":{}}
    
    spa_whofol_results = {}
    spa_train_results = {}
    
    for train_year, train_arch in itertools.product(year_list, train_arch_list):
        
        sheetname = str(train_year)+train_arch
        
        train_fea_df = pd.read_excel(fea_filter, sheet_name=sheetname)        
        test_fea_df = pd.read_excel(fea_filter, sheet_name=sheetname+"_test")          
        
        train_df = train_fea_df[(train_fea_df['year'] == train_year) & (train_fea_df['arch'] == train_arch)]
            
        X_train, y_train = train_df.drop(columns=['location', 'label', 'year', 'arch']), train_df['label']
        
        model_name_list = ['RF', 'KNN', 'SVC', 'MLP']
        
        # Train and evaluate each model
        print(f"{sheetname}:")
        for name in model_name_list:
            # Train the model
            with open("../../Results/BestModels/StrRFEDFrank/StrRFEDFrank"+dtanno_file[0:-5]+"/"+sheetname+f'/best_{name}.pkl', 'rb') as f:
                best_model = pickle.load(f)
            
            y_train_pred = best_model.predict(X_train)
            
            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_f1 = f1_score(y_train, y_train_pred, average='macro')
            
            train_results["acc"].setdefault(name, {})
            train_results["f1"].setdefault(name, {})
            
            train_results["acc"][name][sheetname] = train_accuracy
            train_results["f1"][name][sheetname] = train_f1
            
            print(f"- Accuracy: {train_accuracy:.4f}, F1-Score: {train_f1:.4f}")  
            
            report = classification_report(y_train, y_train_pred, output_dict=True)
            class_metrics = {}

            for label, metrics in report.items():
                if label.isdigit(): 
                    class_metrics[label] = {
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1_score': metrics['f1-score']
                    }
            
            spa_train_results.setdefault(name, {})
            
            spa_train_results[name][sheetname] = class_metrics
            
            print(f"Train F1-Score: {class_metrics}")     
            
            whofol_results["acc"].setdefault(name, {})
            whofol_results["f1"].setdefault(name, {})
            
            whofol_results["acc"][name][sheetname] = {}
            whofol_results["f1"][name][sheetname] = {}
            
            spa_whofol_results.setdefault(name, {})
            
            spa_whofol_results[name][sheetname] = {} 
            
            print(f"- {name}:")
            for test_year in year_list:
                
                print(f"-- {test_year}") 
                whofol_results["acc"][name][sheetname].setdefault(test_year, {})
                whofol_results["f1"][name][sheetname].setdefault(test_year, {})
                
                spa_whofol_results[name][sheetname].setdefault(test_year, {})
                
                for test_arch in arch_list:
                
                    test_df = test_fea_df[(test_fea_df['year'] == test_year) & (test_fea_df['arch'] == test_arch)] 
                    X_test, y_test = test_df.drop(columns=['location', 'label', 'year', 'arch']), test_df['label']
                
                    # Make predictions on the test data
                    y_pred = best_model.predict(X_test)
                    
                    # Calculate accuracy and F1-score
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='macro')
                                       
                    whofol_results["acc"][name][sheetname][test_year][test_arch] = accuracy
                    whofol_results["f1"][name][sheetname][test_year][test_arch] = f1
                    
                    print(f"--- {test_arch} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")   
                    
                    report = classification_report(y_test, y_pred, output_dict=True)
                    class_metrics = {}

                    for label, metrics in report.items():
                        if label.isdigit():  # Only process actual class labels, ignore 'accuracy', 'macro avg', 'weighted avg'
                            class_metrics[label] = {
                                'precision': metrics['precision'],
                                'recall': metrics['recall'],
                                'f1_score': metrics['f1-score']
                            }                   
                                       
                    spa_whofol_results[name][sheetname][test_year][test_arch] = class_metrics
                    
                    print(f"--- {test_arch} - F1-Score: {class_metrics}")
    
        
    result_path = "../../Results/SamSpa/StrRFEDFrank/StrRFEDFrank"+dtanno_file[0:-5]+".txt"
    with open(result_path, 'w') as f:
        json.dump(whofol_results, f, indent=4)     
    
    train_result_path = "../../Results/SamSpa/StrRFEDFrank/StrRFEDFrank"+dtanno_file[0:-5]+"_train.txt"
    with open(train_result_path, 'w') as f:
        json.dump(train_results, f, indent=4)     
        
    spa_result_path = "../../Results/SamSpa/SpaResults/StrRFEDFrank/StrRFEDFrank"+dtanno_file[0:-5]+".txt" 
    with open(spa_result_path, 'w') as f:
        json.dump(spa_whofol_results, f, indent=4)    
    
    spa_train_result_path = "../../Results/SamSpa/SpaResults/StrRFEDFrank/StrRFEDFrank"+dtanno_file[0:-5]+"_train.txt" 
    with open(spa_train_result_path, 'w') as f:
        json.dump(spa_train_results, f, indent=4) 
   
   
if __name__ == "__main__":
    for dtanno_file in ['dt'+str(i)+".xlsx" for i in range(0,10)]:
        optimal_results(dtanno_file)    
        
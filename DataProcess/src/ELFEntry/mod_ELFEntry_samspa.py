# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json
import itertools
import ast
import pickle

from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, f1_score

def optimal_results(dtanno_file):
    fea_filter = "../../Features/ELFEntry/SamSpa/ELFEntry"+dtanno_file[0:-5]+".xlsx" 
    fea_names_path = "../../Features/ELFEntry/SamSpa/FeaELFEntry"+dtanno_file[0:-5]+".txt"
    
    whofol_feas = pd.ExcelFile(fea_filter)
    with open(fea_names_path, 'r') as file:
        fea_names = json.load(file)
        fea_names = {ast.literal_eval(k): v for k, v in fea_names.items()}
    
    year_list = [2020, 2021, 2022]
    train_arch_list = ['ARM', 'MIPS']
    arch_list = ['68K', 'ARM', 'MIPS', 'SH'] 
    
    whofol_results = {"acc": {}, "f1":{}}
    train_results = {"acc": {}, "f1":{}}
    
    spa_whofol_results = {}
    spa_train_results = {}
    
    def feature_transform(x):
        
        print(x)
        feature_list = json.loads(x)
        
        return feature_list
    
    for train_year, train_arch in itertools.product(year_list, train_arch_list):
              
        sheetname = str(train_year)+train_arch
        
        feature_names = fea_names[train_year][train_arch]
        chunk_names = ["chunk"+str(i) for i in range(100)]
        
        train_fea_df = pd.read_excel(fea_filter, sheet_name=sheetname)        
        test_fea_df = pd.read_excel(fea_filter, sheet_name=sheetname+"_test")          
        
        train_df = train_fea_df[(train_fea_df['year'] == train_year) & (train_fea_df['arch'] == train_arch)]
            
        new_columns = train_df[chunk_names].apply(lambda row: sum([ast.literal_eval(x) for x in row], []), axis=1)
        X_train = pd.DataFrame(new_columns.tolist(), columns=feature_names)
        y_train = train_df['label']
        
        # Initialize classifiers
        model_name_list = ['RF', 'KNN', 'SVC', 'MLP']
        
        # Train and evaluate each model
        print(f"{sheetname}:")
        for name in model_name_list:
            # Train the model
            
            with open("../../Results/BestModels/ELFEntry/ELFEntry"+dtanno_file[0:-5]+"/"+sheetname+f'/best_{name}.pkl', 'rb') as f:
                model = pickle.load(f)
            # Train the model
            model.fit(X_train, y_train)
                
            print(f"- {name}:")
            
            y_train_pred = model.predict(X_train)
                      
            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_f1 = f1_score(y_train, y_train_pred, average='macro')
            
            train_results["acc"].setdefault(name, {})
            train_results["f1"].setdefault(name, {})
            
            train_results["acc"][name][sheetname] = train_accuracy
            train_results["f1"][name][sheetname] = train_f1
            
            print(f"- Accuracy: {train_accuracy:.4f}, F1-Score: {train_f1:.4f}")  
            
            # --- test ---
            
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
                        
            for test_year in year_list:
                
                print(f"-- {test_year}") 
                whofol_results["acc"][name][sheetname].setdefault(test_year, {})
                whofol_results["f1"][name][sheetname].setdefault(test_year, {})
                
                spa_whofol_results[name][sheetname].setdefault(test_year, {})
                
                for test_arch in arch_list:
                
                    test_df = test_fea_df[(test_fea_df['year'] == test_year) & (test_fea_df['arch'] == test_arch)] 
                    
                    new_columns = test_df[chunk_names].apply(lambda row: sum([ast.literal_eval(x) for x in row], []), axis=1)
                    X_test = pd.DataFrame(new_columns.tolist(), columns=feature_names)
                    y_test = test_df['label']
                
                    # Make predictions on the test data
                    y_pred = model.predict(X_test)
                    
                    # Calculate accuracy and F1-score
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='macro')
                                       
                    whofol_results["acc"][name][sheetname][test_year][test_arch] = accuracy
                    whofol_results["f1"][name][sheetname][test_year][test_arch] = f1
                    
                    print(f"--- {test_arch} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")     
                    
                    report = classification_report(y_test, y_pred, output_dict=True)
                    class_metrics = {}

                    for label, metrics in report.items():
                        if label.isdigit():  
                            class_metrics[label] = {
                                'precision': metrics['precision'],
                                'recall': metrics['recall'],
                                'f1_score': metrics['f1-score']
                            }                   
                                       
                    spa_whofol_results[name][sheetname][test_year][test_arch] = class_metrics
                    
                    print(f"--- {test_arch} - F1-Score: {class_metrics}")

    result_path = "../../Results/SamSpa/ELFEntry/ELFEntry"+dtanno_file[0:-5]+".txt" 
    
    with open(result_path, 'w') as f:
        json.dump(whofol_results, f, indent=4)     
    
    train_result_path = "../../Results/SamSpa/ELFEntry/ELFEntry"+dtanno_file[0:-5]+"_train.txt" 
    
    with open(train_result_path, 'w') as f:
       json.dump(train_results, f, indent=4)  
       
    spa_result_path = "../../Results/SamSpa/SpaResults/ELFEntry/ELFEntry"+dtanno_file[0:-5]+".txt" 
   
    with open(spa_result_path, 'w') as f:
       json.dump(spa_whofol_results, f, indent=4)     
   
    spa_train_result_path = "../../Results/SamSpa/SpaResults/ELFEntry/ELFEntry"+dtanno_file[0:-5]+"_train.txt" 
   
    with open(spa_train_result_path, 'w') as f:
       json.dump(spa_train_results, f, indent=4)  
        

if __name__ == "__main__":
    for dtanno_file in ['dt'+str(i)+".xlsx" for i in range(3,5)]:
        optimal_results(dtanno_file)   
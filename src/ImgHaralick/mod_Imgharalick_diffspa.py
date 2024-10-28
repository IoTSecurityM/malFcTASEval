# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json
import itertools
import os
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
from sklearn.model_selection import GridSearchCV

def optimal_results(dtanno_file):
    fea_filter = "../../Features/ImgHaralick/DiffSpa/ImgHaralick"+dtanno_file[0:-5]+".xlsx"
    
    whofol_feas = pd.ExcelFile(fea_filter)
    
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
        
        param_grid_rf = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced']           
        }
        
        param_grid_knn = {
            'knn__n_neighbors': [3, 5, 7, 9],
            'knn__weights': ['uniform', 'distance'],
            'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }   
        
        param_grid_lsvc = {
            'linear_svc__C': [0.001, 0.01, 0.1, 1, 10, 100],  # Wider range of C values
            'linear_svc__penalty': ['l1', 'l2'],  # l1 or l2 regularization
            'linear_svc__loss': ['squared_hinge'],  # Only squared_hinge is valid for l1 penalty
            'linear_svc__max_iter': [2000, 5000],  # Increase the number of iterations
            'linear_svc__class_weight': ['balanced']  # Also test without class weighting
        }
        
        # Define the parameter grid for MLPClassifier
        param_grid_mlp = {
            'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],  # Number of neurons in hidden layers
            'mlp__activation': ['tanh', 'relu'],  # Activation function
            'mlp__solver': ['adam', 'sgd'],  # Optimization algorithms
            'mlp__alpha': [0.0001, 0.001, 0.01],  # L2 regularization term
            'mlp__learning_rate': ['constant', 'adaptive'],  # Learning rate schedule
            'mlp__learning_rate_init': [0.001, 0.01, 0.1],  # Initial learning rate
            'mlp__max_iter': [500, 1000, 2000],  # Number of iterations before stopping
        }
       
        # Dictionary to store models and their names
        rf = RandomForestClassifier(random_state=42)
        knn_pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Apply StandardScaler to KNN
            ('knn', KNeighborsClassifier())
        ])      
        lsvc_pipeline = Pipeline([
            ('scaler', StandardScaler()),  # StandardScaler to normalize the data
            ('linear_svc', LinearSVC(random_state=42))  # LinearSVC model
        ])
        mlp_pipeline = Pipeline([
            ('scaler', StandardScaler()),  # StandardScaler to normalize the data
            ('mlp', MLPClassifier(max_iter=2000, early_stopping=True))  # MLPClassifier with increased max_iter
        ])
        
        models = {
            'RF': (rf, param_grid_rf),
            'KNN': (knn_pipeline, param_grid_knn),       
            'SVC': (lsvc_pipeline, param_grid_lsvc),
            'MLP': (mlp_pipeline, param_grid_mlp)
        } 
        
        # Train and evaluate each model
        print(f"{sheetname}:")
        sheetname_best_model_dir = "../../Results/BestModels/ImgHaralick/ImgHaralick"+dtanno_file[0:-5]+"/"+sheetname
        if not os.path.exists(sheetname_best_model_dir):
            os.makedirs(sheetname_best_model_dir)
        
        for name, (model, param_grid) in models.items():
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='f1_macro', n_jobs=-1)
            
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            with open(sheetname_best_model_dir+f'/best_{name}.pkl', 'wb') as f:
                pickle.dump(best_model, f)
                
            print(f"- {name}:")
            
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
                if label.isdigit():  # Only process actual class labels, ignore 'accuracy', 'macro avg', 'weighted avg'
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
        
    result_path = "../../Results/DiffSpa/ImgHaralick/ImgHaralick"+dtanno_file[0:-5]+".txt"
    with open(result_path, 'w') as f:
        json.dump(whofol_results, f, indent=4)     
    
    train_result_path = "../../Results/DiffSpa/ImgHaralick/ImgHaralick"+dtanno_file[0:-5]+"_train.txt"
    with open(train_result_path, 'w') as f:
        json.dump(train_results, f, indent=4)     
        
    spa_result_path = "../../Results/DiffSpa/SpaResults/ImgHaralick/ImgHaralick"+dtanno_file[0:-5]+".txt" 
    with open(spa_result_path, 'w') as f:
        json.dump(spa_whofol_results, f, indent=4)    
    
    spa_train_result_path = "../../Results/DiffSpa/SpaResults/ImgHaralick/ImgHaralick"+dtanno_file[0:-5]+"_train.txt" 
    with open(spa_train_result_path, 'w') as f:
        json.dump(spa_train_results, f, indent=4) 
        
        

if __name__ == "__main__":
    for dtanno_file in ['dt'+str(i)+".xlsx" for i in range(0,10)]:
        optimal_results(dtanno_file)   
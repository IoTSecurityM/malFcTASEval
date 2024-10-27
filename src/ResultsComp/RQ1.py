# -*- coding: utf-8 -*-
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from pandas.plotting import parallel_coordinates

# from figure_show_util import add_avg_std, mark_func, spider_draw, cell_feature_dict, plot_draw, classifier_algorithm_based
    
    
def diff_prec_rec_f1_f1macro():
    
    # file_shown_lists = ["[25]", "[34]", "[35]", "[37]", "[39]", "[40]"]
    file_shown_lists = ['[39]', '[43]', '[24]', '[25]', '[42]', '[37]']
    train_year_order = [2020, 2021, 2022]
    train_alg_order = file_shown_lists
    train_arch_order = ['ARM', 'MIPS']        
    train_model_order = ['RF', 'KNN', 'SVC', 'MLP']
    
    train_who_results_df_0 = pd.read_csv('../../Results/train_result.csv')
    train_who_results_df_0 = train_who_results_df_0[train_who_results_df_0['metric'] == 'f1'].drop(columns=["metric"])
    train_who_results_df_0['train_alg'] = train_who_results_df_0['train_alg'].replace({'[25]': '[39]', '[34]': '[43]', '[35]': '[24]', '[37]': '[25]', '[39]': '[42]', '[40]': '[37]'})   
    train_who_results_df_0 = train_who_results_df_0.groupby(["train_year", "train_alg", "train_arch", "model"], as_index=False).mean()
    train_who_results_df_1 = pd.read_csv('../../Results/spatest_train_result.csv')
    train_who_results_df_1['train_alg'] = train_who_results_df_1['train_alg'].replace({'[25]': '[39]', '[34]': '[43]', '[35]': '[24]', '[37]': '[25]', '[39]': '[42]', '[40]': '[37]'})  
    train_who_results_df_1 = train_who_results_df_1.groupby(["train_year", "train_alg", "train_arch", "model"], as_index=False).mean()

    train_who_results_merged = pd.merge(train_who_results_df_0, train_who_results_df_1, on=["train_year", "train_alg", "train_arch", "model"])    

    # train_who_results_merged = train_who_results_merged[train_who_results_merged['Performance']>=0.9]    
    # train_who_results_merged = train_who_results_merged[(train_who_results_merged['Performance']<0.9)&(train_who_results_merged['Performance']>=0.8)] 
    train_who_results_merged = train_who_results_merged[train_who_results_merged['Performance']<0.8]
    
    test_who_results_df_0 = pd.read_csv('../../Results/samespa_result.csv')
    test_who_results_df_0 = test_who_results_df_0[test_who_results_df_0['metric'] == 'f1'].drop(columns=["metric"])
    test_who_results_df_0['train_alg'] = test_who_results_df_0['train_alg'].replace({'[25]': '[39]', '[34]': '[43]', '[35]': '[24]', '[37]': '[25]', '[39]': '[42]', '[40]': '[37]'}) 
    test_who_results_df_0 = test_who_results_df_0.groupby(["train_alg", "model", "train_arch", "train_year", "Test Year", "test_arch"], as_index=False).mean()
    test_who_results_df_1 = pd.read_csv('../../Results/spatest_samespa_result.csv')
    test_who_results_df_1['train_alg'] = test_who_results_df_1['train_alg'].replace({'[25]': '[39]', '[34]': '[43]', '[35]': '[24]', '[37]': '[25]', '[39]': '[42]', '[40]': '[37]'}) 
    test_who_results_df_1 = test_who_results_df_1.groupby(["train_alg", "model", "train_arch", "train_year", "Test Year", "test_arch"], as_index=False).mean()
    
    test_who_results_merged = pd.merge(test_who_results_df_0, test_who_results_df_1, on=["train_alg", "model", "train_arch", "train_year", "Test Year", "test_arch"])     
    test_who_results_merged = test_who_results_merged[test_who_results_merged['train_arch']==test_who_results_merged['test_arch']].drop(columns=["test_arch"])
    test_who_results_merged = test_who_results_merged[test_who_results_merged['train_year']==test_who_results_merged['Test Year']].drop(columns=["Test Year"])
    
    merged_df = pd.merge(train_who_results_merged, test_who_results_merged, on=["train_year", "train_alg", "train_arch", "model"], suffixes=('_df1', '_df2'), how='inner')
    for diff_col_name_comps in itertools.product(['0_', '1_', '2_'], ['precision', 'recall', 'f1_score']):   
        diff_col_name = diff_col_name_comps[0]+diff_col_name_comps[1]       
        merged_df[diff_col_name] = merged_df[diff_col_name+'_df2'] - merged_df[diff_col_name+'_df1']
    merged_df['f1_mac'] =  merged_df['Performance_df2'] - merged_df['Performance_df1'] 
    merged_df = merged_df.rename(columns={"0_precision": "0_prec", "1_precision": "1_prec", "2_precision": "2_prec", "0_recall": "0_rec", "1_recall": "1_rec", "2_recall": "2_rec", "0_f1_score": "0_f1", "1_f1_score": "1_f1", "2_f1_score": "2_f1"})   
    
    merged_df = merged_df[["train_year", "train_alg", "train_arch", "model", "0_prec", "1_prec", "2_prec", "0_rec", "1_rec", "2_rec", "0_f1", "1_f1", "2_f1", "f1_mac"]]
    
    merged_df['train_year'] = pd.Categorical(merged_df['train_year'], categories=train_year_order, ordered=True)
    merged_df['train_arch'] = pd.Categorical(merged_df['train_arch'], categories=train_arch_order, ordered=True)   
    merged_df['train_alg'] = pd.Categorical(merged_df['train_alg'], categories=train_alg_order, ordered=True)            
    merged_df['model'] = pd.Categorical(merged_df['model'], categories=train_model_order, ordered=True)                
                 
    merged_df = merged_df.sort_values(by=['train_year', 'train_arch', 'model', 'train_alg'])            
    merged_df = merged_df.reset_index(drop=True)
    
    merged_df['whole_model'] = merged_df['train_alg'].str.cat(merged_df['model'], sep=' ')
    merged_df = merged_df.drop(columns=['train_alg', 'model'])          
   
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))    
    # fig, axes = plt.subplots(3, 2, subplot_kw=dict(polar=True), figsize=(10, 15))                                    
    axe_dict_map = [((0,0), 2020, 'ARM'), ((0,1), 2020, 'MIPS'), ((1,0), 2021, 'ARM'), ((1,1), 2021, 'MIPS'), ((2,0), 2022, 'ARM'), ((2,1), 2022, 'MIPS')]    
    
    merged_df = merged_df.groupby(["train_year", "train_arch"], as_index=False)
    
    for axe_dict in axe_dict_map:       
        extracted_group = merged_df.get_group((axe_dict[1], axe_dict[2]))
        extracted_group = extracted_group.drop(["train_year", "train_arch"], axis=1)
        
        # Create the parallel coordinates plot
        parallel_coordinates(extracted_group, 'whole_model', color=sns.color_palette("Set2", 6), ax=axes[axe_dict[0][0]][axe_dict[0][1]])
        axes[axe_dict[0][0]][axe_dict[0][1]].set_ylim([-0.8, 0.1])
        
        # Remove any existing legend
        axes[axe_dict[0][0]][axe_dict[0][1]].legend_.remove()
        
        # Set tick paramfontweight='bold'eters
        axes[axe_dict[0][0]][axe_dict[0][1]].tick_params(axis='both', labelsize=15)
        for label in axes[axe_dict[0][0]][axe_dict[0][1]].get_xticklabels() + axes[axe_dict[0][0]][axe_dict[0][1]].get_yticklabels():
            label.set_fontweight('bold')
        for label in axes[axe_dict[0][0]][axe_dict[0][1]].get_xticklabels():
            label.set_rotation(65)
        
        # Create an inset axis for detailed x-y value changes
        inset_ax = inset_axes(axes[axe_dict[0][0]][axe_dict[0][1]], width="30%", height="30%", loc='lower right')
        
        # Plot the detailed relationship in the inset
        inset_ax.plot(extracted_group['f1_mac'], marker='o')  # Replace with actual column names
        # inset_ax.set_title('Detail View', fontsize=10)
        # inset_ax.set_xlabel('X Value', fontsize=10)
        inset_ax.set_ylim([-0.3, 0.1])
        inset_ax.set_ylabel('f1_mac', fontsize=12, fontweight='bold')
        inset_ax.set_xticks([])
        
        for label in inset_ax.get_yticklabels():
            label.set_fontsize(12)  # Set font size for y-ticks
            label.set_fontweight('bold')  # Set font weight for y-ticks
        
        # Get handles and labels for the current plot
        handles, labels = axes[axe_dict[0][0]][axe_dict[0][1]].get_legend_handles_labels()
        
        # Add legend to the current subplot, positioned outside
        legend = axes[axe_dict[0][0]][axe_dict[0][1]].legend(
            handles, 
            labels, 
            loc='upper left',           
            bbox_to_anchor=(1.02, 1),     
            borderaxespad=0
        )
        
        # Set font size and weight for legend text
        for text in legend.get_texts():
            text.set_fontsize(16)  # Set font size
            text.set_fontweight('bold')  # Set font weight
            
    fig.text(0.06, 0.77, '2020', fontsize=18, ha='center', fontweight='bold')
    fig.text(0.06, 0.48, '2021', fontsize=18, ha='center', fontweight='bold')
    fig.text(0.06, 0.18, '2022', fontsize=18, ha='center', fontweight='bold')
    
    # Add labels for architectures to the top of each column
    fig.text(0.35, -0.02, 'ARM', fontsize=18, ha='center', fontweight='bold')
    fig.text(0.78, -0.02, 'MIPS', fontsize=18, ha='center', fontweight='bold')
    
    # Show the plot
    plt.tight_layout(rect=[0.1, 0, 1, 0.9])
    # plt.savefig('../../Figure/RQ1/recprecf1diffbig09', bbox_inches='tight', dpi=500)    
    # plt.savefig('../../Figure/RQ1/recprecf1diffles09big08', bbox_inches='tight', dpi=500)
    plt.savefig('../../Figure/RQ1/recprecf1diffles08', bbox_inches='tight', dpi=500)
    plt.show()
    
    

    
    
    

if __name__ == "__main__":
    diff_prec_rec_f1_f1macro()
   

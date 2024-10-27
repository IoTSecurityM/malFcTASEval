import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import cm
import itertools

from figure_show_util import rq3_add_avg_std, mark_func, cell_feature_dict, rq3_plot_draw, rq2_diff_plot_draw, rq3_classifier_algorithm_based
from pandas.plotting import parallel_coordinates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
def diff_prec_rec_f1_f1macro():
    
    # file_shown_lists = ["[25]", "[34]", "[35]", "[37]", "[39]", "[40]"]
    file_shown_lists = ['[38]', '[42]', '[23]', '[24]', '[41]', '[36]']
    train_year_order = [2020, 2021, 2022]
    test_year_order = [2020, 2021, 2022]
    train_alg_order = file_shown_lists
    train_arch_order = ['ARM', 'MIPS']   
    test_arch_order = ['68K', 'SH']     
    train_model_order = ['RF', 'KNN', 'SVC', 'MLP']
    
    same_results_df_0 = pd.read_csv('../../Results/goodmodel_samespa_result.csv')
    same_results_df_0['train_alg'] = same_results_df_0['train_alg'].replace({'[25]': '[38]', '[34]': '[42]', '[35]': '[23]', '[37]': '[24]', '[39]': '[41]', '[40]': '[36]'})   
    same_results_df_1 = pd.read_csv('../../Results/goodmodel_spatest_samespa_result.csv')
    same_results_df_1['train_alg'] = same_results_df_1['train_alg'].replace({'[25]': '[38]', '[34]': '[42]', '[35]': '[23]', '[37]': '[24]', '[39]': '[41]', '[40]': '[36]'})   
    
    same_results_df_0 = same_results_df_0.groupby(["train_alg", "model", "train_arch", "train_year", "Test Year", "test_arch"], as_index=False).mean()
    same_results_df_1 = same_results_df_1.groupby(["train_alg", "model", "train_arch", "train_year", "Test Year", "test_arch"], as_index=False).mean()
    
    same_results_df = pd.merge(same_results_df_0, same_results_df_1, on=["train_alg", "model", "train_arch", "train_year", "Test Year", "test_arch"])    
    
    same_results_df = same_results_df[same_results_df["train_year"] == same_results_df["Test Year"]].drop(columns=["Test Year"])
    
    fig, axes = plt.subplots(3, 6, figsize=(36, 16))   
    
    axes = axes.flatten()
    
    plot_count = 0
    
    f1_decline_list = []
    dec = {'min': {'train_year':0, 'train_arch': 0, 'test_arch': 0, 'dec_val':1, 'train_alg':0, 'train_model':0}, 'max': {'train_year':0, 'train_arch': 0, 'test_arch': 0, 'dec_val':0, 'train_alg':0, 'train_model':0}}
    
    for train_year in train_year_order:
        train_year_df = same_results_df[same_results_df["train_year"] == train_year]
        print('-----------------------------------')
        for train_arch in train_arch_order:          
            train_arch_df = train_year_df[train_year_df['train_arch'] == train_arch]
            update_test_arch_order = test_arch_order.copy()
            if train_arch == 'ARM':               
                update_test_arch_order.append('MIPS')
            else:
                update_test_arch_order.append('ARM')
            print('+++++++++++++++++++')
            for test_arch in update_test_arch_order:                                
                same_year_data = train_arch_df[train_arch_df["test_arch"] == train_arch]
                diff_year_data =  train_arch_df[train_arch_df["test_arch"] == test_arch]
                plot_data = pd.merge(same_year_data, diff_year_data, on=["train_alg", "model", "train_arch", "train_year"], suffixes=('_df1', '_df2'), how='inner')
                for diff_col_name_comps in itertools.product(['0_', '1_', '2_'], ['precision', 'recall', 'f1_score']):   
                    diff_col_name = diff_col_name_comps[0]+diff_col_name_comps[1]       
                    plot_data[diff_col_name] = plot_data[diff_col_name+'_df2'] - plot_data[diff_col_name+'_df1']
                plot_data['Performance'] =  plot_data['Performance_df2'] - plot_data['Performance_df1']        

                df_sorted = plot_data.sort_values(by='Performance', ascending=False)
                               
                # Calculate the top 2/3 rows
                top_n = int(len(df_sorted) * (1/3))
               
                # Select the top rows and the specific columns 'col1' and 'col2'
                top_df = df_sorted.head(top_n)[["train_alg", "model", "Performance_df2"]]
                print(f"training setting: {train_year}-{train_arch}-{test_arch}")
                print(top_df)                      
                
                f1_decline_list.extend(plot_data['Performance'].tolist())
                
                row_with_lowest_abs_perform  = plot_data.loc[plot_data['Performance'].abs().idxmin()]
                if np.abs(row_with_lowest_abs_perform['Performance']) < dec['min']['dec_val']:
                    dec['min']['train_year'] = row_with_lowest_abs_perform['train_year']
                    dec['min']['train_arch'] = row_with_lowest_abs_perform['train_arch']
                    dec['min']['test_arch'] = row_with_lowest_abs_perform['test_arch_df2']
                    dec['min']['dec_val'] = np.abs(row_with_lowest_abs_perform['Performance'])
                    dec['min']['train_alg'] = row_with_lowest_abs_perform['train_alg']
                    dec['min']['train_model'] = row_with_lowest_abs_perform['model']
                    
                row_with_highest_abs_perform  = plot_data.loc[plot_data['Performance'].abs().idxmax()]
                if np.abs(row_with_highest_abs_perform['Performance']) > dec['max']['dec_val']:
                    dec['max']['train_year'] = row_with_highest_abs_perform['train_year']
                    dec['max']['train_arch'] = row_with_highest_abs_perform['train_arch']
                    dec['max']['test_arch'] = row_with_highest_abs_perform['test_arch_df2']
                    dec['max']['dec_val'] = np.abs(row_with_highest_abs_perform['Performance'])
                    dec['max']['train_alg'] = row_with_highest_abs_perform['train_alg']
                    dec['max']['train_model'] = row_with_highest_abs_perform['model']
                    
                plot_data = plot_data.rename(columns={"0_precision": "0_prec", "1_precision": "1_prec", "2_precision": "2_prec", "0_recall": "0_rec", "1_recall": "1_rec", "2_recall": "2_rec", "0_f1_score": "0_f1", "1_f1_score": "1_f1", "2_f1_score": "2_f1", "Performance": "f1_mac"})   
                plot_data = plot_data[["train_alg", "model", "0_prec", "1_prec", "2_prec", "0_rec", "1_rec", "2_rec", "0_f1", "1_f1", "2_f1", "f1_mac"]]  
                plot_data['train_alg'] = pd.Categorical(plot_data['train_alg'], categories=train_alg_order, ordered=True)            
                plot_data['model'] = pd.Categorical(plot_data['model'], categories=train_model_order, ordered=True)                                                
                plot_data = plot_data.sort_values(by=['model', 'train_alg'])            
                plot_data = plot_data.reset_index(drop=True)
                
                plot_data['whole_model'] = plot_data['train_alg'].str.cat(plot_data['model'], sep=' ')
                plot_data = plot_data.drop(columns=['train_alg', 'model'])  
                
                parallel_coordinates(plot_data, 'whole_model', color=sns.color_palette("Set2", 6), ax=axes[plot_count])           
                
                # Remove any existing legend
                axes[plot_count].legend_.remove()
                
                # Set tick paramfontweight='bold'eters
                axes[plot_count].tick_params(axis='both', labelsize=15)
                for label in axes[plot_count].get_xticklabels() + axes[plot_count].get_yticklabels():
                    label.set_fontweight('bold')
                for label in axes[plot_count].get_xticklabels():
                    label.set_rotation(35)
                
                # Create an inset axis for detailed x-y value changes
                inset_ax = inset_axes(axes[plot_count], width="30%", height="30%", loc='lower right')
                
                # Plot the detailed relationship in the inset
                inset_ax.plot(plot_data['f1_mac'], marker='o')  # Replace with actual column names
                # inset_ax.set_title('Detail View', fontsize=10)
                # inset_ax.set_xlabel('X Value', fontsize=10)
                # inset_ax.set_ylim([-0.4, 0.2])
                inset_ax.set_ylabel('f1_mac', fontsize=10, fontweight='bold')
                inset_ax.set_xticks([])
                for label in inset_ax.get_yticklabels():
                    label.set_fontsize(12)  # Set font size for y-ticks
                    label.set_fontweight('bold')  # Set font weight for y-ticks
            
                # Get handles and labels for the current plot
                if (plot_count+1)%3 == 0:
                    handles, labels = axes[plot_count].get_legend_handles_labels()
                    
                    legend = axes[plot_count].legend(
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
                
                plot_count += 1
    
    for row_cnt in range(0,3):        
        for i in range(3,6):  # Starting from the second column (index 1)
        # Adjust the position of the second column plots
            axe_index = i+(6*row_cnt)
            axes[axe_index].set_position([axes[axe_index].get_position().x0 + 0.07, 
                                  axes[axe_index].get_position().y0, 
                                  axes[axe_index].get_position().width, 
                                  axes[axe_index].get_position().height])
           
    fig.text(0.095, 0.76, '2020', fontsize=18, ha='center', fontweight='bold')
    fig.text(0.095, 0.5, '2021', fontsize=18, ha='center', fontweight='bold')
    fig.text(0.095, 0.23, '2022', fontsize=18, ha='center', fontweight='bold')
    
    # Add labels for architectures to the top of each column
    fig.text(0.32, 0.05, 'ARM', fontsize=18, ha='center', fontweight='bold')
    fig.text(0.8, 0.05, 'MIPS', fontsize=18, ha='center', fontweight='bold')
    
    fig.text(0.18, 0.9, '68K', fontsize=18, ha='center', fontweight='bold')
    fig.text(0.311, 0.9, 'SH', fontsize=18, ha='center', fontweight='bold')
    fig.text(0.44, 0.9, 'MIPS', fontsize=18, ha='center', fontweight='bold')
    
    fig.text(0.65, 0.9, '68K', fontsize=18, ha='center', fontweight='bold')
    fig.text(0.78, 0.9, 'SH', fontsize=18, ha='center', fontweight='bold')
    fig.text(0.91, 0.9, 'ARM', fontsize=18, ha='center', fontweight='bold')
   
    mean = np.mean(f1_decline_list)
    std_dev = np.std(f1_decline_list)
    
    # print(f"Mean: {mean}, Standard Deviation: {std_dev}")
    # print(dec)
     
    plt.savefig('../../Figure/RQ3/recprecf1diff', bbox_inches='tight', dpi=500)    
    plt.show()


def perform_samTrainArch():
    
    # file_shown_lists = ["[25]", "[34]", "[35]", "[37]", "[39]", "[40]"]
    file_shown_lists = ['[38]', '[42]', '[23]', '[24]', '[41]', '[36]']
    train_year_order = [2020, 2021, 2022]
    train_alg_order = file_shown_lists
    train_arch_order = ['ARM', 'MIPS']  
    test_year_order = [2020, 2021, 2022]      
    train_model_order = ['RF', 'KNN', 'MLP']
    test_arch_order = ['68K', 'SH'] 
    
    same_results_df = pd.read_csv('../../Results/goodmodel_samespa_result.csv')
    same_results_df['train_alg'] = same_results_df['train_alg'].replace({'[25]': '[38]', '[34]': '[42]', '[35]': '[23]', '[37]': '[24]', '[39]': '[41]', '[40]': '[36]'})   
    
    same_results_df = same_results_df.groupby(["train_alg", "model", "train_arch", "train_year", "Test Year", "test_arch"], as_index=False).mean()
    
    same_results_df = same_results_df[same_results_df["train_year"] == same_results_df["Test Year"]].drop(columns=["Test Year"])
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), subplot_kw={"projection": "3d"})
    
    axes = axes.flatten()
    
    plot_count = 0
    
    f1_decline_list = []
    dec = {'min': {'train_year':0, 'train_arch': 0, 'dec_val':1, 'train_alg':0, 'train_model':0}, 'max': {'train_year':0, 'train_arch': 0, 'dec_val':0, 'train_alg':0, 'train_model':0}}
    
    
    for train_year in train_year_order:
        train_year_df = same_results_df[same_results_df["train_year"] == train_year]
        for train_arch in train_arch_order:          
            train_arch_df = train_year_df[train_year_df['train_arch'] == train_arch]
            train_arch_df = train_arch_df.rename(columns={'train_alg': 'Studied Method', 'model': 'Classifier', 'Performance': 'f1_mac'})
            
            check_68K = train_arch_df[train_arch_df['test_arch'] == '68K'][['Studied Method', 'Classifier', 'f1_mac']]
            check_SH = train_arch_df[train_arch_df['test_arch'] == 'SH'][['Studied Method', 'Classifier', 'f1_mac']]
            
            merge_check_68K_SH = pd.merge(check_68K, check_SH, on=['Studied Method', 'Classifier'], suffixes=('_df1', '_df2'), how='inner')
            merge_check_68K_SH['f1_mac'] = (merge_check_68K_SH['f1_mac_df2']-merge_check_68K_SH['f1_mac_df1']).abs()
            
            f1_decline_list.extend(merge_check_68K_SH['f1_mac'].tolist())
            
            row_with_lowest_abs_perform  = merge_check_68K_SH.loc[merge_check_68K_SH['f1_mac'].idxmin()]
            if np.abs(row_with_lowest_abs_perform['f1_mac']) < dec['min']['dec_val']:
                dec['min']['train_year'] = train_year
                dec['min']['train_arch'] = train_arch
                dec['min']['dec_val'] = row_with_lowest_abs_perform['f1_mac']
                dec['min']['train_alg'] = row_with_lowest_abs_perform['Studied Method']
                dec['min']['train_model'] = row_with_lowest_abs_perform['Classifier']
                
            row_with_highest_abs_perform  = merge_check_68K_SH.loc[merge_check_68K_SH['f1_mac'].abs().idxmax()]
            if np.abs(row_with_highest_abs_perform['f1_mac']) > dec['max']['dec_val']:
                dec['max']['train_year'] = train_year
                dec['max']['train_arch'] = train_arch              
                dec['max']['dec_val'] = row_with_highest_abs_perform['f1_mac']
                dec['max']['train_alg'] = row_with_highest_abs_perform['Studied Method']
                dec['max']['train_model'] = row_with_highest_abs_perform['Classifier']
            
            
            df_set1 = train_arch_df[train_arch_df['test_arch'] == '68K'].pivot(index='Studied Method', columns='Classifier', values='f1_mac')
            df_set2 = train_arch_df[train_arch_df['test_arch'] == 'SH'].pivot(index='Studied Method', columns='Classifier', values='f1_mac')
         
            # Create a meshgrid for the x and y values
            x_categories = df_set1.columns
            y_categories = df_set1.index
            x_numeric = np.arange(len(x_categories))
            y_numeric = np.arange(len(y_categories))
            x, y = np.meshgrid(x_numeric, y_numeric)
            
            # Convert pivoted data to numpy arrays for z-values
            z1 = df_set1.to_numpy()
            z2 = df_set2.to_numpy()         
            
            
            # Surface for z1 with color mapping
            surface1 = axes[plot_count].plot_surface(x, y, z1, cmap=cm.coolwarm, alpha=0.3)
            
            # Surface for z2 with color mapping
            surface2 = axes[plot_count].plot_surface(x, y, z2, cmap=cm.inferno, alpha=0.6)
            
            axes[plot_count].tick_params(axis='both', labelsize=11)
            # Set custom tick labels for x and y axes to show the categories
            axes[plot_count].set_xticks(np.arange(len(x_categories)))
            axes[plot_count].set_xticklabels(x_categories)
            
            axes[plot_count].set_yticks(np.arange(len(y_categories)))
            axes[plot_count].set_yticklabels(y_categories)
            
            # Set labels and title
            axes[plot_count].set_xlabel('Classifiers', fontsize=14, labelpad=10)
            axes[plot_count].set_ylabel('Studied Method', fontsize=14, labelpad=10)
            axes[plot_count].set_zlabel('f1_mac', fontsize=14)
                       
            # Draw lines and markers
            for i in range(len(x_categories)):
                for j in range(len(y_categories)):
                    x_pos = x_numeric[i]
                    y_pos = y_numeric[j]
                    if ~np.isnan(z1[j, i]) and ~np.isnan(z2[j, i]):
                        axes[plot_count].plot([x_pos, x_pos], [y_pos, y_pos], [z1[j, i], z2[j, i]], color='black', linestyle='--', linewidth=1)
                        axes[plot_count].scatter(x_pos, y_pos, z1[j, i], color='purple', marker='o')  # Marker for z1
                        axes[plot_count].scatter(x_pos, y_pos, z2[j, i], color='darkred', marker='^')  # Marker for z2
                    elif ~np.isnan(z1[j, i]) and np.isnan(z2[j, i]):                       
                        axes[plot_count].scatter(x_pos, y_pos, z1[j, i], color='purple', marker='o')  # Marker for z1
                    elif np.isnan(z1[j, i]) and ~np.isnan(z2[j, i]):     
                        axes[plot_count].scatter(x_pos, y_pos, z2[j, i], color='purple', marker='o')  # Marker for z1
                            
            # Colorbar handling
            if (plot_count + 1) % 2 == 0:
                # Add colorbars for both surfaces
                cbar1 = fig.colorbar(surface1, ax=axes[plot_count], orientation='vertical', pad=0.1, shrink=0.8)
                cbar1.set_label('68K', fontsize=14)
                cbar2 = fig.colorbar(surface2, ax=axes[plot_count], orientation='vertical', pad=0.2, shrink=0.8)
                cbar2.set_label('SH', fontsize=14)
    
            plot_count += 1 
                    
        
    for fig_idx in range(0, 6, 2):            
        axe_index = fig_idx
        axes[axe_index].set_position([axes[axe_index].get_position().x0 +0.095, 
                              axes[axe_index].get_position().y0, 
                              axes[axe_index].get_position().width, 
                              axes[axe_index].get_position().height])
            
    fig.text(0.25, 0.76, '2020', fontsize=16, ha='center', fontweight='bold')
    fig.text(0.25, 0.5, '2021', fontsize=16, ha='center', fontweight='bold')
    fig.text(0.25, 0.23, '2022', fontsize=16, ha='center', fontweight='bold')
    
    # Add labels for architectures to the top of each column
    fig.text(0.42, 0.05, 'ARM', fontsize=16, ha='center', fontweight='bold')
    fig.text(0.7, 0.05, 'MIPS', fontsize=16, ha='center', fontweight='bold') 
    
    # Adjust layout to ensure there is no overlap
    plt.tight_layout(pad=2.5)
    plt.savefig('../../Figure/RQ3/performSameTrainArch', bbox_inches='tight', dpi=500)
    plt.show() 
    mean = np.mean(f1_decline_list)
    std_dev = np.std(f1_decline_list)
    
    print(f"Mean: {mean}, Standard Deviation: {std_dev}")
    print(dec)
    
    
def perform_samTestArch():
    
    # file_shown_lists = ["[25]", "[34]", "[35]", "[37]", "[39]", "[40]"]
    file_shown_lists = ['[38]', '[42]', '[23]', '[24]', '[41]', '[36]']
    train_year_order = [2020, 2021, 2022]
    train_alg_order = file_shown_lists
    train_arch_order = ['ARM', 'MIPS']  
    test_year_order = [2020, 2021, 2022]      
    train_model_order = ['RF', 'KNN', 'MLP']
    test_arch_order = ['68K', 'SH'] 
    
    same_results_df = pd.read_csv('../../Results/goodmodel_samespa_result.csv')
    same_results_df['train_alg'] = same_results_df['train_alg'].replace({'[25]': '[38]', '[34]': '[42]', '[35]': '[23]', '[37]': '[24]', '[39]': '[41]', '[40]': '[36]'})   
    
    same_results_df = same_results_df.groupby(["train_alg", "model", "train_arch", "train_year", "Test Year", "test_arch"], as_index=False).mean()
    
    same_results_df = same_results_df[same_results_df["train_year"] == same_results_df["Test Year"]].drop(columns=["Test Year"]).reset_index(drop=True)
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), subplot_kw={"projection": "3d"})
    
    axes = axes.flatten()
    
    plot_count = 0
    
    f1_decline_list = []
    dec = {'min': {'train_year':0, 'test_arch': 0, 'dec_val':1, 'train_alg':0, 'train_model':0}, 'max': {'train_year':0, 'test_arch': 0, 'dec_val':0, 'train_alg':0, 'train_model':0}}
    
    
    for train_year in train_year_order:
        train_year_df = same_results_df[same_results_df["train_year"] == train_year]
        for test_arch in test_arch_order:          
            train_arch_df = train_year_df[train_year_df['test_arch'] == test_arch]
            train_arch_df = train_arch_df.rename(columns={'train_alg': 'Studied Method', 'model': 'Classifier', 'Performance': 'f1_mac'})
            
            ARM_df = train_arch_df[train_arch_df['train_arch'] == 'ARM']
            MIPS_df = train_arch_df[train_arch_df['train_arch'] == 'MIPS']
                  
            ARM_df = ARM_df[['Studied Method', 'Classifier', 'f1_mac']]
            MIPS_df = MIPS_df[['Studied Method', 'Classifier', 'f1_mac']]
            
            merged_check_df = pd.merge(ARM_df, MIPS_df, on=['Studied Method', 'Classifier'], suffixes=('_df1', '_df2'), how='inner')
            merged_check_df['f1_mac'] = (merged_check_df['f1_mac_df2']-merged_check_df['f1_mac_df1']).abs()
            
            f1_decline_list.extend(merged_check_df['f1_mac'].tolist())
            
            row_with_lowest_abs_perform  = merged_check_df.loc[merged_check_df['f1_mac'].idxmin()]
            if np.abs(row_with_lowest_abs_perform['f1_mac']) < dec['min']['dec_val']:
                dec['min']['train_year'] = train_year
                dec['min']['test_arch'] = test_arch
                dec['min']['dec_val'] = row_with_lowest_abs_perform['f1_mac']
                dec['min']['train_alg'] = row_with_lowest_abs_perform['Studied Method']
                dec['min']['train_model'] = row_with_lowest_abs_perform['Classifier']
                
            row_with_highest_abs_perform  = merged_check_df.loc[merged_check_df['f1_mac'].abs().idxmax()]
            if np.abs(row_with_highest_abs_perform['f1_mac']) > dec['max']['dec_val']:
                dec['max']['train_year'] = train_year
                dec['max']['test_arch'] = test_arch              
                dec['max']['dec_val'] = row_with_highest_abs_perform['f1_mac']
                dec['max']['train_alg'] = row_with_highest_abs_perform['Studied Method']
                dec['max']['train_model'] = row_with_highest_abs_perform['Classifier']
                      
            # Identify combinations in df2 that are not in df1
            combined_MIPS_df = MIPS_df.merge(ARM_df[['Studied Method', 'Classifier']], on=['Studied Method', 'Classifier'], how='left', indicator=True)
            missing_in_ARM_df = combined_MIPS_df[combined_MIPS_df['_merge'] == 'left_only'][['Studied Method', 'Classifier']]
            
            # Set col3 to NaN and append to df1
            missing_in_ARM_df['f1_mac'] = np.nan
            updated_ARM_df = pd.concat([ARM_df, missing_in_ARM_df], ignore_index=True)
            
            # Identify combinations in df1 that are not in df2
            combined_ARM_df = ARM_df.merge(MIPS_df[['Studied Method', 'Classifier']], on=['Studied Method', 'Classifier'], how='left', indicator=True)
            missing_in_MIPS_df = combined_ARM_df[combined_ARM_df['_merge'] == 'left_only'][['Studied Method', 'Classifier']]
            
            # Set col3 to NaN and append to df2
            missing_in_MIPS_df['f1_mac'] = np.nan
            updated_MIPS_df = pd.concat([MIPS_df, missing_in_MIPS_df], ignore_index=True)
            
            df_set1 = updated_ARM_df.pivot(index='Studied Method', columns='Classifier', values='f1_mac')
            df_set2 = updated_MIPS_df.pivot(index='Studied Method', columns='Classifier', values='f1_mac')
            
            # Create a meshgrid for the x and y values
            x_categories = df_set1.columns
            y_categories = df_set1.index
            x_numeric = np.arange(len(x_categories))
            y_numeric = np.arange(len(y_categories))
            x, y = np.meshgrid(x_numeric, y_numeric)
            
            # Convert pivoted data to numpy arrays for z-values
            z1 = df_set1.to_numpy()
            z2 = df_set2.to_numpy()         
            
            
            # Surface for z1 with color mapping
            surface1 = axes[plot_count].plot_surface(x, y, z1, cmap=cm.coolwarm, alpha=0.3)
            
            # Surface for z2 with color mapping
            surface2 = axes[plot_count].plot_surface(x, y, z2, cmap=cm.inferno, alpha=0.6)
            
            axes[plot_count].tick_params(axis='both', labelsize=11)
            # Set custom tick labels for x and y axes to show the categories
            axes[plot_count].set_xticks(np.arange(len(x_categories)))
            axes[plot_count].set_xticklabels(x_categories)
            
            axes[plot_count].set_yticks(np.arange(len(y_categories)))
            axes[plot_count].set_yticklabels(y_categories)
            
            # Set labels and title
            axes[plot_count].set_xlabel('Classifiers', fontsize=14, labelpad=10)
            axes[plot_count].set_ylabel('Studied Method', fontsize=14, labelpad=10)
            axes[plot_count].set_zlabel('f1_mac', fontsize=14)
                       
            # Draw lines and markers
            for i in range(len(x_categories)):
                for j in range(len(y_categories)):
                    x_pos = x_numeric[i]
                    y_pos = y_numeric[j]
                    if ~np.isnan(z1[j, i]) and ~np.isnan(z2[j, i]):
                        axes[plot_count].plot([x_pos, x_pos], [y_pos, y_pos], [z1[j, i], z2[j, i]], color='black', linestyle='--', linewidth=1)
                        axes[plot_count].scatter(x_pos, y_pos, z1[j, i], color='purple', marker='o')  # Marker for z1
                        axes[plot_count].scatter(x_pos, y_pos, z2[j, i], color='darkred', marker='^')  # Marker for z2
                    elif ~np.isnan(z1[j, i]) and np.isnan(z2[j, i]):                       
                        axes[plot_count].scatter(x_pos, y_pos, z1[j, i], color='purple', marker='o')  # Marker for z1
                    elif np.isnan(z1[j, i]) and ~np.isnan(z2[j, i]):     
                        axes[plot_count].scatter(x_pos, y_pos, z2[j, i], color='purple', marker='o')  # Marker for z1
                            
            # Colorbar handling
            if (plot_count + 1) % 2 == 0:
                # Add colorbars for both surfaces
                cbar1 = fig.colorbar(surface1, ax=axes[plot_count], orientation='vertical', pad=0.1, shrink=0.8)
                cbar1.set_label('ARM', fontsize=14)
                cbar2 = fig.colorbar(surface2, ax=axes[plot_count], orientation='vertical', pad=0.2, shrink=0.8)
                cbar2.set_label('MIPS', fontsize=14)
    
            plot_count += 1 
                    
        
    for fig_idx in range(0, 6, 2):            
        axe_index = fig_idx
        axes[axe_index].set_position([axes[axe_index].get_position().x0 +0.095, 
                              axes[axe_index].get_position().y0, 
                              axes[axe_index].get_position().width, 
                              axes[axe_index].get_position().height])
            
    fig.text(0.25, 0.76, '2020', fontsize=16, ha='center', fontweight='bold')
    fig.text(0.25, 0.5, '2021', fontsize=16, ha='center', fontweight='bold')
    fig.text(0.25, 0.23, '2022', fontsize=16, ha='center', fontweight='bold')
    
    # Add labels for architectures to the top of each column
    fig.text(0.42, 0.05, '68K', fontsize=16, ha='center', fontweight='bold')
    fig.text(0.7, 0.05, 'SH', fontsize=16, ha='center', fontweight='bold') 
    
    # Adjust layout to ensure there is no overlap
    plt.tight_layout(pad=2.5)
    plt.savefig('../../Figure/RQ3/performSameTestArch', bbox_inches='tight', dpi=500)
    plt.show()   
    
    mean = np.mean(f1_decline_list)
    std_dev = np.std(f1_decline_list)
    
    print(f"Mean: {mean}, Standard Deviation: {std_dev}")
    print(dec)
    

if __name__ == "__main__":
    # diff_prec_rec_f1_f1macro()
    # perform_samTrainArch()
    perform_samTestArch()
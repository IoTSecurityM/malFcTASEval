import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import cm
import itertools

from figure_show_util import rq2_add_avg_std, mark_func, cell_feature_dict, rq2_plot_draw, rq2_diff_plot_draw, classifier_algorithm_based
from pandas.plotting import parallel_coordinates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    
def diff_prec_rec_f1_f1macro():
    
    # file_shown_lists = ["[25]", "[34]", "[35]", "[37]", "[39]", "[40]"]
    file_shown_lists = ['[38]', '[42]', '[23]', '[24]', '[41]', '[36]']
    train_year_order = [2020, 2021, 2022]
    test_year_order = [2020, 2021, 2022]
    train_alg_order = file_shown_lists
    train_arch_order = ['ARM', 'MIPS']        
    train_model_order = ['RF', 'KNN', 'SVC', 'MLP']
    
    same_results_df_0 = pd.read_csv('../../Results/goodmodel_samespa_result.csv')
    same_results_df_0['train_alg'] = same_results_df_0['train_alg'].replace({'[25]': '[38]', '[34]': '[42]', '[35]': '[23]', '[37]': '[24]', '[39]': '[41]', '[40]': '[36]'})   
    same_results_df_1 = pd.read_csv('../../Results/goodmodel_spatest_samespa_result.csv')
    same_results_df_1['train_alg'] = same_results_df_1['train_alg'].replace({'[25]': '[38]', '[34]': '[42]', '[35]': '[23]', '[37]': '[24]', '[39]': '[41]', '[40]': '[36]'})   
    
    same_results_df_0 = same_results_df_0.groupby(["train_alg", "model", "train_arch", "train_year", "Test Year", "test_arch"], as_index=False).mean()
    same_results_df_1 = same_results_df_1.groupby(["train_alg", "model", "train_arch", "train_year", "Test Year", "test_arch"], as_index=False).mean()
    
    same_results_df = pd.merge(same_results_df_0, same_results_df_1, on=["train_alg", "model", "train_arch", "train_year", "Test Year", "test_arch"])    
    
    same_results_df = same_results_df[same_results_df["train_arch"] == same_results_df["test_arch"]].drop(columns=["test_arch"])
    
    fig, axes = plt.subplots(3, 6, figsize=(36, 16))   
    
    axes = axes.flatten()
    
    plot_count = 0
    
    f1_decline_list = []
    dec = {'min': {'train_year':0, 'train_arch': 0, 'test_year': 0, 'dec_val':1, 'train_alg':0, 'train_model':0}, 'max': {'train_year':0, 'train_arch': 0, 'test_year': 0, 'dec_val':0, 'train_alg':0, 'train_model':0}}
    
    for train_year in train_year_order:
        train_year_df = same_results_df[same_results_df["train_year"] == train_year]
        for train_arch in train_arch_order:          
            train_arch_df = train_year_df[train_year_df['train_arch'] == train_arch]
            for test_year in test_year_order:                 
                if test_year == train_year:
                    plot_data = train_arch_df[train_arch_df["Test Year"] == train_year]
                else:
                    same_year_data = train_arch_df[train_arch_df["Test Year"] == train_year]
                    diff_year_data =  train_arch_df[train_arch_df["Test Year"] == test_year]
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
                    print(f"training setting: {train_year}-{train_arch}-{test_year}")
                    print(top_df)
                    
                    f1_decline_list.extend(plot_data['Performance'].tolist())
                    
                    row_with_lowest_abs_perform  = plot_data.loc[plot_data['Performance'].abs().idxmin()]
                    if np.abs(row_with_lowest_abs_perform['Performance']) < dec['min']['dec_val']:
                        dec['min']['train_year'] = row_with_lowest_abs_perform['train_year']
                        dec['min']['train_arch'] = row_with_lowest_abs_perform['train_arch']
                        dec['min']['test_year'] = row_with_lowest_abs_perform['Test Year_df2']
                        dec['min']['dec_val'] = np.abs(row_with_lowest_abs_perform['Performance'])
                        dec['min']['train_alg'] = row_with_lowest_abs_perform['train_alg']
                        dec['min']['train_model'] = row_with_lowest_abs_perform['model']
                        
                    row_with_highest_abs_perform  = plot_data.loc[plot_data['Performance'].abs().idxmax()]
                    if np.abs(row_with_highest_abs_perform['Performance']) > dec['max']['dec_val']:
                        dec['max']['train_year'] = row_with_highest_abs_perform['train_year']
                        dec['max']['train_arch'] = row_with_highest_abs_perform['train_arch']
                        dec['max']['test_year'] = row_with_highest_abs_perform['Test Year_df2']
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
                
                if test_year == train_year:
                    axes[plot_count].set_ylim([0, 1])
                else:
                    axes[plot_count].set_ylim([-1, 0.1])
                
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
                inset_ax.set_ylabel('f1_mac', fontsize=12, fontweight='bold')
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
                    
                    # # Add legend to the current subplot, positioned outside
                    # axes[plot_count].legend(
                    #     handles, 
                    #     labels, 
                    #     loc='upper left',           
                    #     bbox_to_anchor=(1.02, 1),     
                    #     fontsize=14,
                    #     borderaxespad=0.
                    # )
                
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
    
    fig.text(0.18, 0.9, '2020', fontsize=18, ha='center', fontweight='bold')
    fig.text(0.311, 0.9, '2021', fontsize=18, ha='center', fontweight='bold')
    fig.text(0.44, 0.9, '2022', fontsize=18, ha='center', fontweight='bold')
    
    fig.text(0.65, 0.9, '2020', fontsize=18, ha='center', fontweight='bold')
    fig.text(0.78, 0.9, '2021', fontsize=18, ha='center', fontweight='bold')
    fig.text(0.91, 0.9, '2022', fontsize=18, ha='center', fontweight='bold')
   
    mean = np.mean(f1_decline_list)
    std_dev = np.std(f1_decline_list)
    
    # print(f"Mean: {mean}, Standard Deviation: {std_dev}")
    # print(dec)
   
    # # Show the plot
    # # plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.savefig('../../Figure/RQ2/recprecf1diff', bbox_inches='tight', dpi=500)    
    plt.show()


def model_perform():
    
    # file_shown_lists = ["[25]", "[34]", "[35]", "[37]", "[39]", "[40]"]
    file_shown_lists = ['[38]', '[42]', '[23]', '[24]', '[41]', '[36]']
    train_year_order = [2020, 2021, 2022]
    train_alg_order = file_shown_lists
    train_arch_order = ['ARM', 'MIPS']  
    test_year_order = [2020, 2021, 2022]      
    train_model_order = ['RF', 'KNN', 'SVC', 'MLP']
    
    same_results_df = pd.read_csv('../../Results/goodmodel_samespa_result.csv')
    same_results_df['train_alg'] = same_results_df['train_alg'].replace({'[25]': '[38]', '[34]': '[42]', '[35]': '[23]', '[37]': '[24]', '[39]': '[41]', '[40]': '[36]'})   
    
    same_results_df = same_results_df.groupby(["train_alg", "model", "train_arch", "train_year", "Test Year", "test_arch"], as_index=False).mean()     
    
    same_results_df = same_results_df[same_results_df["train_arch"] == same_results_df["test_arch"]].drop(columns=["test_arch"])
    
    same_results_df = same_results_df.rename(columns={'train_alg': 'Studied Method', 'model': 'Classifier', "Performance": "f1_mac"})   
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), subplot_kw={"projection": "3d"})
    
    axes = axes.flatten()
    
    plot_count = 0  # Initialize plot count
    
       
    for train_year in train_year_order:
        train_year_df = same_results_df[same_results_df["train_year"] == train_year]
        for train_arch in train_arch_order:
            train_arch_df = train_year_df[train_year_df['train_arch'] == train_arch]
            related_test_year = test_year_order.copy()
            related_test_year.remove(train_year)
    
            # Pivot data
            df_set1 = train_arch_df[train_arch_df['Test Year'] == related_test_year[0]].pivot(index='Studied Method', columns='Classifier', values='f1_mac')
            df_set2 = train_arch_df[train_arch_df['Test Year'] == related_test_year[1]].pivot(index='Studied Method', columns='Classifier', values='f1_mac')  
    
            # Prepare data for plotting
            x_categories = df_set1.columns
            y_categories = df_set1.index
            x_numeric = np.arange(len(x_categories))
            y_numeric = np.arange(len(y_categories))
            x, y = np.meshgrid(x_numeric, y_numeric)
    
            z1 = df_set1.to_numpy()
            z2 = df_set2.to_numpy()
    
            # z_min = np.min([z1, z2])
            # z_max = np.max([z1, z2])
            # norm = plt.Normalize(z_min, z_max)
    
            # Plot surfaces
            surface1 = axes[plot_count].plot_surface(x, y, z1, cmap=cm.coolwarm, alpha=0.3)
            surface2 = axes[plot_count].plot_surface(x, y, z2, cmap=cm.inferno, alpha=0.6)
            
            axes[plot_count].tick_params(axis='both', labelsize=11)
            # for label in axes[plot_count].get_xticklabels():
            #     label.set_rotation(35)
    
            # Set ticks and labels
            axes[plot_count].set_xticks(np.arange(len(x_categories)))
            axes[plot_count].set_xticklabels(x_categories)
            
            axes[plot_count].set_yticks(np.arange(len(y_categories)))
            axes[plot_count].set_yticklabels(y_categories)
    
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
                            
            # if (plot_count + 1) % 2 == 1:
            #     axes[plot_count].set_position([0.1, 0.55, 0.35, 0.35])
            
            # Colorbar handling
            if (plot_count + 1) % 2 == 0:
                # Add colorbars for both surfaces
                cbar1 = fig.colorbar(surface1, ax=axes[plot_count], orientation='vertical', pad=0.1, shrink=0.8)
                cbar1.set_label(str(related_test_year[0]), fontsize=14)
                cbar2 = fig.colorbar(surface2, ax=axes[plot_count], orientation='vertical', pad=0.2, shrink=0.8)
                cbar2.set_label(str(related_test_year[1]), fontsize=14)
    
            plot_count += 1 
            
    for fig_idx in range(0, 6, 2):            
        axe_index = fig_idx
        axes[axe_index].set_position([axes[axe_index].get_position().x0 +0.095, 
                              axes[axe_index].get_position().y0, 
                              axes[axe_index].get_position().width, 
                              axes[axe_index].get_position().height])
            
    fig.text(0.28, 0.76, '2020', fontsize=16, ha='center', fontweight='bold')
    fig.text(0.28, 0.5, '2021', fontsize=16, ha='center', fontweight='bold')
    fig.text(0.28, 0.23, '2022', fontsize=16, ha='center', fontweight='bold')
    
    # Add labels for architectures to the top of each column
    fig.text(0.42, 0.05, 'ARM', fontsize=16, ha='center', fontweight='bold')
    fig.text(0.7, 0.05, 'MIPS', fontsize=16, ha='center', fontweight='bold') 
    
    plt.tight_layout(pad=0)
    plt.savefig('../../Figure/RQ2/perform', bbox_inches='tight', dpi=300)
    plt.show() 

        
    
if __name__ == "__main__":
   
    # diff_prec_rec_f1_f1macro()
    model_perform()
    
    
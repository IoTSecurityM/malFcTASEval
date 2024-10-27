# -*- coding: utf-8 -*-
import pandas as pd
import itertools
from scipy.stats import f_oneway

import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

import textwrap


def anova(dtanno_file, column_list, train_arch_list):
    fea_filter = "../../Features/StrRFEDFrank/SamSpa/StrRFEDFrank"+dtanno_file[0:-5]+".xlsx"
    # fea_filter = "../../Features/ELFMiner/SamSpa/ELFMiner"+dtanno_file[0:-5]+".xlsx"
    
    # Load the Excel file
    whofol_feas = pd.ExcelFile(fea_filter)
    
    # Get the sheet names    
    year_list = [2020, 2021, 2022]
    test_arch_list = ['68K', 'SH']
    
    print(f'--------------------------{dtanno_file[0:-5]}')
    
    for train_year, train_arch in itertools.product(year_list, train_arch_list):
        
        print(f'----------------{train_year}-{train_arch}')
        
        sheetname = str(train_year)+train_arch
        
        test_fea_df = pd.read_excel(fea_filter, sheet_name=sheetname+"_test") 
        
        same_year_test_fea_df = test_fea_df[test_fea_df['year'] == train_year]
        
        updated_test_arch_list = test_arch_list + [train_arch] 
        
        anno_dict = {}
        common_filtered_fea = []        
        
        for test_arch in updated_test_arch_list:
            
            print(f'---------{test_arch}')
            
            test_arch_same_year_test_fea_df = same_year_test_fea_df[same_year_test_fea_df['arch'] == test_arch]
            
            tested_df = test_arch_same_year_test_fea_df.drop(columns=['location', 'year', 'arch']).reset_index(drop=True)           
            
            features = [col for col in tested_df.columns if col not in ['label']]
            
            anova_results = {}
            for feature in features:
                # Group feature values by family within each dataset
                values_by_family = [tested_df[tested_df['label'] == label][feature].values for label in tested_df['label'].unique()]
                
                # Perform one-way ANOVA
                f_stat, p_value = f_oneway(*values_by_family)
                
                # Store the results
                anova_results[feature] = {'F-statistic': f_stat, 'p-value': p_value}

            # Convert results to DataFrame for better visualization
            anova_df = pd.DataFrame(anova_results).T                     
            
            sorted_df = anova_df.sort_values(by=['F-statistic', 'p-value'], ascending=[False, True]).head(20)
            
            filtered_features = sorted_df.index.tolist()
            
            # print(sorted_df)       
            
            if test_arch == '68K':
                common_filtered_fea = filtered_features
            else:
                common_filtered_fea = list(set(common_filtered_fea) & set(filtered_features))
                
            columns_to_select = filtered_features + ['label']
            filtered_column_df = tested_df[columns_to_select] 
            anno_dict[test_arch] = filtered_column_df
        
        # print('+=+=++=++=+=++=++=+=++=+')
        # print(common_filtered_fea)
        
        # break
        # if dtanno_file[0:-5] == 'dt0' and train_year == 2020: 
        # if dtanno_file[0:-5] == 'dt0' and train_year == 2020: 
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(19, 5))
        plot_count = 0    
            
        for test_arch in updated_test_arch_list:
            
            plot_show_df = anno_dict[test_arch][common_filtered_fea+['label']]
          
            df_melted = plot_show_df.melt(id_vars='label', var_name='Feature', value_name='Value')
            
            # Create a boxplot using seaborn with hue as label
           
            # ax = sns.barplot(x='Feature', y='Value', hue='label', data=df_melted, ax=axes[plot_count])
            # sns.stripplot(
            #     data=df_melted, x="Feature", y="Value", hue="label",
            #     dodge=True, alpha=.2, legend=False, ax=axes[plot_count]
            # )
            
            ax = sns.pointplot(
                data=df_melted, x="Feature", y="Value", hue="label",
                dodge=.4, ax=axes[plot_count],linestyles="none",
                
            )
            
            if train_arch == 'ARM':
                ax.set_ylim(0, 400)
            else:
                ax.set_ylim(0, 400)
                
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=['Mirai', 'Gafgyt', 'Tsunami'])
    
            ax.tick_params(axis='x', labelsize=16)  # Adjust as needed
            ax.tick_params(axis='y', labelsize=16)  # Adjust as needed
            ax.set_xlabel('Feature', fontsize=18)  # Set x-axis label size
            ax.set_ylabel('Value', fontsize=18)      
            
            wrapped_labels = [textwrap.fill(label.get_text(), width=5) for label in ax.get_xticklabels()]  # Wrap with a width of your choice
            ax.set_xticklabels(wrapped_labels, rotation=25,)  # Set the wrapped labels and rotate
            
            plot_count += 1
            
        # plt.title(f'{dtanno_file[0:-5]} : {train_year}')
        fig.text(0.2, -0.04, '68K', fontsize=18, ha='center', fontweight='bold')
        fig.text(0.52, -0.04, 'SH', fontsize=18, ha='center', fontweight='bold') 
        fig.text(0.85, -0.04, 'ARM', fontsize=18, ha='center', fontweight='bold') 
        
        plt.tight_layout()
        plt.savefig('../../Figure/ModelAnaly/FDAG/' + train_arch + '/37/FDAG' + dtanno_file[0:-5] + str(train_year), bbox_inches='tight', dpi=500)  
        # plt.savefig('../../Figure/ModelAnaly/MIPSFDTS', bbox_inches='tight', dpi=500)  
        plt.show()
        
        
        
            
        column_list.append(common_filtered_fea)
            
    return column_list
                
        #     filtered_features = sorted_df.index.tolist()

        #     # Ensure 'label' is included in the selected columns
        #     columns_to_select = filtered_features + ['label']
            
        #     # Select only those columns from tested_df
        #     filtered_column_df = tested_df[columns_to_select]
            
        #     anno_dict[test_year] = filtered_column_df
            
        #     if test_year == 2020:
        #         common_filtered_fea = filtered_features
        #     else:
        #         common_filtered_fea = list(set(common_filtered_fea) & set(filtered_features))
               
        # print(common_filtered_fea)
        
        # for test_year in year_list:
            
        #     plot_show_df = anno_dict[test_year][common_filtered_fea+['label']]
          
        #     df_melted = plot_show_df.melt(id_vars='label', var_name='feature', value_name='value')
            
        #     # Create a boxplot using seaborn with hue as label
        #     plt.figure(figsize=(10, 6))
        #     sns.boxplot(x='feature', y='value', hue='label', data=df_melted)
            
        #     plt.ylim(0, 100)

            
        #     plt.title('Boxplot of Features by Label')
        #     plt.show()
            
           
        
            
    


if __name__ == "__main__":
    
    train_arch_list = ['ARM', 'MIPS'] #, 'MIPS'
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))   
    
    axes = axes.flatten()
    
    plot_count = 0
    
    # reference 37
    # column_list_list = [[['bin26', 'bin20', 'bin33', 'bin28', 'bin40', 'bin13', 'bin15', 'bin21', 'bin24', 'bin22', 'bin10', 'bin11', 'bin23', 'bin18', 'bin36', 'bin14', 'bin19', 'bin12', 'bin30', 'bin27', 'bin29', 'bin32', 'bin25', 'bin39', 'bin16', 'bin17'], ['bin26', 'bin20', 'bin33', 'bin28', 'bin40', 'bin13', 'bin15', 'bin24', 'bin31', 'bin22', 'bin10', 'bin11', 'bin23', 'bin14', 'bin19', 'bin12', 'bin30', 'bin27', 'bin29', 'bin25', 'bin39', 'bin42', 'bin41', 'bin16', 'bin21'], ['bin26', 'bin20', 'bin33', 'bin13', 'bin8', 'bin15', 'bin24', 'bin31', 'bin22', 'bin10', 'bin11', 'bin23', 'bin34', 'bin18', 'bin19', 'bin12', 'bin27', 'bin29', 'bin9', 'bin32', 'bin25', '0000', 'bin21'], ['bin26', 'bin20', 'bin33', 'bin28', 'bin40', 'bin13', 'bin15', 'bin21', 'bin24', 'bin31', 'bin22', 'bin10', 'bin23', 'bin34', 'bin18', 'bin14', 'bin36', 'bin19', 'bin12', 'bin30', 'bin27', 'bin29', 'bin32', 'bin25', 'bin39', 'bin16', 'bin17'], ['bin26', 'bin33', 'bin28', 'bin40', 'bin13', 'bin15', 'bin24', 'bin31', 'bin10', 'bin11', 'bin23', 'bin18', 'bin14', 'bin19', 'bin12', 'bin30', 'bin27', 'bin29', 'bin25', 'bin39', 'bin42', 'bin41', 'bin16'], ['bin26', 'bin20', 'bin28', 'bin33', 'bin8', 'bin24', 'bin31', 'bin35', 'bin10', 'bin11', 'bin43', 'bin23', 'bin34', 'bin14', 'bin19', 'bin12', 'bin30', 'bin27', 'bin29', 'bin25', 'bin16', 'bin21'], ['bin26', 'bin20', 'bin28', 'bin33', 'bin15', 'bin21', 'bin24', 'bin31', 'bin22', 'bin35', 'bin38', 'bin23', 'bin34', 'bin18', 'bin14', 'bin19', 'bin12', 'bin30', 'bin27', 'bin29', 'bin9', 'bin49', 'bin32', 'bin25', 'bin39', 'bin16', 'bin17'], ['bin26', 'bin20', 'bin33', 'bin28', 'bin40', 'bin13', 'bin15', 'bin24', 'bin31', 'bin22', 'bin10', 'bin11', 'bin23', 'bin18', 'bin14', 'bin19', 'bin12', 'bin30', 'bin27', 'bin29', 'bin25', 'bin39', 'bin16', 'bin21'], ['bin26', 'bin28', 'bin13', 'bin15', 'bin22', 'bin35', 'bin38', 'bin10', 'bin11', 'bin36', 'bin14', 'bin12', 'bin29', 'bin27', 'bin9', 'bin25', 'bin17', 'bin16', 'bin21'], ['bin26', 'bin20', 'bin33', 'bin28', 'bin40', 'bin13', 'bin15', 'bin21', 'bin24', 'bin31', 'bin22', 'bin10', 'bin43', 'bin23', 'bin34', 'bin18', 'bin36', 'bin14', 'bin19', 'bin12', 'bin30', 'bin27', 'bin29', 'bin32', 'bin25', 'bin39', 'bin16', 'bin17'], ['bin26', 'bin20', 'bin33', 'bin28', 'bin40', 'bin13', 'bin15', 'bin24', 'bin31', 'bin22', 'bin10', 'bin11', 'bin23', 'bin14', 'bin19', 'bin12', 'bin30', 'bin27', 'bin29', 'bin25', 'bin39', 'bin16', 'bin21'], ['bin20', 'bin33', 'bin40', 'bin13', 'bin8', 'bin15', 'bin24', 'bin31', 'bin22', 'bin10', 'bin11', 'bin23', 'bin14', 'bin19', 'bin12', 'bin29', 'bin9', 'bin25', 'bin21'], ['bin26', 'bin20', 'bin33', 'bin28', 'bin40', 'bin13', 'bin15', 'bin21', 'bin24', 'bin31', 'bin22', 'bin10', 'bin43', 'bin23', 'bin34', 'bin18', 'bin36', 'bin14', 'bin19', 'bin12', 'bin30', 'bin27', 'bin29', 'bin32', 'bin25', 'bin39', 'bin16', 'bin17'], ['bin26', 'bin20', 'bin33', 'bin28', 'bin40', 'bin13', 'bin15', 'bin24', 'bin31', 'bin22', 'bin10', 'bin11', 'bin23', 'bin18', 'bin14', 'bin19', 'bin12', 'bin30', 'bin27', 'bin29', 'bin25', 'bin39', 'bin16', 'bin21'], ['bin26', 'bin20', 'bin33', 'bin13', 'bin8', 'bin15', 'bin24', 'bin31', 'bin22', 'bin35', 'bin10', 'bin11', 'bin23', 'bin18', 'bin14', 'bin36', 'bin19', 'bin12', 'bin29', 'bin9', 'bin32', 'bin25', 'bin17', 'bin21']],
    #                     [['000000', 'bin26', 'bin20', 'bin28', 'bin13', 'bin15', 'bin21', 'bin24', 'bin31', 'bin22', 'bin38', 'bin23', 'bin18', 'bin14', 'bin19', 'bin12', 'bin30', 'bin29', 'bin27', 'bin9', 'bin32', 'bin25', 'bin39', 'bin16', 'bin17'], ['bin26', 'bin28', 'bin33', 'bin13', 'bin46', 'bin15', 'bin31', 'bin22', 'bin34', 'bin18', 'bin14', 'bin12', 'bin30', 'bin27', 'bin29', 'bin32', 'bin25', 'bin39', 'bin37', 'bin16', 'bin17'], ['bin23', 'bin34', 'bin26', 'bin20', 'bin8', 'bin21', 'bin19', 'bin24', 'bin12', 'bin30', 'bin40', 'bin31', 'bin22', 'bin27', 'bin35', 'bin10', 'bin25', 'bin43'], ['bin26', 'bin20', 'bin33', 'bin28', 'bin40', 'bin15', 'bin21', 'bin24', 'bin22', 'bin10', 'bin23', 'bin18', 'bin14', 'bin36', 'bin19', 'bin12', 'bin30', 'bin27', 'bin29', 'bin32', 'bin25', 'bin39', 'bin16', 'bin17'], ['bin26', 'bin33', 'bin28', 'bin13', 'bin46', 'bin15', 'bin24', 'bin31', 'bin22', 'bin23', 'bin34', 'bin18', 'bin14', 'bin12', 'bin30', 'bin27', 'bin29', 'bin32', 'bin25', 'bin39', 'bin37', 'bin16', 'bin17'], ['bin34', 'bin36', 'bin14', 'bin28', 'bin30', 'bin31', 'bin22', 'bin29', 'bin32', 'bin10', 'bin38', 'bin11', 'bin25', 'bin41', 'bin39'], ['000000', 'bin26', 'bin20', 'bin28', 'bin33', 'bin15', 'bin21', 'bin24', 'bin31', 'bin22', 'bin38', 'bin23', 'bin18', 'bin14', 'bin19', 'bin12', 'bin30', 'bin27', 'bin29', 'bin9', 'bin32', 'bin25', 'bin39', 'bin16', 'bin17'], ['bin26', 'bin28', 'bin33', 'bin13', 'bin46', 'bin15', 'bin31', 'bin22', 'bin23', 'bin34', 'bin18', 'bin14', 'bin12', 'bin30', 'bin27', 'bin29', 'bin32', 'bin25', 'bin39', 'bin37', 'bin16', 'bin17'], ['000000', 'bin26', 'bin40', 'bin13', '0000000000', 'bin8', 'bin24', 'bin31', 'bin35', 'bin10', 'bin11', 'bin34', 'bin12', 'bin30', 'bin27', 'bin32', 'bin25', 'bin37', '10765'], ['000000', 'bin26', 'bin20', 'bin33', 'bin28', 'bin40', 'bin13', 'bin15', 'bin21', 'bin24', 'bin31', 'bin22', 'bin10', 'bin23', 'bin18', 'bin36', 'bin14', 'bin19', 'bin12', 'bin30', 'bin27', 'bin29', 'bin32', 'bin25', 'bin39', 'bin16', 'bin17'], ['bin26', 'bin33', 'bin28', 'bin13', 'bin46', 'bin15', 'bin24', 'bin31', 'bin22', 'bin23', 'bin34', 'bin18', 'bin14', 'bin12', 'bin30', 'bin29', 'bin27', 'bin32', 'bin25', 'bin39', 'bin37', 'bin16', 'bin17'], ['bin26', 'bin20', 'bin40', 'bin13', 'bin8', 'bin15', 'bin35', 'bin10', 'bin11', 'bin43', 'bin23', 'bin36', 'bin14', 'bin19', 'bin12', 'bin30', 'bin27', 'bin32', 'bin25', 'bin21'], ['000000', 'bin26', 'bin20', 'bin28', 'bin33', 'bin13', 'bin15', 'bin21', 'bin24', 'bin31', 'bin22', 'bin35', 'bin38', 'bin23', 'bin18', 'bin14', 'bin19', 'bin12', 'bin30', 'bin29', 'bin27', 'bin9', 'bin25', 'bin39', 'bin42', 'bin16', 'bin17'], ['bin26', 'bin28', 'bin33', 'bin13', 'bin46', 'bin15', 'bin31', 'bin22', 'bin23', 'bin34', 'bin18', 'bin14', 'bin12', 'bin30', 'bin27', 'bin29', 'bin32', 'bin25', 'bin39', 'bin37', 'bin16', 'bin17'], ['000000', 'bin26', 'bin20', 'bin28', 'bin8', 'bin21', 'bin31', 'bin22', 'bin35', 'bin10', 'bin34', 'bin18', 'bin19', 'bin12', 'bin30', 'bin27', 'bin29', 'bin25', '0000', 'bin17']]]
    # # reference 25
    # column_list_list = [[['symtab_local_objects', 'symtab_global_objects', '.rodata_section_size', 'entry_point', 'symtab_files', 'symtab_global_symbols'], ['symtab_global_functions', 'symtab_global_objects', 'symtab_weak_objects', 'symtab_files', 'symtab_global_symbols', '.comment_section_size'], ['0_file_size', '.comment_section_type', '.comment_section_info', 'symtab_files', '.comment_section_flags', '0_memory_size', '1_memory_size', 'symtab_weak_functions', 'symtab_weak_symbols', '.text_section_size', '3_alignment', '.comment_section_size'], ['symtab_local_objects', 'symtab_global_objects', 'symtab_files', '.rodata_section_size'], ['symtab_global_functions', 'symtab_local_objects', 'symtab_global_objects', '.symtab_section_size', 'symtab_files', 'symtab_global_symbols', '.comment_section_size'], ['.dynsym_section_flags', '1_file_size', '.hash_section_flags', '0_file_size', '.comment_section_link', '.comment_section_addr_align', '.dynstr_section_info', '4_segment_type', '.dynstr_section_flags', 'symtab_files', '0_memory_size', '1_memory_size', 'symtab_weak_functions', 'symtab_weak_symbols', '.text_section_size', '.comment_section_size', '.dynamic_section_info'], ['symtab_local_objects', 'symtab_global_objects', '.rodata_section_size'], ['symtab_global_functions', 'symtab_local_objects', 'symtab_global_objects', '.symtab_section_size', 'symtab_files', 'symtab_global_symbols', 'symtab_total_symbols'], ['.dynsym_section_flags', 'symtab_global_objects', '0_file_size', '.dynamic_section_flags', '.comment_section_type', '.comment_section_info', '.plt_section_type', '.hash_section_link', '.dynamic_section_type', 'symtab_files', '0_memory_size', '1_memory_size', '.interp_section_type', '.dynsym_section_type', '.dynstr_section_type', 'symtab_weak_functions', '.comment_section_size'], ['symtab_local_objects', 'symtab_global_objects', '.text_section_size', 'symtab_global_symbols'], ['symtab_global_functions', 'symtab_global_objects', 'symtab_weak_objects', 'symtab_files', 'symtab_global_symbols', '.comment_section_size'], ['1_file_size', '.data_section_addr_align', '.comment_section_link', '.got_section_addr_align', '.comment_section_type', '.comment_section_info', 'symtab_files', '.dynamic_section_type', '.got_section_size', 'symtab_weak_symbols', '1_memory_size', 'symtab_weak_functions', '.dynsym_section_type', '.dynstr_section_type', '.text_section_size', '.comment_section_size'], ['entry_point', '.text_section_size', '.rodata_section_size'], ['symtab_global_functions', 'symtab_local_objects', 'symtab_global_objects', '.symtab_section_size', 'symtab_files', 'symtab_global_symbols', '.comment_section_size'], ['symtab_global_objects', '.comment_section_info', '.comment_section_size', '1_file_size', '1_memory_size', '.text_section_size', '.dynstr_section_flags', 'symtab_weak_functions', '.dynstr_section_type', '.dynstr_section_link', '.dynamic_section_info', '0_file_size', '.comment_section_link', '.dynsym_section_info', 'symtab_files', '0_memory_size', '.interp_section_flags', '.hash_section_type', '.dynamic_section_flags', 'symtab_weak_symbols']],
    #                     [['symtab_global_functions', 'symtab_global_objects', 'symtab_local_functions', '.strtab_section_size', '.symtab_section_size', '.got_section_addr_align', '.got_section_type', '.got_section_flags', 'symtab_global_symbols', '2_memory_size', 'symtab_total_symbols'], ['symtab_global_objects', '.strtab_section_type', '.comment_section_size', '.strtab_section_flags', '1_file_size', 'symtab_sections', '.symtab_section_flags', '.symtab_section_link', '.symtab_section_addr_align', 'symtab_weak_objects', '.got_section_info', '2_segment_type', 'symtab_weak_functions', '.strtab_section_info', '0_file_size', '.strtab_section_addr_align', 'symtab_files', '.symtab_section_type', '0_memory_size', 'symtab_weak_symbols', '.strtab_section_link', 'symtab_stt_notype'], ['.comment_section_info', '.strtab_section_type', '.comment_section_size', '.strtab_section_flags', '.symtab_section_flags', 'symtab_weak_objects', 'symtab_weak_functions', '.strtab_section_info', '0_file_size', '.comment_section_link', '.comment_section_addr_align', '.comment_section_type', '.comment_section_flags', '.strtab_section_addr_align', '.symtab_section_type', '0_memory_size', 'symtab_weak_symbols', '.strtab_section_link', 'symtab_stt_notype'], ['1_file_size', '0_file_size', 'symtab_local_functions', 'symtab_weak_objects', 'symtab_stt_notype', '.got_section_addr_align', '.symtab_section_addr_align', '.comment_section_type', '0_memory_size', '.rodata_section_addr_align', '2_segment_type', '.symtab_section_type', '2_memory_size', 'symtab_weak_symbols', '.text_section_link'], ['symtab_global_objects', '.strtab_section_type', '.comment_section_size', '.strtab_section_flags', '1_file_size', 'symtab_sections', '.symtab_section_link', '3_segment_type', '.symtab_section_addr_align', '2_flags', 'symtab_weak_objects', '.got_section_info', '2_segment_type', 'symtab_weak_functions', '0_file_size', '.strtab_section_addr_align', 'symtab_files', '0_memory_size', 'symtab_weak_symbols', '.strtab_section_link', 'symtab_stt_notype'], ['.comment_section_info', '.strtab_section_type', '.comment_section_size', '.strtab_section_flags', '.symtab_section_flags', '.symtab_section_addr_align', '.got_section_type', 'symtab_weak_objects', 'symtab_weak_functions', '0_file_size', '.comment_section_link', '.comment_section_addr_align', '.comment_section_type', '.strtab_section_addr_align', '.symtab_section_type', '.got_section_flags', 'symtab_weak_symbols', '.strtab_section_link', 'symtab_stt_notype'], ['1_file_size', '0_file_size', 'symtab_local_functions', 'symtab_weak_objects', '.got_section_addr_align', '.symtab_section_addr_align', '.comment_section_type', '.strtab_section_addr_align', '.symtab_section_type', '.got_section_flags', '.strtab_section_type', '.symtab_section_info', '0_memory_size', 'symtab_global_symbols', 'symtab_stt_notype'], ['symtab_global_objects', '.strtab_section_type', '.comment_section_size', '1_file_size', 'symtab_sections', '.symtab_section_flags', '.symtab_section_link', '3_segment_type', '.got_section_addr_align', '.symtab_section_addr_align', '.got_section_type', 'symtab_weak_objects', '2_segment_type', 'symtab_weak_functions', '0_file_size', '.strtab_section_addr_align', 'symtab_files', '.symtab_section_type', '0_memory_size', 'symtab_weak_symbols', '.strtab_section_link', 'symtab_stt_notype'], ['.comment_section_info', '.comment_section_size', 'symtab_sections', '.symtab_section_link', '.symtab_section_flags', '2_flags', '.data_section_addr_align', 'symtab_weak_objects', '2_segment_type', 'symtab_weak_functions', '.strtab_section_info', '0_file_size', '.comment_section_addr_align', '.comment_section_type', '.comment_section_flags', '.got_section_flags', '0_memory_size', 'symtab_weak_symbols', '.strtab_section_link', 'symtab_stt_notype'], ['1_file_size', '0_file_size', '.text_section_info', '.comment_section_addr_align', 'symtab_weak_objects', 'symtab_stt_notype', '.text_section_link', '.got_section_addr_align', '.symtab_section_addr_align', '.got_section_link', '0_memory_size', '2_segment_type', '2_memory_size', 'symtab_weak_symbols', '.data_section_link', '.bss_section_link'], ['symtab_global_objects', '.strtab_section_type', '.comment_section_size', '.strtab_section_flags', '1_file_size', 'symtab_sections', '.symtab_section_flags', '.symtab_section_link', '.got_section_addr_align', '.symtab_section_addr_align', 'symtab_weak_objects', '.got_section_info', 'symtab_weak_functions', '0_file_size', '.got_section_link', '.strtab_section_addr_align', 'symtab_files', '.symtab_section_type', '0_memory_size', '.got_section_flags', 'symtab_weak_symbols', '.strtab_section_link', 'symtab_stt_notype'], ['.comment_section_link', '.comment_section_addr_align', 'symtab_stt_notype', '.got_section_type', '.comment_section_type', '.comment_section_info', '.comment_section_flags', '.symtab_section_type', '.strtab_section_type', 'symtab_weak_functions', 'symtab_weak_symbols', '.comment_section_size'], ['1_file_size', '0_file_size', 'symtab_stt_notype', '.text_section_link', '.symtab_section_addr_align', '0_memory_size', '.text_section_type', '2_segment_type', 'symtab_weak_functions', '2_memory_size', '.text_section_flags', 'symtab_weak_symbols', '.strtab_section_flags'], ['symtab_global_objects', '.comment_section_size', '.strtab_section_flags', '1_file_size', 'symtab_sections', '.symtab_section_flags', '.symtab_section_link', '.got_section_addr_align', '.symtab_section_addr_align', '.got_section_type', 'symtab_weak_objects', 'symtab_weak_functions', '.strtab_section_info', '0_file_size', '.strtab_section_addr_align', 'symtab_files', '.symtab_section_type', '0_memory_size', 'symtab_weak_symbols', '.strtab_section_link', 'symtab_stt_notype'], ['symtab_sections', '0_file_size', '2_alignment', '.got_section_link', '.got_section_type', 'entry_point', '0_segment_type', '2_segment_type', '0_memory_size', 'symtab_weak_functions', 'symtab_weak_symbols', '.comment_section_size']]]
    
    column_list_list = []
    
    for train_arch in train_arch_list:
   
        column_list = [] 
         
        for dtanno_file in ['dt'+str(i)+".xlsx" for i in range(0,10)]:
            column_list = anova(dtanno_file, column_list, [train_arch])  
            
       
            
        print(column_list)
           
        inner_counts = [Counter(inner_list) for inner_list in column_list]
        
          # Extract all unique strings across all inner lists
        all_strings = set()
        for inner_list in column_list:
              all_strings.update(inner_list)
         
          # Calculate the percentage of inner lists that include each string
        string_presence = {string: sum(string in inner_count for inner_count in inner_counts) / len(inner_counts) * 100
                            for string in all_strings}
         
          # Convert the result to a DataFrame for plotting
        df_string_presence = pd.DataFrame(list(string_presence.items()), columns=['Feature', 'Percentage'])
           
        df_string_presence_sorted = df_string_presence.sort_values(by='Percentage', ascending=False)
        
        sns.barplot(x='Percentage', y='Feature', data=df_string_presence_sorted, palette='viridis', ax=axes[plot_count])
        axes[plot_count].tick_params(axis='x', labelsize=14)  # Adjust as needed
        axes[plot_count].tick_params(axis='y', labelsize=14)  # Adjust as needed
        
        plot_count += 1
        
    
    # for column_list in column_list_list:
        
    #     inner_counts = [Counter(inner_list) for inner_list in column_list]
        
          
        
    #       # Extract all unique strings across all inner lists
    #     all_strings = set()
    #     for inner_list in column_list:
    #           all_strings.update(inner_list)
         
    #       # Calculate the percentage of inner lists that include each string
    #     string_presence = {string: sum(string in inner_count for inner_count in inner_counts) / len(inner_counts) * 100
    #                         for string in all_strings}
         
    #       # Convert the result to a DataFrame for plotting
    #     df_string_presence = pd.DataFrame(list(string_presence.items()), columns=['Feature', 'Percentage'])
           
    #     df_string_presence_sorted = df_string_presence.sort_values(by='Percentage', ascending=False)
        
    #     # sns.barplot(x='Percentage', y='Feature', data=df_string_presence_sorted, palette='viridis', ax=axes[plot_count])
        
       
    #     sns.barplot(x='Percentage', y='Feature', data=df_string_presence_sorted, palette='coolwarm', color='steelblue', ax=axes[plot_count])
        
    #     for index, bar in enumerate(axes[plot_count].patches):
    #         axes[plot_count].text(
    #             bar.get_width()/3 + 0.1,  # x position (slightly offset from bar end)
    #             bar.get_y() + bar.get_height() / 2,  # y position (middle of bar)
    #             df_string_presence_sorted['Feature'].iloc[index],  # feature name to annotate
    #             ha='left',  # horizontal alignment
    #             va='center',  # vertical alignment
    #             fontsize=10,  # adjust font size as needed
    #             color='black'
                
    #         )       
    #     axes[plot_count].set_yticklabels([])
    #     axes[plot_count].tick_params(axis='x', labelsize=13)  # Adjust as needed
    #     # axes[plot_count].tick_params(axis='y', labelsize=13)  # Adjust as needed
    #     axes[plot_count].set_xlabel('Percentage', fontsize=14)  # Set x-axis label size
    #     axes[plot_count].set_ylabel('Feature', fontsize=14)
    #     plot_count += 1
    
    # # Add labels for architectures to the top of each column
    # fig.text(0.3, 0.02, 'ARM', fontsize=16, ha='center', fontweight='bold')
    # fig.text(0.73, 0.02, 'MIPS', fontsize=16, ha='center', fontweight='bold')
    
    # plt.savefig('../../Figure/ModelAnaly/FDAG/FDAGFeatures25', bbox_inches='tight', dpi=500)    
    # plt.show()
  



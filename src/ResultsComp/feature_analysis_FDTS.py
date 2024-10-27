# -*- coding: utf-8 -*-
import pandas as pd
import itertools
from scipy.stats import f_oneway

import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

import textwrap


def anova(dtanno_file, column_list, train_arch_list):
    # fea_filter = "../../Features/StrRFEDFrank/SamSpa/StrRFEDFrank"+dtanno_file[0:-5]+".xlsx"
    fea_filter = "../../Features/ELFMiner/SamSpa/ELFMiner"+dtanno_file[0:-5]+".xlsx"
    
    # Load the Excel file
    whofol_feas = pd.ExcelFile(fea_filter)
    
    # Get the sheet names    
    year_list = [2020, 2021, 2022]
    print(f'--------------------------{dtanno_file[0:-5]}')
    
    for train_year, train_arch in itertools.product(year_list, train_arch_list):
        
        print(f'----------------{train_year}-{train_arch}')
        
        sheetname = str(train_year)+train_arch
        
        test_fea_df = pd.read_excel(fea_filter, sheet_name=sheetname+"_test") 
        
        same_arch_test_fea_df = test_fea_df[test_fea_df['arch'] == train_arch]
        
        anno_dict = {}
        common_filtered_fea = []     
        
        year_filtered_fea = {}
        
        for test_year in year_list:
            
            print(f'---------{test_year}')
            
            test_year_same_arch_test_fea_df = same_arch_test_fea_df[same_arch_test_fea_df['year'] == test_year]
            
            tested_df = test_year_same_arch_test_fea_df.drop(columns=['location', 'year', 'arch']).reset_index(drop=True)           
            
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
            
            sorted_df = anova_df.sort_values(by=['F-statistic', 'p-value'], ascending=[False, True]).head(40)
            
            filtered_features = sorted_df.index.tolist()
            
            # print(sorted_df)       
            year_filtered_fea[test_year] = filtered_features
            
            if test_year == 2020:
                common_filtered_fea = filtered_features
            else:
                common_filtered_fea = list(set(common_filtered_fea) & set(filtered_features))
                
            columns_to_select = filtered_features + ['label']
            filtered_column_df = tested_df[columns_to_select] 
            anno_dict[test_year] = filtered_column_df
        
        # print('+=+=++=++=+=++=++=+=++=+')
        # print(common_filtered_fea)
        
        # break
        # if dtanno_file[0:-5] == 'dt0' and train_year == 2020: 
        # if dtanno_file[0:-5] == 'dt0' and train_year == 2020: 
            
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(19, 5))
        plot_count = 0
        for test_year in year_list: 
            plot_show_df = anno_dict[test_year][common_filtered_fea + ['label']]
            df_melted = plot_show_df.melt(id_vars='label', var_name='Feature', value_name='Value')
            
            # Create a boxplot using seaborn with hue as label
           
            # ax = sns.boxplot(x='Feature', y='Value', hue='label', data=df_melted, ax=axes[plot_count])
            ax = sns.pointplot(
                data=df_melted, x="Feature", y="Value", hue="label",
                dodge=.4, ax=axes[plot_count],linestyles="none",
                
            )
            
            if train_arch == 'ARM':
                plt.ylim(0, 1000)
            else:
                plt.ylim(0, 1000)
                
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=['Mirai', 'Gafgyt', 'Tsunami'], fontsize=16)
        
            ax.tick_params(axis='x', labelsize=16)  # Adjust as needed
            ax.tick_params(axis='y', labelsize=16)  # Adjust as needed
            ax.set_xlabel('Feature', fontsize=18)  # Set x-axis label size
            ax.set_ylabel('Value', fontsize=18)      
            
            ax.set_ylim(0, 800)
        
            # Wrap text for x-tick labels
            wrapped_labels = [textwrap.fill(label.get_text(), width=7) for label in ax.get_xticklabels()]  # Wrap with a width of your choice
            ax.set_xticklabels(wrapped_labels)  # Set the wrapped labels and rotate
            
            plot_count += 1
        
        # plt.title(f'{dtanno_file[0:-5]} : {train_year}')
        fig.text(0.2, -0.04, '2020', fontsize=18, ha='center', fontweight='bold')
        fig.text(0.52, -0.04, '2021', fontsize=18, ha='center', fontweight='bold') 
        fig.text(0.85, -0.04, '2022', fontsize=18, ha='center', fontweight='bold') 
        
        plt.tight_layout()
        plt.savefig('../../Figure/ModelAnaly/FDTS/' + train_arch + '/25/FDTS' + dtanno_file[0:-5] + str(train_year), bbox_inches='tight', dpi=500)  
        # plt.savefig('../../Figure/ModelAnaly/MIPSFDTS', bbox_inches='tight', dpi=500)  
        plt.show()
            
            # return 
        
        # for test_year in year_list:
            
        #     rest_filtered_fea = list(set(year_filtered_fea[test_year])-set(common_filtered_fea))[:5]
            
        #     plot_show_df = anno_dict[test_year][rest_filtered_fea+['label']]
          
        #     df_melted = plot_show_df.melt(id_vars='label', var_name='Feature', value_name='Value')
            
        #     # Create a boxplot using seaborn with hue as label
        #     plt.figure(figsize=(8, 5))
        #     ax = sns.boxplot(x='Feature', y='Value', hue='label', data=df_melted)
            
        #     if train_arch == 'ARM':
        #         plt.ylim(0, 1500)
        #     else:
        #         plt.ylim(0, 1500)
                
        #     handles, labels = ax.get_legend_handles_labels()
        #     ax.legend(handles=handles, labels=['Mirai', 'Gafgyt', 'Tsunami'])
    
        #     ax.tick_params(axis='x', labelsize=13)  # Adjust as needed
        #     ax.tick_params(axis='y', labelsize=13)  # Adjust as needed
        #     ax.set_xlabel('Feature', fontsize=14)  # Set x-axis label size
        #     ax.set_ylabel('Value', fontsize=14)       
            
        #     for label in ax.get_xticklabels():
        #         label.set_rotation(8)  # Rotate labels by 45 degrees
        #         label.set_ha('right')   # Optional: align labels to the right
            
        #     # plt.title(f'{dtanno_file[0:-5]} : {train_year}')
        #     plt.savefig('../../Figure/ModelAnaly/FDTS/'+train_arch+'/25/FDTS_non'+dtanno_file[0:-5]+str(train_year)+str(test_year), bbox_inches='tight', dpi=500)  
        #     # plt.savefig('../../Figure/ModelAnaly/MIPSFDTS', bbox_inches='tight', dpi=500)  
        #     plt.show()
        
            
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
    # column_list_list = [[['bin36', 'bin16', 'bin13', 'bin32', 'bin28', 'bin34', 'bin30', 'bin15', 'bin27', 'bin31', 'bin40', 'bin12', 'bin26', 'bin29', 'bin19', 'bin10', 'bin33', 'bin39', 'bin11', 'bin14'], ['bin24', 'bin16', 'bin13', 'bin32', 'bin22', 'bin23', 'bin28', 'bin20', 'bin30', 'bin27', 'bin31', 'bin25', 'bin26', 'bin12', 'bin29', 'bin19', 'bin10', 'bin8', 'bin9', 'bin33', 'bin21', 'bin11', 'bin14'], ['bin24', 'bin13', 'bin32', 'bin22', 'bin23', 'bin28', 'bin35', 'bin15', 'bin27', 'bin31', 'bin25', 'bin26', 'bin12', 'bin29', 'bin10', 'bin8', 'bin9', 'bin33', 'bin21', 'bin11', 'bin18', 'bin14'], ['bin36', 'bin16', 'bin13', 'bin32', 'bin28', 'bin34', 'bin30', 'bin15', 'bin27', 'bin31', 'bin40', 'bin12', 'bin26', 'bin29', 'bin19', 'bin10', 'bin33', 'bin39', 'bin11', 'bin14'], ['bin24', 'bin16', 'bin13', 'bin32', 'bin23', 'bin28', 'bin30', 'bin15', 'bin27', 'bin25', 'bin26', 'bin12', 'bin29', 'bin19', 'bin10', 'bin8', 'bin9', 'bin33', 'bin42', 'bin11', 'bin14'], ['bin24', 'bin16', 'bin13', 'bin22', 'bin23', 'bin28', 'bin35', 'bin34', 'bin30', 'bin15', 'bin27', 'bin31', 'bin25', 'bin12', 'bin26', 'bin29', 'bin19', 'bin10', 'bin8', 'bin33', 'bin11', 'bin43', 'bin14'], ['bin16', 'bin13', 'bin28', 'bin17', 'bin35', 'bin30', 'bin15', 'bin27', 'bin31', 'bin12', 'bin26', 'bin25', 'bin38', 'bin29', 'bin19', 'bin9', 'bin39', 'bin42', 'bin11', 'bin18', 'bin14'], ['bin24', 'bin16', 'bin13', 'bin32', 'bin22', 'bin23', 'bin34', 'bin20', 'bin30', 'bin15', 'bin25', 'bin26', 'bin12', 'bin19', 'bin10', 'bin8', 'bin9', 'bin33', 'bin21', 'bin11', 'bin18', 'bin14'], ['bin24', 'bin36', 'bin16', 'bin13', 'bin32', 'bin28', 'bin17', 'bin30', 'bin15', 'bin27', 'bin31', 'bin12', 'bin26', 'bin25', 'bin38', 'bin29', 'bin10', 'bin9', 'bin37', 'bin21', 'bin11', 'bin14'], ['bin41', 'bin36', 'bin16', 'bin13', 'bin32', 'bin28', 'bin17', 'bin30', 'bin15', 'bin27', 'bin31', 'bin40', 'bin12', 'bin26', 'bin29', 'bin19', 'bin10', 'bin39', 'bin11', 'bin14'], ['bin24', 'bin16', 'bin13', 'bin32', 'bin22', 'bin23', 'bin34', 'bin20', 'bin30', 'bin25', 'bin26', 'bin12', 'bin19', 'bin10', 'bin8', 'bin9', 'bin33', 'bin21', 'bin11', 'bin14'], ['bin23', 'bin31', 'bin24', 'bin25', 'bin26', 'bin12', 'bin14', 'bin21', 'bin13', 'bin29', 'bin15', 'bin10', 'bin11', 'bin32', 'bin9', 'bin8', 'bin22', 'bin33'], ['bin36', 'bin16', 'bin13', 'bin32', 'bin28', 'bin17', 'bin34', 'bin30', 'bin15', 'bin27', 'bin31', 'bin40', 'bin12', 'bin26', 'bin29', 'bin19', 'bin10', 'bin33', 'bin39', 'bin11', 'bin14'], ['bin24', 'bin16', 'bin13', 'bin32', 'bin22', 'bin23', 'bin28', 'bin20', 'bin30', 'bin27', 'bin25', 'bin26', 'bin12', 'bin29', 'bin19', 'bin10', 'bin8', 'bin9', 'bin33', 'bin21', 'bin11', 'bin14'], ['bin24', 'bin13', 'bin32', 'bin22', 'bin23', 'bin28', 'bin17', 'bin35', 'bin15', 'bin31', 'bin25', 'bin26', 'bin40', 'bin12', 'bin29', 'bin10', 'bin8', 'bin9', 'bin33', 'bin21', 'bin11', 'bin18', 'bin14']],
    #                     [['bin24', 'bin36', 'bin13', 'bin32', 'bin28', 'bin30', 'bin27', 'bin31', 'bin12', 'bin25', 'bin40', 'bin38', 'bin29', 'bin9', '000000', 'bin39', 'bin42', 'bin11', 'bin14'], ['bin36', 'bin13', '3vo4', 'bin32', 'bin23', 'bin28', 'bin30', 'bin27', 'bin31', 'bin12', 'bin25', 'bin38', 'bin29', 'bin9', 'bin33', 'bin39', 'bin42', 'bin11', 'bin14'], ['bin24', 'bin13', '3vo4', 'bin22', 'bin23', 'bin17', 'bin35', 'bin34', 'bin20', 'bin30', 'bin27', 'bin31', 'bin25', 'bin26', 'bin12', 'bin19', 'bin10', 'bin8', 'bin37', 'bin21', 'bin11', 'bin43'], ['bin27', 'bin28', 'bin31', 'bin40', 'bin12', 'bin26', 'bin25', 'bin13', '3vo4', 'bin38', 'bin30', 'bin15', 'bin29', 'bin10', 'bin11', 'bin43', 'bin32', 'bin14'], ['bin24', 'bin36', 'bin13', '3vo4', 'bin23', 'bin28', 'bin30', 'bin27', 'bin31', 'bin12', 'bin25', 'bin29', 'bin9', 'bin33', 'bin39', 'bin42', 'bin11', 'bin18', 'bin14'], ['bin28', 'bin31', 'bin12', 'bin36', 'bin26', 'bin25', 'bin39', 'bin13', '3vo4', 'bin29', 'bin30', 'bin10', 'bin11', 'bin43', 'bin32', 'bin14', 'bin33'], ['bin24', 'bin13', 'bin32', 'bin22', 'bin28', 'bin35', 'bin30', 'bin27', 'bin31', 'bin12', 'bin25', 'bin38', 'bin29', 'bin9', 'bin33', 'bin39', 'bin42', 'bin11', 'bin14'], ['bin13', 'bin32', 'bin22', 'bin23', 'bin28', 'bin34', 'bin30', 'bin27', 'bin31', 'bin12', 'bin25', 'bin38', 'bin29', 'bin9', 'bin33', 'bin37', 'bin39', 'bin42', 'bin11', 'bin14'], ['bin24', 'bin13', '3vo4', 'bin22', 'bin28', 'bin35', 'bin34', 'bin30', 'bin27', 'bin31', 'bin25', 'bin26', 'bin12', 'bin10', 'bin8', 'bin33', 'bin37', '000000', 'bin21', 'bin11', 'bin43'], ['bin36', 'bin13', 'bin32', 'bin23', 'bin28', 'bin30', 'bin27', 'bin31', 'bin40', 'bin12', 'bin26', 'bin25', 'bin38', 'bin29', 'bin10', '000000', 'bin39', 'bin11', 'bin43', 'bin14'], ['bin24', 'bin36', 'bin13', 'bin22', 'bin23', 'bin28', 'bin30', 'bin27', 'bin31', 'bin12', 'bin26', 'bin25', 'bin40', 'bin38', 'bin29', 'bin9', 'bin33', 'bin39', 'bin42', 'bin11', 'bin14'], ['bin24', 'bin13', 'bin22', 'bin23', 'bin28', 'bin17', 'bin35', 'bin20', 'bin30', 'bin15', 'bin27', 'bin31', 'bin25', 'bin26', 'bin12', 'bin29', 'bin19', 'bin10', 'bin8', 'bin37', 'bin21', 'bin11', 'bin43', 'bin18', 'bin14'], ['bin24', 'bin13', 'bin22', 'bin23', 'bin28', 'bin17', 'bin35', 'bin30', 'bin15', 'bin27', 'bin12', 'bin26', 'bin25', 'bin38', 'bin29', 'bin49', 'bin9', 'bin37', 'bin39', 'bin11', 'bin14'], ['bin16', 'bin13', '3vo4', 'bin22', 'bin23', 'bin28', 'bin17', 'bin30', 'bin15', 'bin27', 'bin12', 'bin26', 'bin25', 'bin29', 'bin9', 'bin37', 'bin39', 'bin42', 'bin11', 'bin14'], ['bin24', '3vo4', 'bin22', 'bin28', 'bin17', 'bin35', 'bin34', 'bin20', 'bin30', 'bin27', 'bin31', 'bin25', 'bin26', 'bin12', 'bin29', 'bin19', 'bin10', 'bin8', 'bin33', 'bin37', '000000', 'bin21', 'bin11']]]
    # # reference 25
    # column_list_list = [[['.data_section_addr_align', 'symtab_files', 'symtab_global_objects', 'symtab_global_symbols'], ['entry_point', 'symtab_global_objects', '.data_section_addr_align'], ['symtab_weak_symbols', '.bss_section_type', '.comment_section_type', 'symtab_weak_functions', '.fini_section_flags', 'got_size', '.got_section_size', '.rodata_section_type', 'entry_point', '.data_section_type', '.comment_section_size', '.fini_section_link', '.comment_section_info', 'symtab_files', '.text_section_size'], ['symtab_files', 'symtab_global_objects', '.data_section_addr_align'], ['symtab_global_objects', '.data_section_addr_align'], ['symtab_weak_symbols', 'symtab_weak_functions', '.comment_section_addr_align', 'got_size', '.got_section_size', '.text_section_size', '.comment_section_link', 'entry_point', '.comment_section_size', '0_file_size', 'symtab_files', '0_memory_size'], ['symtab_global_objects', '.data_section_addr_align'], ['symtab_global_symbols', 'symtab_global_functions', '.data_section_addr_align', 'symtab_global_objects', 'symtab_files'], ['.comment_section_type', 'symtab_weak_functions', 'dynsym_global_symbols', '.text_section_size', 'entry_point', '.comment_section_size', 'symtab_global_objects', '.comment_section_info', '0_file_size', 'symtab_files', '0_memory_size', '.dynamic_section_type'], ['.data_section_addr_align', 'symtab_global_objects', 'symtab_global_symbols'], ['entry_point', '.data_section_addr_align', 'symtab_global_objects', 'symtab_global_symbols'], ['symtab_weak_symbols', '.comment_section_type', 'symtab_weak_functions', 'got_size', '.got_section_size', '.comment_section_link', '.data_section_addr_align', 'entry_point', '.comment_section_size', '.got_section_addr_align', '.comment_section_info', '1_flags', 'symtab_files', '.text_section_size'], ['.data_section_addr_align'], ['symtab_local_objects', 'symtab_global_objects', '.data_section_addr_align'], ['.interp_section_size', 'symtab_weak_symbols', '.dynstr_section_flags', 'symtab_weak_functions', 'got_size', '.got_section_size', '.comment_section_link', 'entry_point', '.comment_section_size', '.dynamic_section_info', 'symtab_global_objects', '.comment_section_info', '.dynstr_section_link', 'symtab_files', '.got_section_type', '.text_section_size']],
    #                     [['.text_section_link', '.bss_section_type', '.reginfo_section_addr_align', '.text_section_type', '.shstrtab_section_type', '.rodata_section_type', '.shstrtab_section_addr_align', '.bss_section_flags', '.reginfo_section_size', '.reginfo_section_flags', '6_alignment', '2_alignment', '.got_section_type'], ['symtab_stt_notype', 'symtab_weak_symbols', '.symtab_section_type', '.strtab_section_type', '.strtab_section_info', 'symtab_weak_functions', 'symtab_weak_objects', '.strtab_section_addr_align', '.strtab_section_flags', '2_segment_type', '0_memory_size', '.strtab_section_link', '.comment_section_size', '.symtab_section_addr_align', '0_file_size', '2_alignment', '.symtab_section_flags'], ['.strtab_section_info', 'symtab_weak_functions', 'symtab_weak_objects', '.strtab_section_addr_align', '0_file_size', '2_alignment', '.symtab_section_flags', 'symtab_weak_symbols', '.comment_section_type', '0_segment_type', '.comment_section_link', '.comment_section_size', '.reginfo_section_flags', '.comment_section_info', '.symtab_section_type', '.strtab_section_flags', '.reginfo_section_addr_align', '0_alignment', '.comment_section_addr_align', '.reginfo_section_link', 'symtab_stt_notype', '.comment_section_flags', '.strtab_section_type', '.reginfo_section_type', '.strtab_section_link'], ['symtab_stt_notype', 'symtab_weak_symbols', '.text_section_link', '.symtab_section_type', 'symtab_weak_objects', '.reginfo_section_addr_align', '0_memory_size', '2_segment_type', '.text_section_flags', '.text_section_addr_align', '.sbss_section_size', '.reginfo_section_flags', '.got_section_addr_align', '.symtab_section_addr_align', '0_file_size', '2_alignment', '.bss_section_addr_align'], ['symtab_stt_notype', 'symtab_weak_symbols', '.strtab_section_type', 'symtab_weak_objects', 'symtab_weak_functions', '.reginfo_section_addr_align', '.strtab_section_addr_align', '.strtab_section_flags', '2_segment_type', '.symtab_section_link', 'symtab_sections', '5_alignment', '.strtab_section_link', '.comment_section_size', '.symtab_section_addr_align', '0_file_size', '2_alignment', '0_memory_size'], ['symtab_weak_objects', 'symtab_weak_functions', '.strtab_section_addr_align', '0_file_size', '2_alignment', '.symtab_section_flags', 'symtab_weak_symbols', '2_segment_type', '.comment_section_size', '.reginfo_section_flags', '.symtab_section_type', '.strtab_section_flags', '.symtab_section_addr_align', '.reginfo_section_addr_align', '.reginfo_section_size', '0_memory_size', '.strtab_section_type', '.reginfo_section_type', '.strtab_section_link'], ['symtab_weak_objects', '.strtab_section_addr_align', '7_alignment', '0_file_size', '2_alignment', '.text_section_link', '2_segment_type', '.bss_section_flags', '.text_section_addr_align', '.symtab_section_type', '.symtab_section_addr_align', '.bss_section_addr_align', '.bss_section_type', '.reginfo_section_addr_align', '0_memory_size', 'symtab_stt_notype', '.strtab_section_type', '.reginfo_section_type', '.text_section_type', '5_alignment', '.got_section_addr_align', '8_alignment'], ['symtab_stt_notype', 'symtab_weak_symbols', '.symtab_section_type', '.strtab_section_type', 'symtab_weak_objects', 'symtab_weak_functions', '0_alignment', '.strtab_section_addr_align', '0_memory_size', '2_segment_type', '.strtab_section_link', '.comment_section_size', '.symtab_section_addr_align', '0_file_size', '2_alignment', '.symtab_section_flags'], ['.strtab_section_info', 'symtab_weak_functions', 'symtab_weak_objects', '0_file_size', '2_alignment', '.symtab_section_flags', '2_memory_size', 'symtab_weak_symbols', '2_segment_type', '.symtab_section_link', '0_segment_type', '.comment_section_size', '.reginfo_section_flags', '0_alignment', 'symtab_sections', '8_memory_size', '5_memory_size', '0_memory_size', 'symtab_stt_notype', '.reginfo_section_type', '.strtab_section_link'], ['symtab_weak_objects', '.text_section_info', '.shstrtab_section_link', '.got_section_link', '7_alignment', '0_file_size', '.bss_section_link', '2_alignment', 'symtab_weak_symbols', '.text_section_link', '2_segment_type', '.text_section_addr_align', '6_alignment', '.symtab_section_addr_align', '.bss_section_addr_align', '.sbss_section_size', '0_memory_size', 'symtab_stt_notype', '.rodata_section_link', '.reginfo_section_type', '.text_section_flags', '.shstrtab_section_addr_align', '5_alignment', '.shstrtab_section_info', '.got_section_addr_align', '8_alignment'], ['symtab_weak_functions', 'symtab_weak_objects', '.strtab_section_addr_align', '0_file_size', '2_alignment', '.symtab_section_flags', 'symtab_weak_symbols', '2_segment_type', '.symtab_section_link', '.comment_section_size', '6_alignment', '.symtab_section_type', '.strtab_section_flags', '.symtab_section_addr_align', 'symtab_sections', '.sbss_section_size', '0_memory_size', 'symtab_stt_notype', '.strtab_section_type', '.strtab_section_link'], ['symtab_stt_notype', 'symtab_weak_symbols', '.symtab_section_type', '.strtab_section_type', 'symtab_weak_functions', '.reginfo_section_info', '2_segment_type', '0_segment_type', '.reginfo_section_link', '.reginfo_section_size', '.comment_section_size', '.reginfo_section_flags', '0_file_size', '2_alignment', '0_memory_size', '2_memory_size'], ['.rodata_section_info', '.text_section_link', '.bss_section_type', '.rodata_section_link', '.text_section_type', '7_memory_size', '.text_section_info', '2_segment_type', '.text_section_flags', '.bss_section_flags', '.shstrtab_section_info', '6_memory_size', '.shstrtab_section_flags', '0_file_size', '2_alignment', '0_memory_size'], ['7_memory_size', '8_memory_size', '6_memory_size', '5_memory_size', '0_file_size', '2_alignment', '0_memory_size', '2_memory_size'], ['symtab_weak_symbols', 'symtab_weak_functions', '8_alignment', 'symtab_sections', '0_memory_size', '2_segment_type', '0_segment_type', '5_alignment', '.comment_section_size', '6_alignment', '7_alignment', '0_file_size', '2_alignment', '.bss_section_addr_align']]]
    
    column_list_list = []
    
    for train_arch in train_arch_list:
   
        column_list = [] 
         
        for dtanno_file in ['dt'+str(i)+".xlsx" for i in range(0,10)]:
            column_list = anova(dtanno_file, column_list, [train_arch])  
            
           
        
        break
            
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
    
    # Add labels for architectures to the top of each column
    # fig.text(0.3, 0.02, 'ARM', fontsize=16, ha='center', fontweight='bold')
    # fig.text(0.73, 0.02, 'MIPS', fontsize=16, ha='center', fontweight='bold')
    
    # plt.savefig('../../Figure/ModelAnaly/FDTSFeatures25', bbox_inches='tight', dpi=500)    
    # plt.show()
  



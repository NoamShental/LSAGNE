import pandas as pd
import functools

# filestomerge=[
#                 ["_Triangle_R_0", "E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_0_50_HCC515 lung carcinoma_vorinostat_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
#                 ["_Alpha_T_0","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/DUDI_drug_combinations_Trial0-A549 lung non small cell lung cancer carcinoma_wortmannin_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
#                 ["_Alpha_R_0","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/DUDI_SR_drug_combinations_Trial0-A549 lung non small cell lung cancer carcinoma_wortmannin_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
#                 ["_Alpha_R_1","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/DUDI_drug_combinations_Trial1-A549 lung non small cell lung cancer carcinoma_wortmannin_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"]
#     ]
filestomerge=[
                ["_Alpha_43R_0","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/DUDI_SR_drug_combinations_Trial0-A549 lung non small cell lung cancer carcinoma_wortmannin_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Alpha_43R_1","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/DUDI_drug_combinations_Trial1-A549 lung non small cell lung cancer carcinoma_wortmannin_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Alpha_43R_2","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/DUDI_drug_combinations_Trial2-A549 lung non small cell lung cancer carcinoma_wortmannin_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Alpha_43R_3","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/DUDI_drug_combinations_Trial3-A549 lung non small cell lung cancer carcinoma_wortmannin_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Alpha_43R_4","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/DUDI_drug_combinations_Trial4-A549 lung non small cell lung cancer carcinoma_wortmannin_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Alpha_43R_5","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/DUDI_drug_combinations_Trial5-A549 lung non small cell lung cancer carcinoma_wortmannin_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Alpha_43R_6","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/DUDI_drug_combinations_Trial6-A549 lung non small cell lung cancer carcinoma_wortmannin_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Alpha_43R_7","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/DUDI_drug_combinations_Trial7-A549 lung non small cell lung cancer carcinoma_wortmannin_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Alpha_43R_8","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/DUDI_drug_combinations_Trial8-A549 lung non small cell lung cancer carcinoma_wortmannin_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Alpha_43R_9","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/DUDI_drug_combinations_Trial9-A549 lung non small cell lung cancer carcinoma_wortmannin_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Alpha_138R_D_0","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial0-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Alpha_13R_D_1","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial1-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Alpha_13R_D_2","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial2-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Alpha_13R_D_3","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial3-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Alpha_13R_D_4","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial4-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Alpha_13R_D_5","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial5-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Alpha_13R_D_6","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial6-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Alpha_13R_D_7","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial7-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Alpha_13R_D_8","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial8-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Alpha_13R_D_9","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial9-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Alpha_T_0","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/DUDI_drug_combinations_Trial0-A549 lung non small cell lung cancer carcinoma_wortmannin_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Alpha_T_1","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/DUDI_drug_combinations_Trial1-A549 lung non small cell lung cancer carcinoma_wortmannin_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Alpha_T_2","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/DUDI_drug_combinations_Trial2-A549 lung non small cell lung cancer carcinoma_wortmannin_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Alpha_T_3","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/DUDI_drug_combinations_Trial3-A549 lung non small cell lung cancer carcinoma_wortmannin_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Alpha_T_4","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/DUDI_drug_combinations_Trial4-A549 lung non small cell lung cancer carcinoma_wortmannin_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Alpha_T_5","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/DUDI_drug_combinations_Trial5-A549 lung non small cell lung cancer carcinoma_wortmannin_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Alpha_T_6","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/DUDI_drug_combinations_Trial6-A549 lung non small cell lung cancer carcinoma_wortmannin_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Alpha_T_7","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/DUDI_drug_combinations_Trial7-A549 lung non small cell lung cancer carcinoma_wortmannin_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Alpha_T_8","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/DUDI_drug_combinations_Trial8-A549 lung non small cell lung cancer carcinoma_wortmannin_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Alpha_T_9","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/DUDI_drug_combinations_Trial9-A549 lung non small cell lung cancer carcinoma_wortmannin_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Alpha_183T_D_0","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial0-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Alpha_183T_D_1","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial1-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Alpha_183T_D_2","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial2-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Alpha_183T_D_3","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial3-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Alpha_183T_D_4","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial4-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Alpha_183T_D_5","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial5-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Alpha_183T_D_6","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial6-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Alpha_183T_D_7","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial7-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Alpha_183T_D_8","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial8-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Alpha_183T_D_9","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial9-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Alpha_183R_A_0","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial0-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Alpha_183R_A_1","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial1-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Alpha_183R_A_2","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial2-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Alpha_183R_A_3","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial3-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Alpha_183R_A_4","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial4-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Alpha_183R_A_5","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial5-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Alpha_183R_A_6","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial6-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Alpha_183R_A_7","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial7-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Alpha_183R_A_8","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial8-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Alpha_183R_A_9","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial9-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Alpha_183T_A_0","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial0-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Alpha_183T_A_1","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial1-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Alpha_183T_A_2","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial2-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Alpha_183T_A_3","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial3-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Alpha_183T_A_4","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial4-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Alpha_183T_A_5","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial5-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Alpha_183T_A_6","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial6-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Alpha_183T_A_7","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial7-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Alpha_183T_A_8","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial8-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Alpha_183T_A_9","E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial9-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Triangle_R_0","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_0_50_HCC515 lung carcinoma_vorinostat_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Triangle_R_1","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_1_50_HCC515 lung carcinoma_vorinostat_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Triangle_R_2","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_2_50_HCC515 lung carcinoma_vorinostat_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Triangle_R_3","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_3_50_HCC515 lung carcinoma_vorinostat_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Triangle_R_4","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_4_50_HCC515 lung carcinoma_vorinostat_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Triangle_R_5","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_5_50_HCC515 lung carcinoma_vorinostat_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Triangle_R_6","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_6_50_HCC515 lung carcinoma_vorinostat_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Triangle_R_7","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_7_50_HCC515 lung carcinoma_vorinostat_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Triangle_R_8","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_8_50_HCC515 lung carcinoma_vorinostat_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Triangle_R_9","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_9_50_HCC515 lung carcinoma_vorinostat_43Pathways_deltaDMSO_livedead_PREDICTED_v1.txt"],
                ["_Triangle_183R_D_0","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_0_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Triangle_183R_D_1","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_1_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Triangle_183R_D_2","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_2_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Triangle_183R_D_3","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_3_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Triangle_183R_D_4","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_4_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Triangle_183R_D_5","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_5_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Triangle_183R_D_6","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_6_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Triangle_183R_D_7","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_7_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Triangle_183R_D_8","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_8_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Triangle_183R_D_9","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_9_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_PREDICTED_DEAD_v1.txt"],
                ["_Triangle_T_0","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/to_Michael/death_prediction_combination_0_58_HCC515 lung carcinoma_wortmannin.hdf"],
                ["_Triangle_T_1","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/to_Michael/death_prediction_combination_1_58_HCC515 lung carcinoma_wortmannin.hdf"],
                ["_Triangle_T_2","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/to_Michael/death_prediction_combination_2_58_HCC515 lung carcinoma_wortmannin.hdf"],
                ["_Triangle_T_3","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/to_Michael/death_prediction_combination_3_58_HCC515 lung carcinoma_wortmannin.hdf"],
                ["_Triangle_T_4","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/to_Michael/death_prediction_combination_4_58_HCC515 lung carcinoma_wortmannin.hdf"],
                ["_Triangle_T_5","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/to_Michael/death_prediction_combination_5_58_HCC515 lung carcinoma_wortmannin.hdf"],
                ["_Triangle_T_6","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/to_Michael/death_prediction_combination_6_58_HCC515 lung carcinoma_wortmannin.hdf"],
                ["_Triangle_T_7","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/to_Michael/death_prediction_combination_7_58_HCC515 lung carcinoma_wortmannin.hdf"],
                ["_Triangle_T_8","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/to_Michael/death_prediction_combination_8_58_HCC515 lung carcinoma_wortmannin.hdf"],
                ["_Triangle_T_9","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/to_Michael/death_prediction_combination_9_58_HCC515 lung carcinoma_wortmannin.hdf"],
                ["_Triangle_183R_A_0","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_0_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Triangle_183R_A_1","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_1_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Triangle_183R_A_2","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_2_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Triangle_183R_A_3","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_3_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Triangle_183R_A_4","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_4_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Triangle_183R_A_5","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_5_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Triangle_183R_A_6","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_6_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Triangle_183R_A_7","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_7_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Triangle_183R_A_8","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_8_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_Triangle_183R_A_9","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_9_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_PREDICTED_ALIVE_v1.txt"],
                ["_TriangleSphere_0","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/to_Michael/death_prediction_sphere_0_58_HCC515 lung carcinoma_wortmannin.hdf"],
                ["_TriangleSphere_1","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/to_Michael/death_prediction_sphere_1_58_HCC515 lung carcinoma_wortmannin.hdf"],
                ["_TriangleSphere_2","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/to_Michael/death_prediction_sphere_2_58_HCC515 lung carcinoma_wortmannin.hdf"],
                ["_TriangleSphere_3","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/to_Michael/death_prediction_sphere_3_58_HCC515 lung carcinoma_wortmannin.hdf"],
                ["_TriangleSphere_4","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/to_Michael/death_prediction_sphere_4_58_HCC515 lung carcinoma_wortmannin.hdf"],
                ["_TriangleSphere_5","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/to_Michael/death_prediction_sphere_5_58_HCC515 lung carcinoma_wortmannin.hdf"],
                ["_TriangleSphere_6","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/to_Michael/death_prediction_sphere_6_58_HCC515 lung carcinoma_wortmannin.hdf"],
                ["_TriangleSphere_7","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/to_Michael/death_prediction_sphere_7_58_HCC515 lung carcinoma_wortmannin.hdf"],
                ["_TriangleSphere_8","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/to_Michael/death_prediction_sphere_8_58_HCC515 lung carcinoma_wortmannin.hdf"],
                ["_TriangleSphere_9","E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/to_Michael/death_prediction_sphere_9_58_HCC515 lung carcinoma_wortmannin.hdf"]
                ]

def Dcompare(item1, item2):
    if item1 == '-':
        if item2 == '-':
            return 0
        else:
            return 1
    else:
        if item2 == '-':
            return -1
        else:
            if item1 < item2:
                return -1
            elif item1 > item2:
                return 1
            else:
                return 0


print("Reading combination predictions...")
totdata=[]
for i in range(len(filestomerge)):
    print(filestomerge[i][1])
    if ('.txt' in filestomerge[i][1]):
        data = pd.read_csv(filestomerge[i][1],sep="\t", index_col=0)
        data = data.iloc[:data.shape[0] - 1]
        data["drug1"] = [i.strip('(').strip(')').replace('\'', '').replace(', ', ',').split(',')[1] for i in data.index]
        data = data[["non-viability", "drug1"]]
        data["drug2"] = [i.strip('(').strip(')').replace('\'', '').replace(', ', ',').split(',')[2] for i in data.index]
        data["drug3"] = [i.strip('(').strip(')').replace('\'', '').replace(', ', ',').split(',')[3] for i in data.index]
        data["cell_id"] = [i.strip('(').strip(')').replace('\'', '').replace(', ', ',').split(',')[0].split(' ')[0] for
                           i in data.index]
        if 'encoded' in data.index[0].strip('(').strip(')').replace('\'', '').replace(', ', ',').split(',')[-1]:
            data.index = [i.strip('(').strip(')').replace('\'', '').replace(', ', ',').split(',')[-1].split('encoded_')[1]
                          for i in data.index]
        else:
            data.index = [i.strip('(').strip(')').replace('\'', '').replace(', ', ',').split(',')[-1] for i in data.index]

    elif('.hdf' in filestomerge[i][1]):
        data = pd.read_hdf(filestomerge[i][1])
        data["cell_id"]=[i[0].split(' ')[0] for i in data.index]
        data["drug1"]=[i[1] for i in data.index]
        data["drug2"]=[i[2] for i in data.index]
        data["drug3"]=[i[3] for i in data.index]
        data.index=[i[4] for i in data.index]
        a=list(data.columns)
        a[0]="non-viability"
        data.columns=a
    else:
        print("Unknown format")
        exit(0)
    data=data[["cell_id", "drug1", "drug2", "drug3", "non-viability"]]
    data["non-viability"]=[float(i) for i in data["non-viability"]]
    data_grp=pd.DataFrame({'MeanDeath':data.groupby(["cell_id", "drug1", "drug2", "drug3"])['non-viability'].mean()}).reset_index()
    data_grp["Death_AvgInOtherCelllines"]=[-1 for i in data_grp.index]
    for index1, row1 in data_grp.iterrows():
        data_t=data_grp[(data_grp["drug1"] == row1["drug1"]) & (data_grp["drug2"] == row1["drug2"]) & (data_grp["drug3"] == row1["drug3"])]
        data_t=data_t[data_t.index != index1]
        data_grp.loc[index1, "Death_AvgInOtherCelllines"]=data_t["MeanDeath"].mean()
        tto = sorted([row1["drug1"], row1["drug2"], row1["drug3"]], key=functools.cmp_to_key(Dcompare))
        data_grp.loc[index1,"drug1"]=tto[0];data_grp.loc[index1,"drug2"]=tto[1];data_grp.loc[index1,"drug3"]=tto[2];

    data_grp["Death_delta"]= data_grp["MeanDeath"] - data_grp["Death_AvgInOtherCelllines"]
    totdata.append(data_grp.copy(deep=True))
print("Done...")

print("Merging all predictions")
print(filestomerge[0][1])
data_grp = totdata[0]
data_grp = data_grp[data_grp["cell_id"].isin(["A375", "MCF7", "HEPG2"])]
a = list(data_grp.columns)
a[4] += filestomerge[0][0];a[5] += filestomerge[0][0];a[6] += filestomerge[0][0];
data_grp.columns = a
alldata = data_grp
for i in range(1,len(filestomerge)):
    print(filestomerge[i][1])
    data_grp = totdata[i].copy(deep=True)
    data_grp=data_grp[data_grp["cell_id"].isin(["A375", "MCF7", "HEPG2"])]
    a = list(data_grp.columns);a[4] += filestomerge[i][0];a[5] += filestomerge[i][0];a[6] += filestomerge[i][0];
    data_grp.columns = a
    alldata=pd.merge(alldata, data_grp, on=["cell_id", "drug1", "drug2", "drug3"], how="outer", suffixes=('', ''))

a = list(alldata.columns)
mdid=[i for i in a if 'MeanDeath' in i]; daid=[i for i in a if 'Death_AvgInOtherCelllines' in i]; delid=[i for i in a if 'Death_delta' in i]

#calc means
sumcols=["MeanDeath_Alpha_R","STD_Alpha_R","MeanDeath_Alpha_183R_D","STD_Alpha_183R_D",
         "MeanDeath_Alpha_T","STD_Alpha_T","MeanDeath_Alpha_183T_D","STD_Alpha_183T_D",
         "MeanDeath_Alpha_183R_A","STD_Alpha_183R_A","MeanDeath_Alpha_183T_A","STD_Alpha_183T_A",
         "MeanDeath_Triangle_R","STD_Triangle_R","MeanDeath_Triangle_183R_D","STD_Triangle_183R_D",
         "MeanDeath_Triangle_T","STD_Triangle_T","MeanDeath_Triangle_183R_A","STD_Triangle_183R_A",
         "MeanDeath_TriangleSphere","STD_TriangleSphere"]
for i in range(0,len(sumcols),2):
    avid1=[j for j in a if sumcols[i] in j]
    alldata[sumcols[i]]=alldata[avid1].mean(axis=1).fillna(100)
    alldata[sumcols[i+1]]=alldata[avid1].std(axis=1).fillna(-0.01)

b=a[:4]; b.extend(sumcols);b.extend(mdid);b.extend(daid);b.extend(delid);
alldata=alldata[b]
alldata.to_csv("E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/NovelDrugComboPredictions_Alpha_Triangle_Spheres_Live_Dead_v52.txt",sep="\t")


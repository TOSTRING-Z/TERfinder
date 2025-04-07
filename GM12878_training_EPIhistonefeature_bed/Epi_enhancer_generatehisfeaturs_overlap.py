#Enhancer_Extended_f->以增强子延伸，统一长度为1000时:_1000
#GM12878
#生成所有GM12878_training_Dnasepairs的 Histone_feature_matrix（enhancer_Extended）
import pandas as pd
# #to generate Enhancer_Extended_his_feature
#
# #H2AFZ
GM12878_train_Ehancer_Extended_H2AFZ_feature=pd.read_excel("Enhancer_extended_f\GM12878_Enhancer_extended_Overlap_Histonefeatures_1000\Overlap_GM12878_train_Enhancer_extended_H2AFZ.xls",header=None,names=["chr","Enhancer_start","Enhancer_end","pair_rows"])
print(GM12878_train_Ehancer_Extended_H2AFZ_feature.head())

#取EPIs pair的 pair_rows
EPI_pairrows_H2AFZ_feature=GM12878_train_Ehancer_Extended_H2AFZ_feature["pair_rows"]
#<class 'pandas.core.series.Series'>
#print(type(EPI_pairrows))

EPI_pairrows_H2AFZ_feature_list=EPI_pairrows_H2AFZ_feature.tolist()
#print(type(EPI_pairrows_H2AFZ_feature_list))
#print(EPI_pairrows_H2AFZ_feature_list)
#
#列表去重
#不是原位排序
EPI_pairrows_H2AFZ_feature_list=list(set(EPI_pairrows_H2AFZ_feature_list))
#print(EPI_pairrows_H2AFZ_feature_list)
#一共有0到5347个EPI_pairs,0 or 1 his_feature of enhancer_Extended of per pair->Enhancer_Extended_H2AFZ_feature
Enhancer_Extended_H2AFZ_feature=[]
for i in range(5348):
    if i in EPI_pairrows_H2AFZ_feature_list:
        Enhancer_Extended_H2AFZ_feature.append(1)
    else:
        Enhancer_Extended_H2AFZ_feature.append(0)
#finished generating Enhancer_Extended_H2AFZ_feature
#print(len(Enhancer_Extended_H2AFZ_feature))
print(Enhancer_Extended_H2AFZ_feature)
#
# #h3k4me1
GM12878_train_Ehancer_Extended_h3k4me1_feature=pd.read_excel("Enhancer_extended_f\GM12878_Enhancer_extended_Overlap_Histonefeatures_1000\Overlap_GM12878_train_Enhancer_extended_H3K4me1.xls",header=None,names=["chr","Enhancer_start","Enhancer_end","pair_rows"])
#取EPIs pair的 pair_rows
EPI_pairrows_h3k4me1_feature=GM12878_train_Ehancer_Extended_h3k4me1_feature["pair_rows"]
EPI_pairrows_h3k4me1_feature_list=EPI_pairrows_h3k4me1_feature.tolist()
#列表去重
#不是原位排序
EPI_pairrows_h3k4me1_feature_list=list(set(EPI_pairrows_h3k4me1_feature_list))
#一共有0到5347个EPI_pairs,0 or 1 his_feature of enhancer of per pair->Enhancer_h3k4me1_feature
Enhancer_Extended_h3k4me1_feature=[]
for i in range(5348):
    if i in EPI_pairrows_h3k4me1_feature_list:
        Enhancer_Extended_h3k4me1_feature.append(1)
    else:
        Enhancer_Extended_h3k4me1_feature.append(0)
print(Enhancer_Extended_h3k4me1_feature)
#
# #GM12878_train_Ehancer_Extended_h3k4me2
GM12878_train_Ehancer_Extended_h3k4me2_feature=pd.read_excel("Enhancer_extended_f\GM12878_Enhancer_extended_Overlap_Histonefeatures_1000\Overlap_GM12878_train_Enhancer_extended_H3K4me2.xls",header=None,names=["chr","Enhancer_start","Enhancer_end","pair_rows"])
#取EPIs pair的 pair_rows
EPI_pairrows_h3k4me2_feature=GM12878_train_Ehancer_Extended_h3k4me2_feature["pair_rows"]
EPI_pairrows_h3k4me2_feature_list=EPI_pairrows_h3k4me2_feature.tolist()
#列表去重
#不是原位排序
EPI_pairrows_h3k4me2_feature_list=list(set(EPI_pairrows_h3k4me2_feature_list))
#一共有0到5347个EPI_pairs,0 or 1 his_feature of enhancer_Extended of per pair->Enhancer_Extended_h3k4me2_feature
Enhancer_Extended_h3k4me2_feature=[]
for i in range(5348):
    if i in EPI_pairrows_h3k4me2_feature_list:
        Enhancer_Extended_h3k4me2_feature.append(1)
    else:
        Enhancer_Extended_h3k4me2_feature.append(0)
print(Enhancer_Extended_h3k4me2_feature)
#
#GM12878_train_Ehancer_Extended_h3k4me3
GM12878_train_Ehancer_Extended_h3k4me3_feature=pd.read_excel("Enhancer_extended_f\GM12878_Enhancer_extended_Overlap_Histonefeatures_1000\Overlap_GM12878_train_Enhancer_extended_H3K4me3.xls",header=None,names=["chr","Enhancer_start","Enhancer_end","pair_rows"])
EPI_pairrows_h3k4me3_feature=GM12878_train_Ehancer_Extended_h3k4me3_feature["pair_rows"]
EPI_pairrows_h3k4me3_feature_list=EPI_pairrows_h3k4me3_feature.tolist()
#列表去重
#不是原位排序
EPI_pairrows_h3k4me3_feature_list=list(set(EPI_pairrows_h3k4me3_feature_list))
#一共有0到5347个EPI_pairs,0 or 1 his_feature of enhancer_Extended of per pair->Enhancer_Extended_h3k4me3_feature
Enhancer_Extended_h3k4me3_feature=[]
for i in range(5348):
    if i in EPI_pairrows_h3k4me3_feature_list:
        Enhancer_Extended_h3k4me3_feature.append(1)
    else:
        Enhancer_Extended_h3k4me3_feature.append(0)
print(Enhancer_Extended_h3k4me3_feature)
#
#GM12878_train_Ehancer_Extended_h3k9ac
GM12878_train_Ehancer_Extended_h3k9ac_feature=pd.read_excel("Enhancer_extended_f\GM12878_Enhancer_extended_Overlap_Histonefeatures_1000\Overlap_GM12878_train_Enhancer_extended_H3K9ac.xls",header=None,names=["chr","Enhancer_start","Enhancer_end","pair_rows"])
EPI_pairrows_h3k9ac_feature=GM12878_train_Ehancer_Extended_h3k9ac_feature["pair_rows"]
EPI_pairrows_h3k9ac_feature_list=EPI_pairrows_h3k9ac_feature.tolist()
#列表去重
#不是原位排序
EPI_pairrows_h3k9ac_feature_list=list(set(EPI_pairrows_h3k9ac_feature_list))
#一共有0到5347个EPI_pairs,0 or 1 his_feature of enhancer_Extended of per pair->Enhancer_Extended_h3k9ac_feature
Enhancer_Extended_h3k9ac_feature=[]
for i in range(5348):
    if i in EPI_pairrows_h3k9ac_feature_list:
        Enhancer_Extended_h3k9ac_feature.append(1)
    else:
        Enhancer_Extended_h3k9ac_feature.append(0)
print(Enhancer_Extended_h3k9ac_feature)
#
# #K562_train_Ehancer_Extended_h3k9me1
# K562_train_Ehancer_Extended_h3k9me1_feature=pd.read_excel("Enhancer_Extended_f\K562_train_Ehancer_Extended_h3k9me1.xls",header=None,names=["chr","Enhancer_start","Enhancer_end","pair_rows"])
# EPI_pairrows_h3k9me1_feature=K562_train_Ehancer_Extended_h3k9me1_feature["pair_rows"]
# EPI_pairrows_h3k9me1_feature_list=EPI_pairrows_h3k9me1_feature.tolist()
# #列表去重
# #不是原位排序
# EPI_pairrows_h3k9me1_feature_list=list(set(EPI_pairrows_h3k9me1_feature_list))
# #一共有0到8649个EPI_pairs,0 or 1 his_feature of enhancer of per pair->Enhancer_h3k9me1_feature
# Enhancer_Extended_h3k9me1_feature=[]
# for i in range(8650):
#     if i in EPI_pairrows_h3k9me1_feature_list:
#         Enhancer_Extended_h3k9me1_feature.append(1)
#     else:
#         Enhancer_Extended_h3k9me1_feature.append(0)
# print(Enhancer_Extended_h3k9me1_feature)
#
#K562_train_Ehancer_Extended_h3k9me3
GM12878_train_Ehancer_Extended_h3k9me3_feature=pd.read_excel("Enhancer_extended_f\GM12878_Enhancer_extended_Overlap_Histonefeatures_1000\Overlap_GM12878_train_Enhancer_extended_H3K9me3.xls",header=None,names=["chr","Enhancer_start","Enhancer_end","pair_rows"])
EPI_pairrows_h3k9me3_feature=GM12878_train_Ehancer_Extended_h3k9me3_feature["pair_rows"]
EPI_pairrows_h3k9me3_feature_list=EPI_pairrows_h3k9me3_feature.tolist()
#列表去重
#不是原位排序
EPI_pairrows_h3k9me3_feature_list=list(set(EPI_pairrows_h3k9me3_feature_list))
#一共有0到5347个EPI_pairs,0 or 1 his_feature of enhancer of per pair->Enhancer_h3k9me3_feature
Enhancer_Extended_h3k9me3_feature=[]
for i in range(5348):
    if i in EPI_pairrows_h3k9me3_feature_list:
        Enhancer_Extended_h3k9me3_feature.append(1)
    else:
        Enhancer_Extended_h3k9me3_feature.append(0)
print(Enhancer_Extended_h3k9me3_feature)
#
# #GM12878_train_Ehancer_Extended_h3k27ac
GM12878_train_Ehancer_Extended_h3k27ac_feature=pd.read_excel("Enhancer_extended_f\GM12878_Enhancer_extended_Overlap_Histonefeatures_1000\Overlap_GM12878_train_Enhancer_extended_H3K27ac.xls",header=None,names=["chr","Enhancer_start","Enhancer_end","pair_rows"])
EPI_pairrows_h3k27ac_feature=GM12878_train_Ehancer_Extended_h3k27ac_feature["pair_rows"]
EPI_pairrows_h3k27ac_feature_list=EPI_pairrows_h3k27ac_feature.tolist()
#列表去重
#不是原位排序
EPI_pairrows_h3k27ac_feature_list=list(set(EPI_pairrows_h3k27ac_feature_list))
#一共有0到5347个EPI_pairs,0 or 1 his_feature of enhancer_Extended of per pair->Enhancer_Extended_h3k27ac_feature
Enhancer_Extended_h3k27ac_feature=[]
for i in range(5348):
    if i in EPI_pairrows_h3k27ac_feature_list:
        Enhancer_Extended_h3k27ac_feature.append(1)
    else:
        Enhancer_Extended_h3k27ac_feature.append(0)
print(Enhancer_Extended_h3k27ac_feature)

#GM12878_train_Ehancer_Extended_h3k27me3
GM12878_train_Ehancer_Extended_h3k27me3_feature=pd.read_excel("Enhancer_extended_f\GM12878_Enhancer_extended_Overlap_Histonefeatures_1000\Overlap_GM12878_train_Enhancer_extended_H3K27me3.xls",header=None,names=["chr","Enhancer_start","Enhancer_end","pair_rows"])
EPI_pairrows_h3k27me3_feature=GM12878_train_Ehancer_Extended_h3k27me3_feature["pair_rows"]
EPI_pairrows_h3k27me3_feature_list=EPI_pairrows_h3k27me3_feature.tolist()
#列表去重
#不是原位排序
EPI_pairrows_h3k27me3_feature_list=list(set(EPI_pairrows_h3k27me3_feature_list))
#一共有0到5347个EPI_pairs,0 or 1 his_feature of enhancer of per pair->Enhancer_h3k27me3_feature
Enhancer_Extended_h3k27me3_feature=[]
for i in range(5348):
    if i in EPI_pairrows_h3k27me3_feature_list:
        Enhancer_Extended_h3k27me3_feature.append(1)
    else:
        Enhancer_Extended_h3k27me3_feature.append(0)
print(Enhancer_Extended_h3k27me3_feature)

#GM12878_train_Ehancer_Extended_h3k36me3
GM12878_train_Ehancer_Extended_h3k36me3_feature=pd.read_excel("Enhancer_extended_f\GM12878_Enhancer_extended_Overlap_Histonefeatures_1000\Overlap_GM12878_train_Enhancer_extended_H3K36me3.xls",header=None,names=["chr","Enhancer_start","Enhancer_end","pair_rows"])
EPI_pairrows_h3k36me3_feature=GM12878_train_Ehancer_Extended_h3k36me3_feature["pair_rows"]
EPI_pairrows_h3k36me3_feature_list=EPI_pairrows_h3k36me3_feature.tolist()
#列表去重
#不是原位排序
EPI_pairrows_h3k36me3_feature_list=list(set(EPI_pairrows_h3k36me3_feature_list))
#一共有0到5347个EPI_pairs,0 or 1 his_feature of enhancer_Extended of per pair->Enhancer_Extended_h3k36me3_feature
Enhancer_Extended_h3k36me3_feature=[]
for i in range(5348):
    if i in EPI_pairrows_h3k36me3_feature_list:
        Enhancer_Extended_h3k36me3_feature.append(1)
    else:
        Enhancer_Extended_h3k36me3_feature.append(0)
print(Enhancer_Extended_h3k36me3_feature)

#GM12878_train_Ehancer_Extended_h3k79me2
GM12878_train_Ehancer_Extended_h3k79me2_feature=pd.read_excel("Enhancer_extended_f\GM12878_Enhancer_extended_Overlap_Histonefeatures_1000\Overlap_GM12878_train_Enhancer_extended_H3K79me2.xls",header=None,names=["chr","Enhancer_start","Enhancer_end","pair_rows"])
EPI_pairrows_h3k79me2_feature=GM12878_train_Ehancer_Extended_h3k79me2_feature["pair_rows"]
EPI_pairrows_h3k79me2_feature_list=EPI_pairrows_h3k79me2_feature.tolist()
#列表去重
#不是原位排序
EPI_pairrows_h3k79me2_feature_list=list(set(EPI_pairrows_h3k79me2_feature_list))
#一共有0到5347个EPI_pairs,0 or 1 his_feature of enhancer_Extended of per pair->Enhancer_Extended_h3k79me2_feature
Enhancer_Extended_h3k79me2_feature=[]
for i in range(5348):
    if i in EPI_pairrows_h3k79me2_feature_list:
        Enhancer_Extended_h3k79me2_feature.append(1)
    else:
        Enhancer_Extended_h3k79me2_feature.append(0)
print(Enhancer_Extended_h3k79me2_feature)
#
#GM12878_train_Ehancer_Extended_h4k20me1
GM12878_train_Ehancer_Extended_h4k20me1_feature=pd.read_excel("Enhancer_extended_f\GM12878_Enhancer_extended_Overlap_Histonefeatures_1000\Overlap_GM12878_train_Enhancer_extended_H4K20me1.xls",header=None,names=["chr","Enhancer_start","Enhancer_end","pair_rows"])
EPI_pairrows_h4k20me1_feature=GM12878_train_Ehancer_Extended_h4k20me1_feature["pair_rows"]
EPI_pairrows_h4k20me1_feature_list=EPI_pairrows_h4k20me1_feature.tolist()
#列表去重
#不是原位排序
EPI_pairrows_h4k20me1_feature_list=list(set(EPI_pairrows_h4k20me1_feature_list))
#一共有0到5347个EPI_pairs,0 or 1 his_feature of enhancer_Extended of per pair->Enhancer_Extended_h4k20me1_feature
Enhancer_Extended_h4k20me1_feature=[]
for i in range(5348):
    if i in EPI_pairrows_h4k20me1_feature_list:
        Enhancer_Extended_h4k20me1_feature.append(1)
    else:
        Enhancer_Extended_h4k20me1_feature.append(0)
print(Enhancer_Extended_h4k20me1_feature)
#
#多个GM12878_train_Ehancer_Extended_histone_features形成矩阵
import numpy as np
Enhancer_Extended_H2AFZ_feature=np.array(Enhancer_Extended_H2AFZ_feature)
#print(type(Enhancer_Extended_H2AFZ_feature))
Enhancer_Extended_h3k4me1_feature=np.array(Enhancer_Extended_h3k4me1_feature)
Enhancer_Extended_h3k4me2_feature=np.array(Enhancer_Extended_h3k4me2_feature)
Enhancer_Extended_h3k4me3_feature=np.array(Enhancer_Extended_h3k4me3_feature)
Enhancer_Extended_h3k9ac_feature=np.array(Enhancer_Extended_h3k9ac_feature)
#Enhancer_Extended_h3k9me1_feature=np.array(Enhancer_Extended_h3k9me1_feature)
Enhancer_Extended_h3k9me3_feature=np.array(Enhancer_Extended_h3k9me3_feature)
Enhancer_Extended_h3k27ac_feature=np.array(Enhancer_Extended_h3k27ac_feature)
Enhancer_Extended_h3k27me3_feature=np.array(Enhancer_Extended_h3k27me3_feature)
Enhancer_Extended_h3k36me3_feature=np.array(Enhancer_Extended_h3k36me3_feature)
Enhancer_Extended_h3k79me2_feature=np.array(Enhancer_Extended_h3k79me2_feature)
Enhancer_Extended_h4k20me1_feature=np.array(Enhancer_Extended_h4k20me1_feature)
GM12878_training_EPI_Enhancer_Extended_histonefeature_Matrix=np.vstack((Enhancer_Extended_H2AFZ_feature,Enhancer_Extended_h3k4me1_feature,
                                                  Enhancer_Extended_h3k4me2_feature,Enhancer_Extended_h3k4me3_feature,
                                                  Enhancer_Extended_h3k9ac_feature,
                                                  Enhancer_Extended_h3k9me3_feature,Enhancer_Extended_h3k27ac_feature,
                                                 Enhancer_Extended_h3k27me3_feature,Enhancer_Extended_h3k36me3_feature,
                                                  Enhancer_Extended_h3k79me2_feature,Enhancer_Extended_h4k20me1_feature)).T
print("finished Matrix_Extended")
print(GM12878_training_EPI_Enhancer_Extended_histonefeature_Matrix.shape)
np.savetxt("GM12878_training_EPI_Enhancer_Extended_histonefeature_Matrix_1000.txt",GM12878_training_EPI_Enhancer_Extended_histonefeature_Matrix)

import numpy as np
EMatrix_Extended=np.loadtxt("GM12878_training_EPI_Enhancer_Extended_histonefeature_Matrix_1000.txt")
print(EMatrix_Extended.shape)
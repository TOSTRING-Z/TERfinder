#to generate Promoter_his_feature

#生成所有K562_training_Dnasepairs的 Histone_feature_matrix
import pandas as pd
#to generate Promoter_his_feature

#H2AFZ
K562_train_Promoter_H2AFZ_feature=pd.read_excel("Promoter_f\K562_train_Promoter_H2AFZ.xls",header=None,names=["chr","Promoter_start","Promoter_end","pair_rows"])
#print(K562_train_Promoter_H2AFZ_feature.head())

#取EPIs pair的 pair_rows
EPI_pairrows_H2AFZ_feature=K562_train_Promoter_H2AFZ_feature["pair_rows"]
#<class 'pandas.core.series.Series'>
#print(type(EPI_pairrows))

EPI_pairrows_H2AFZ_feature_list=EPI_pairrows_H2AFZ_feature.tolist()
#print(type(EPI_pairrows_H2AFZ_feature_list))
#print(EPI_pairrows_H2AFZ_feature_list)

#列表去重
#不是原位排序
EPI_pairrows_H2AFZ_feature_list=list(set(EPI_pairrows_H2AFZ_feature_list))
#print(EPI_pairrows_H2AFZ_feature_list)
#一共有0到8649个EPI_pairs,0 or 1 his_feature of Promoter of per pair->Promoter_H2AFZ_feature
Promoter_H2AFZ_feature=[]
for i in range(8650):
    if i in EPI_pairrows_H2AFZ_feature_list:
        Promoter_H2AFZ_feature.append(1)
    else:
        Promoter_H2AFZ_feature.append(0)
#finished generating Promoter_H2AFZ_feature
#print(len(Promoter_H2AFZ_feature))
print(Promoter_H2AFZ_feature)

#h3k4me1
K562_train_Promoter_h3k4me1_feature=pd.read_excel("Promoter_f\K562_train_Promoter_h3k4me1.xls",header=None,names=["chr","Promoter_start","Promoter_end","pair_rows"])
#取EPIs pair的 pair_rows
EPI_pairrows_h3k4me1_feature=K562_train_Promoter_h3k4me1_feature["pair_rows"]
EPI_pairrows_h3k4me1_feature_list=EPI_pairrows_h3k4me1_feature.tolist()
#列表去重
#不是原位排序
EPI_pairrows_h3k4me1_feature_list=list(set(EPI_pairrows_h3k4me1_feature_list))
#一共有0到8649个EPI_pairs,0 or 1 his_feature of Promoter of per pair->Promoter_h3k4me1_feature
Promoter_h3k4me1_feature=[]
for i in range(8650):
    if i in EPI_pairrows_h3k4me1_feature_list:
        Promoter_h3k4me1_feature.append(1)
    else:
        Promoter_h3k4me1_feature.append(0)
print(Promoter_h3k4me1_feature)

# #K562_train_Promoter_h3k4me2
K562_train_Promoter_h3k4me2_feature=pd.read_excel("Promoter_f\K562_train_Promoter_h3k4me2.xls",header=None,names=["chr","Promoter_start","Promoter_end","pair_rows"])
#取EPIs pair的 pair_rows
EPI_pairrows_h3k4me2_feature=K562_train_Promoter_h3k4me2_feature["pair_rows"]
EPI_pairrows_h3k4me2_feature_list=EPI_pairrows_h3k4me2_feature.tolist()
#列表去重
#不是原位排序
EPI_pairrows_h3k4me2_feature_list=list(set(EPI_pairrows_h3k4me2_feature_list))
#一共有0到8649个EPI_pairs,0 or 1 his_feature of Promoter of per pair->Promoter_h3k4me2_feature
Promoter_h3k4me2_feature=[]
for i in range(8650):
    if i in EPI_pairrows_h3k4me2_feature_list:
        Promoter_h3k4me2_feature.append(1)
    else:
        Promoter_h3k4me2_feature.append(0)
print(Promoter_h3k4me2_feature)

#K562_train_Promoter_h3k4me3
K562_train_Promoter_h3k4me3_feature=pd.read_excel("Promoter_f\K562_train_Promoter_h3k4me3.xls",header=None,names=["chr","Promoter_start","Promoter_end","pair_rows"])
EPI_pairrows_h3k4me3_feature=K562_train_Promoter_h3k4me3_feature["pair_rows"]
EPI_pairrows_h3k4me3_feature_list=EPI_pairrows_h3k4me3_feature.tolist()
#列表去重
#不是原位排序
EPI_pairrows_h3k4me3_feature_list=list(set(EPI_pairrows_h3k4me3_feature_list))
#一共有0到8649个EPI_pairs,0 or 1 his_feature of Promoter of per pair->Promoter_h3k4me3_feature
Promoter_h3k4me3_feature=[]
for i in range(8650):
    if i in EPI_pairrows_h3k4me3_feature_list:
        Promoter_h3k4me3_feature.append(1)
    else:
        Promoter_h3k4me3_feature.append(0)
print(Promoter_h3k4me3_feature)

#K562_train_Promoter_h3k9ac
K562_train_Promoter_h3k9ac_feature=pd.read_excel("Promoter_f\K562_train_Promoter_h3k9ac.xls",header=None,names=["chr","Promoter_start","Promoter_end","pair_rows"])
EPI_pairrows_h3k9ac_feature=K562_train_Promoter_h3k9ac_feature["pair_rows"]
EPI_pairrows_h3k9ac_feature_list=EPI_pairrows_h3k9ac_feature.tolist()
#列表去重
#不是原位排序
EPI_pairrows_h3k9ac_feature_list=list(set(EPI_pairrows_h3k9ac_feature_list))
#一共有0到8649个EPI_pairs,0 or 1 his_feature of Promoter of per pair->Promoterr_h3k9ac_feature
Promoter_h3k9ac_feature=[]
for i in range(8650):
    if i in EPI_pairrows_h3k9ac_feature_list:
        Promoter_h3k9ac_feature.append(1)
    else:
        Promoter_h3k9ac_feature.append(0)
print(Promoter_h3k9ac_feature)

#K562_train_Promoter_h3k9me1
K562_train_Promoter_h3k9me1_feature=pd.read_excel("Promoter_f\K562_train_Promoter_h3k9me1.xls",header=None,names=["chr","Promoter_start","Promoter_end","pair_rows"])
EPI_pairrows_h3k9me1_feature=K562_train_Promoter_h3k9me1_feature["pair_rows"]
EPI_pairrows_h3k9me1_feature_list=EPI_pairrows_h3k9me1_feature.tolist()
#列表去重
#不是原位排序
EPI_pairrows_h3k9me1_feature_list=list(set(EPI_pairrows_h3k9me1_feature_list))
#一共有0到8649个EPI_pairs,0 or 1 his_feature of Promoter of per pair->Promoter_h3k9me1_feature
Promoter_h3k9me1_feature=[]
for i in range(8650):
    if i in EPI_pairrows_h3k9me1_feature_list:
        Promoter_h3k9me1_feature.append(1)
    else:
        Promoter_h3k9me1_feature.append(0)
print(Promoter_h3k9me1_feature)

#K562_train_Promoter_h3k9me3
K562_train_Promoter_h3k9me3_feature=pd.read_excel("Promoter_f\K562_train_Promoter_h3k9me3.xls",header=None,names=["chr","Promoter_start","Promoter_end","pair_rows"])
EPI_pairrows_h3k9me3_feature=K562_train_Promoter_h3k9me3_feature["pair_rows"]
EPI_pairrows_h3k9me3_feature_list=EPI_pairrows_h3k9me3_feature.tolist()
#列表去重
#不是原位排序
EPI_pairrows_h3k9me3_feature_list=list(set(EPI_pairrows_h3k9me3_feature_list))
#一共有0到8649个EPI_pairs,0 or 1 his_feature of Promoter of per pair->Promoter_h3k9me3_feature
Promoter_h3k9me3_feature=[]
for i in range(8650):
    if i in EPI_pairrows_h3k9me3_feature_list:
        Promoter_h3k9me3_feature.append(1)
    else:
        Promoter_h3k9me3_feature.append(0)
print(Promoter_h3k9me3_feature)

# #K562_train_Promoter_h3k27ac
K562_train_Promoter_h3k27ac_feature=pd.read_excel("Promoter_f\K562_train_Promoter_h3k27ac.xls",header=None,names=["chr","Promoter_start","Promoter_end","pair_rows"])
EPI_pairrows_h3k27ac_feature=K562_train_Promoter_h3k27ac_feature["pair_rows"]
EPI_pairrows_h3k27ac_feature_list=EPI_pairrows_h3k27ac_feature.tolist()
#列表去重
#不是原位排序
EPI_pairrows_h3k27ac_feature_list=list(set(EPI_pairrows_h3k27ac_feature_list))
#一共有0到8649个EPI_pairs,0 or 1 his_feature of Promoter of per pair->Promoter_h3k27ac_feature
Promoter_h3k27ac_feature=[]
for i in range(8650):
    if i in EPI_pairrows_h3k27ac_feature_list:
        Promoter_h3k27ac_feature.append(1)
    else:
        Promoter_h3k27ac_feature.append(0)
print(Promoter_h3k27ac_feature)

#K562_train_Promoter_h3k27me3
K562_train_Promoter_h3k27me3_feature=pd.read_excel("Promoter_f\K562_train_Promoter_h3k27me3.xls",header=None,names=["chr","Promoter_start","Promoter_end","pair_rows"])
EPI_pairrows_h3k27me3_feature=K562_train_Promoter_h3k27me3_feature["pair_rows"]
EPI_pairrows_h3k27me3_feature_list=EPI_pairrows_h3k27me3_feature.tolist()
#列表去重
#不是原位排序
EPI_pairrows_h3k27me3_feature_list=list(set(EPI_pairrows_h3k27me3_feature_list))
#一共有0到8649个EPI_pairs,0 or 1 his_feature of Promoter of per pair->Promoter_h3k27me3_feature
Promoter_h3k27me3_feature=[]
for i in range(8650):
    if i in EPI_pairrows_h3k27me3_feature_list:
        Promoter_h3k27me3_feature.append(1)
    else:
        Promoter_h3k27me3_feature.append(0)
print(Promoter_h3k27me3_feature)

#K562_train_Promoter_h3k36me3
K562_train_Promoter_h3k36me3_feature=pd.read_excel("Promoter_f\K562_train_Promoter_h3k36me3.xls",header=None,names=["chr","Promoter_start","Promoter_end","pair_rows"])
EPI_pairrows_h3k36me3_feature=K562_train_Promoter_h3k36me3_feature["pair_rows"]
EPI_pairrows_h3k36me3_feature_list=EPI_pairrows_h3k36me3_feature.tolist()
#列表去重
#不是原位排序
EPI_pairrows_h3k36me3_feature_list=list(set(EPI_pairrows_h3k36me3_feature_list))
#一共有0到8649个EPI_pairs,0 or 1 his_feature of Promoter of per pair->Promoter_h3k36me3_feature
Promoter_h3k36me3_feature=[]
for i in range(8650):
    if i in EPI_pairrows_h3k36me3_feature_list:
        Promoter_h3k36me3_feature.append(1)
    else:
        Promoter_h3k36me3_feature.append(0)
print(Promoter_h3k36me3_feature)

#K562_train_Promoter_h3k79me2
K562_train_Promoter_h3k79me2_feature=pd.read_excel("Promoter_f\K562_train_Promoter_h3k79me2.xls",header=None,names=["chr","Promoter_start","Promoter_end","pair_rows"])
EPI_pairrows_h3k79me2_feature=K562_train_Promoter_h3k79me2_feature["pair_rows"]
EPI_pairrows_h3k79me2_feature_list=EPI_pairrows_h3k79me2_feature.tolist()
#列表去重
#不是原位排序
EPI_pairrows_h3k79me2_feature_list=list(set(EPI_pairrows_h3k79me2_feature_list))
#一共有0到8649个EPI_pairs,0 or 1 his_feature of Promoter of per pair->Promoter_h3k79me2_feature
Promoter_h3k79me2_feature=[]
for i in range(8650):
    if i in EPI_pairrows_h3k79me2_feature_list:
        Promoter_h3k79me2_feature.append(1)
    else:
        Promoter_h3k79me2_feature.append(0)
print(Promoter_h3k79me2_feature)

#K562_train_Promoter_h4k20me1
K562_train_Promoter_h4k20me1_feature=pd.read_excel("Promoter_f\K562_train_Promoter_h4k20me1.xls",header=None,names=["chr","Promoter_start","Promoter_end","pair_rows"])
EPI_pairrows_h4k20me1_feature=K562_train_Promoter_h4k20me1_feature["pair_rows"]
EPI_pairrows_h4k20me1_feature_list=EPI_pairrows_h4k20me1_feature.tolist()
#列表去重
#不是原位排序
EPI_pairrows_h4k20me1_feature_list=list(set(EPI_pairrows_h4k20me1_feature_list))
#一共有0到8649个EPI_pairs,0 or 1 his_feature of Promoter of per pair->Promoter_h4k20me1_feature
Promoter_h4k20me1_feature=[]
for i in range(8650):
    if i in EPI_pairrows_h4k20me1_feature_list:
        Promoter_h4k20me1_feature.append(1)
    else:
        Promoter_h4k20me1_feature.append(0)
print(Promoter_h4k20me1_feature)

#多个K562_train_Promoter_histone_features形成矩阵
import numpy as np
Promoter_H2AFZ_feature=np.array(Promoter_H2AFZ_feature)
#print(type(Promoter_H2AFZ_feature))
Promoter_h3k4me1_feature=np.array(Promoter_h3k4me1_feature)
Promoter_h3k4me2_feature=np.array(Promoter_h3k4me2_feature)
Promoter_h3k4me3_feature=np.array(Promoter_h3k4me3_feature)
Promoter_h3k9ac_feature=np.array(Promoter_h3k9ac_feature)
Promoter_h3k9me1_feature=np.array(Promoter_h3k9me1_feature)
Promoter_h3k9me3_feature=np.array(Promoter_h3k9me3_feature)
Promoter_h3k27ac_feature=np.array(Promoter_h3k27ac_feature)
Promoter_h3k27me3_feature=np.array(Promoter_h3k27me3_feature)
Promoter_h3k36me3_feature=np.array(Promoter_h3k36me3_feature)
Promoter_h3k79me2_feature=np.array(Promoter_h3k79me2_feature)
Promoter_h4k20me1_feature=np.array(Promoter_h4k20me1_feature)
K562_training_EPI_Promoter_histonefeature_Matrix=np.vstack((Promoter_H2AFZ_feature,Promoter_h3k4me1_feature,
                                                            Promoter_h3k4me2_feature,Promoter_h3k4me3_feature,
                                                            Promoter_h3k9ac_feature,Promoter_h3k9me1_feature,
                                                            Promoter_h3k9me3_feature,Promoter_h3k27ac_feature,
                                                            Promoter_h3k27me3_feature,Promoter_h3k36me3_feature,
                                                            Promoter_h3k79me2_feature,Promoter_h4k20me1_feature)).T
print("finished Promoter Matrix")
print(K562_training_EPI_Promoter_histonefeature_Matrix.shape)
np.savetxt("K562_training_EPI_Promoter_histonefeature_Matrix.txt",K562_training_EPI_Promoter_histonefeature_Matrix)

import numpy as np
PMatrix=np.loadtxt("K562_training_EPI_Promoter_histonefeature_Matrix.txt")
print(PMatrix.shape)
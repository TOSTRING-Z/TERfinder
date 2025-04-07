"""
GM12878细胞系:11+11=22
DeepEPI: Predict Ehancer-Promoter interactions status using DNA sequence and histone so on of EPI pairs.
Copyright (C) 2022  Xuxiaoqiang.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.metrics import auc
from DeepEPI import EPI_AutoEncoders
from sklearn import preprocessing
import random
from pandas import DataFrame as df
torch.manual_seed(2022912)
random.seed(2022912)

#加载增强子启动子pairs信息
#enhancer
f=open('Enhancer_extended_f\GM12878_training_extended_enhancercor12_h19_DNASequence.fa')
ls_GM12878_training_extended_enhancercor12_h19_DNASequence=[]
for line in f:
        if not line.startswith('>'):
                ls_GM12878_training_extended_enhancercor12_h19_DNASequence.append(line.replace('\n',''))    #去掉行尾的换行符真的很重要！
f.close()
print("load  each enhancer sequence inlist and all pairs len")
print(len(ls_GM12878_training_extended_enhancercor12_h19_DNASequence))

#one-hot coding
#定义一个4*1000的array
#1000=enlen->增强子
#2000=prlen->启动子

#对DNA序列进行one-hot encoding
#test data processing
# function to one-hot encode a DNA sequence string
# non 'acgt' bases (n) are 0000
# returns a L x 4 numpy array

def One_hot_Coding(Str_Sequence):
    SequenceLengh=len(Str_Sequence)
    One_hotCoding_Matrix=np.zeros((4,SequenceLengh))
    for index in range(SequenceLengh):
        #print(Str_Sequence[index])
        if Str_Sequence[index] == "a" or Str_Sequence[index] == "A":
            One_hotCoding_Matrix[0][index]=1
        elif Str_Sequence[index] == "c" or Str_Sequence[index] == "C":
            One_hotCoding_Matrix[1][index] = 1
        elif Str_Sequence[index] == "g" or Str_Sequence[index] == "G":
            One_hotCoding_Matrix[2][index] = 1
        elif Str_Sequence[index] == "t" or Str_Sequence[index] == "T":
            One_hotCoding_Matrix[3][index] = 1
        else:
            continue
    return One_hotCoding_Matrix
print("enhancer sequence one-hot coding")
# test_sequence = 'GCaAaGGTTNN'
# TestMatrix=One_hot_Coding(test_sequence)
# print(TestMatrix)
ls_GM12878_training_extended_enhancercor12_h19_DNASequence_OnehotCoding=[]
for seq in ls_GM12878_training_extended_enhancercor12_h19_DNASequence:
        ls_GM12878_training_extended_enhancercor12_h19_DNASequence_OnehotCoding.append(One_hot_Coding(seq))
print("编码后的all len")
print(len(ls_GM12878_training_extended_enhancercor12_h19_DNASequence_OnehotCoding))
#print(ls_GM12878_training_extended_enhancercor12_h19_DNASequence_OnehotCoding[0].shape)

print("load  each promoter sequence inlist")
#promoter
f=open('Promoter_extended_f\GM12878_training_promotercor34_h19_DNASequence.fa')
ls_GM12878_training_promoter_h19_DNASequence=[]
for line in f:
        if not line.startswith('>'):
                ls_GM12878_training_promoter_h19_DNASequence.append(line.replace('\n',''))    #去掉行尾的换行符真的很重要！
f.close()
print("load  each promoter sequence inlist and all len")
print(len(ls_GM12878_training_promoter_h19_DNASequence))

print("promoter sequence one-hot coding &len")
#One hot Coding
ls_GM12878_training_promoter_h19_DNASequence_OnehotCoding=[]
for seq in ls_GM12878_training_promoter_h19_DNASequence:
        ls_GM12878_training_promoter_h19_DNASequence_OnehotCoding.append(One_hot_Coding(seq))
print(len(ls_GM12878_training_promoter_h19_DNASequence_OnehotCoding))
# print(ls_GM12878_training_promoter_h19_DNASequence_OnehotCoding[0].shape)
#数据加载完成
print("数据加载完成！")

#加载GM12878的dpair数量和label
#DataPrepare
#GM12878_training_pairs_DNase
GM12878_training_pairs_DNase=pd.read_csv("GM12878_training_pairs_DNase.bed",sep='\t',index_col=0)
print(GM12878_training_pairs_DNase.head())
# 将GM12878_training_pairs_DNase的label保存到GM12878_training_pairs_label.txt
filename='GM12878_training_pairs_label.txt'
np.savetxt(filename,GM12878_training_pairs_DNase.values[:,-1])
#以上代码跑过一次即可，已将label保存到txt文件中
#统计正负样本的数量
from collections import Counter
GM12878_training_pairs_label=np.loadtxt(filename)
d=Counter(GM12878_training_pairs_label)
print(d)
#Counter({-1.0: 3601, 1.0: 1747})
#5 crossValidation 1747*0.8=1397

filename1='GM12878_training_pairs_label.txt'
GM12878_training_pairs_label = np.loadtxt(filename1)
GM12878_training_pairs_label = list(GM12878_training_pairs_label)
#print(K562_training_pairs_label)
GM12878_training_pairs_label_tuple=[]
for i in range(len(GM12878_training_pairs_label)):
    GM12878_training_pairs_label_tuple.append((i,GM12878_training_pairs_label[i]))     #（行号，标签）
GM12878_training_pairs_label_tuple_df=df(GM12878_training_pairs_label_tuple)
GM12878_training_pairs_label_tuple_df.to_csv("GM12878_training_pairs_label_tuple.csv")
GM12878_training_positive_pairs_label=GM12878_training_pairs_label_tuple[0:1747]
GM12878_training_positive_pairs_label_df=df(GM12878_training_positive_pairs_label)
GM12878_training_positive_pairs_label_df.to_csv("GM12878_training_positive_pairs_label.csv")
GM12878_training_negative_pairs_label=GM12878_training_pairs_label_tuple[1747:]
GM12878_training_negative_pairs_label_df=df(GM12878_training_negative_pairs_label)
GM12878_training_negative_pairs_label_df.to_csv("GM12878_training_negative_pairs_label.csv")

#打乱
random.shuffle(GM12878_training_positive_pairs_label)
random.shuffle(GM12878_training_negative_pairs_label)
#[(985, 1.0), (2024, 1.0),...,]
# print(GM12878_training_negative_pairs_label)
# print(GM12878_training_positive_pairs_label)
#分割为训练集和测试集
#训练集
Model_GM12878_training_positive_pairs_label=GM12878_training_positive_pairs_label[0:1397]#阳性集合
Model_GM12878_training_negative_pairs_label=GM12878_training_negative_pairs_label[0:1397]#阴性集；暂定1:1
Model_GM12878_training_pairs=Model_GM12878_training_positive_pairs_label+Model_GM12878_training_negative_pairs_label
print("Training dataset len 2794")
print(len(Model_GM12878_training_pairs))
Model_GM12878_training_pairs_df=df(Model_GM12878_training_pairs)
Model_GM12878_training_pairs_df.to_csv("Model_GM12878_training_pairs.csv")
#测试集
Model_GM12878_testing_positive_pairs_label=GM12878_training_positive_pairs_label[1397:]#阳性集合
Model_GM12878_testing_negative_pairs_label=GM12878_training_negative_pairs_label[1397:1747]#阴性集；剩余集合都是阴性集合--可修改
Model_GM12878_testing_pairs=Model_GM12878_testing_positive_pairs_label+Model_GM12878_testing_negative_pairs_label
Model_GM12878_testing_pairs_df=df(Model_GM12878_testing_pairs)
Model_GM12878_testing_pairs_df.to_csv("Model_GM12878_testing_pairs.csv")
print("Test dataset len")
print(len(Model_GM12878_testing_pairs))
#定义训练集和验证集的数据加载器---def load_data：
#分割为训练集和验证集
#转为张量数据
#将X, Y转化为数据集
'''
def load_data(ls_Enhancer_h19_DNASequence_OnehotCoding,ls_Promoter_h19_DNASequence_OnehotCoding,ls_GM12878_EPI_pairs):
    pass
#One hot Coding
ls_Enhancer_h19_DNASequence_OnehotCoding:list,元素为Enhancer_h19_DNASequence_OnehotCoding；4*L
ls_Promoter_h19_DNASequence_OnehotCoding:list,元素为Promoter_h19_DNASequence_OnehotCoding：4*L
K562_EPI_pairs:list，元素为(i,pairs_label),比如[(985, 1.0), (2024, 1.0),...,]；实际为训练集和验证集
EPIhistonefeature_Matrix:增强子启动子对的特征矩阵pairs*features
'''
print("加载组蛋白特征")
#加载组蛋白特征
#enhancer
GM12878_training_EPI_Enhancer_Extended_histonefeature_Matrix=np.loadtxt("GM12878_training_EPI_Enhancer_Extended_histonefeature_Matrix_1000.txt")
#print(GM12878_training_EPI_Enhancer_Extended_histonefeature_Matrix.shape)
#promoter
GM12878_training_EPI_Promoter_histonefeature_Matrix=np.loadtxt("GM12878_training_EPI_Promoter_histonefeature_Matrix.txt")
#print(GM12878_training_EPI_Promoter_histonefeature_Matrix.shape)
Enhancer_Promoter_Histone_fetures=np.concatenate((GM12878_training_EPI_Enhancer_Extended_histonefeature_Matrix,GM12878_training_EPI_Promoter_histonefeature_Matrix),axis=1)
print(Enhancer_Promoter_Histone_fetures.shape)
np.savetxt("Enhancer_Promoter_Histone_fetures.txt",Enhancer_Promoter_Histone_fetures)
print("loading NN training data")
def load_data(ls_Enhancer_h19_DNASequence_OnehotCoding,ls_Promoter_h19_DNASequence_OnehotCoding,EPIhistonefeature_Matrix,ls_GM12878_EPI_pairs,BATCHSIZE):
    X=[]
    Y=[]
    Z=[]
    for t in ls_GM12878_EPI_pairs:
        # print(t[0])
        # print(type(ls_Enhancer_h19_DNASequence_OnehotCoding[t[0]]))
        # print(ls_Enhancer_h19_DNASequence_OnehotCoding[t[0]].shape)
        # print(type(ls_Promoter_h19_DNASequence_OnehotCoding[t[0]]))
        # print(ls_Promoter_h19_DNASequence_OnehotCoding[t[0]].shape)
        X.append(np.concatenate((ls_Enhancer_h19_DNASequence_OnehotCoding[t[0]],ls_Promoter_h19_DNASequence_OnehotCoding[t[0]]),axis=1))

        Y.append(np.concatenate((ls_Enhancer_h19_DNASequence_OnehotCoding[t[0]], ls_Promoter_h19_DNASequence_OnehotCoding[t[0]]), axis=1))

        Z.append(EPIhistonefeature_Matrix[t[0]])

    print(X[0])
    print(X[1])
    print(Y[0])
    print(Y[1])
    print(Z[0])
    print(Z[1])
    X = torch.FloatTensor(X)
    #print(X.size())
    Y = torch.FloatTensor(np.array(Y))
    #print(Y.size())
    Z=torch.FloatTensor(np.array(Z))
    torch_dataset = Data.TensorDataset(X,Y,Z)
    data_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCHSIZE,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )
    return data_loader
#调用,Batchsize=10
training_dataloder=load_data(ls_GM12878_training_extended_enhancercor12_h19_DNASequence_OnehotCoding,ls_GM12878_training_promoter_h19_DNASequence_OnehotCoding,Enhancer_Promoter_Histone_fetures,Model_GM12878_training_pairs,64)
# lsX=df(lsX)
# lsX.to_csv("loadinglsX.csv")
# lsY=df(lsY)
# lsY.to_csv("loadinglsY.csv")
# lsZ=df(lsZ)
# lsZ.to_csv("loadinglsZ.csv")
#模型超参数；#模型超参数
EPOCH=10000
LR=0.001
K=1# loss = K*loss_func1(xlatten,z)+loss_func2(Yout,train_label)      # 计算损失函数
EPIpred=EPI_AutoEncoders()
if torch.cuda.is_available():
    EPIpred.cuda()
optimizer=torch.optim.Adam(EPIpred.parameters(),lr=LR)    #定义优化方式
loss_func1=nn.BCEWithLogitsLoss()# 相似性误差;nn.BCEWithLogitsLoss();yuanlaicode:nn.BCELoss(reduction="mean")
loss_func2=nn.MSELoss(size_average=True,reduction="mean")#重构误差
print("开始训练")
#开始训练
for epoch in range(EPOCH):
    train_loss=0
    train_acc=0
    for step, (x, train_label,z) in enumerate(training_dataloder):
        b_x = Variable(x).cuda()
        #b_x = Variable(x)
        #train_label=Variable(train_label.squeeze(1)).cuda()
        train_label = Variable(train_label).cuda()
        z=Variable(z).cuda()
        xlatten,Yout=EPIpred(b_x)
        loss = K*loss_func1(xlatten,z)+loss_func2(Yout,train_label)      # 计算损失函数
        optimizer.zero_grad()                  # 梯度清零
        loss.backward()                        # 反向传播
        optimizer.step()                       # 梯度优化
        train_loss+=loss.item()
        if step % 100 == 0:        #每10步显示一次
            print('epoch: ', epoch, '| train loss: %.8f' % loss.data.cpu().numpy())
    print('Epoch: {}, Train Loss: {:.8f}'
          .format(epoch, train_loss / len(training_dataloder)))
print("DL训练完成,开始ML部分")

EPIpred.eval()
#开始训练ML部分使用相同的训练集合，比如--GBDT
ML_training_dataloder=load_data(ls_GM12878_training_extended_enhancercor12_h19_DNASequence_OnehotCoding,ls_GM12878_training_promoter_h19_DNASequence_OnehotCoding,Enhancer_Promoter_Histone_fetures,Model_GM12878_training_pairs,1)
MLtraining_ls_EPIpred_R=[]
for step, (x0, test_label0, z) in enumerate(ML_training_dataloder):
    b_x0 = Variable(x0).cuda()
  #  b_x0 = Variable(x0)
    xlatten0, Yout0 = EPIpred(b_x0)
    MLtraining_ls_EPIpred_R.append(xlatten0.detach().cpu().numpy().tolist())
print(len(MLtraining_ls_EPIpred_R))
print("(training)Type of output Result of NN")
print(type(MLtraining_ls_EPIpred_R[0]))
#print(MLtraining_ls_EPIpred_R[1])
#加载测试集合
testing_dataloder=load_data(ls_GM12878_training_extended_enhancercor12_h19_DNASequence_OnehotCoding,ls_GM12878_training_promoter_h19_DNASequence_OnehotCoding,Enhancer_Promoter_Histone_fetures,Model_GM12878_testing_pairs,1)
#o=np.zeros((0,24))
#print(o.shape)
MLtesting_ls_EPIpred_R=[]
for step, (x, test_label, z) in enumerate(testing_dataloder):
    b_x0 = Variable(x).cuda()
    #b_x = Variable(x)
    xlatten, Yout = EPIpred(b_x0)
    MLtesting_ls_EPIpred_R.append(xlatten.detach().cpu().numpy().tolist())
print(len(MLtesting_ls_EPIpred_R))
print(type(MLtesting_ls_EPIpred_R[0]))
# print(MLtesting_ls_EPIpred_R[1])
#GBDT
from sklearn.ensemble import GradientBoostingClassifier
#RF
from sklearn.ensemble import RandomForestClassifier
#Adaboost
from sklearn.ensemble import AdaBoostClassifier
#xgboost
import xgboost as xgb
#LR
from sklearn.linear_model import LogisticRegression
#SVM
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix,f1_score,precision_recall_curve
import matplotlib.pyplot as plt

#processing ML Xtraining_data
print("processing ML Xtraining_data")
#三维列表，元素为[[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.4263348581775565e-34, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]]
print(sum(MLtraining_ls_EPIpred_R,[])[0])
ML_Xtraining_data=df(sum(MLtraining_ls_EPIpred_R,[]))
ML_Xtraining_data.to_csv("ML_Xtraining_data.csv")
#processing ML Ytraining_data
ls_Ytraining_data=[]
for t in Model_GM12878_training_pairs:
    ls_Ytraining_data.append(t[1])
print(len(ls_Ytraining_data))
ML_Ytraining_data=df(ls_Ytraining_data)
print("查看label")
print(ML_Ytraining_data)
#ML_Ytraining_data=ML_Ytraining_data.map({'1':1,"-1":0}).astype(int)
print(ML_Ytraining_data.values.shape)
ML_Xtesting_data=df(sum(MLtesting_ls_EPIpred_R,[]))
ML_Xtesting_data.to_csv("ML_Xtesting_data.csv")
ls_Ytesting_data=[]
for t in Model_GM12878_testing_pairs:
    ls_Ytesting_data.append(t[1])
ML_Ytesting_data=df(ls_Ytesting_data)
print("查看label")
print(ML_Ytesting_data)
#ML_Ytesting_data=ML_Ytesting_data.map({'1':1,"-1":0}).astype(int)
print(ML_Ytesting_data.values.shape)
print("trainingdata")
print(ML_Xtraining_data.head())
print(ML_Xtraining_data.values.shape)
print(ML_Ytraining_data.head())
print(ML_Ytraining_data.values.shape)
print("testdata")
print(ML_Xtesting_data.head())
print(ML_Xtesting_data.values.shape)
print(ML_Ytesting_data.head())
print(ML_Ytesting_data.values.shape)
print("finished加载数据")
print("GBDT training")
#GBDT--gbdt_
gbdt = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=50, subsample=1
                                  , min_samples_split=2, min_samples_leaf=1, max_depth=3
                                  , init=None, random_state=None, max_features=None
                                  , verbose=0, max_leaf_nodes=None, warm_start=False
                                  )

gbdt.fit(ML_Xtraining_data,ML_Ytraining_data)
gbdt_train_score=gbdt.score(ML_Xtraining_data,ML_Ytraining_data)
print("gbdt_训练集准确率得分：",gbdt_train_score)
gbdt_test_pred_proba=gbdt.predict_proba(ML_Xtesting_data)
gbdt_test_pred=gbdt.predict(ML_Xtesting_data)
print(gbdt_test_pred_proba)
print(gbdt_test_pred)
#统计F1分数
ls_F1_Methods=[]
ls_F1_Score=[]
#统计训练集和测试集合的准确率
ls_training_acc=[]
ls_testing_acc=[]
print("finished GBDT training ")
print("gbdt_testing_acc:{:.4f}".format(accuracy_score(ML_Ytesting_data,gbdt_test_pred)))
print("gbdt_F1_Score:{:.4f}".format(f1_score(ML_Ytesting_data,gbdt_test_pred)))
ls_F1_Methods.append(" GBDT")
ls_F1_Score.append(f1_score(ML_Ytesting_data,gbdt_test_pred))
ls_training_acc.append(gbdt_train_score)
ls_testing_acc.append(accuracy_score(ML_Ytesting_data,gbdt_test_pred))

#LR--lr_
print("LogisticRegression training")
LR=LogisticRegression()
LR.fit(ML_Xtraining_data,ML_Ytraining_data)
lr_score=LR.score(ML_Xtraining_data,ML_Ytraining_data)
print("lr_训练集准确率得分：",lr_score)
lr_test_pred_proba=LR.predict_proba(ML_Xtesting_data)
lr_test_pred=LR.predict(ML_Xtesting_data)
print("finished LogisticRegression training ")
print("lr_testing_acc:{:.4f}".format(accuracy_score(ML_Ytesting_data,lr_test_pred)))
print("lr__F1_Score:{:.4f}".format(f1_score(ML_Ytesting_data,lr_test_pred)))
ls_F1_Methods.append("LogisticRegression")
ls_F1_Score.append(f1_score(ML_Ytesting_data,lr_test_pred))
ls_training_acc.append(lr_score)
ls_testing_acc.append(accuracy_score(ML_Ytesting_data,lr_test_pred))

#RF--rf_
print("RandomForest training")
rf=RandomForestClassifier(n_estimators=50)
rf.fit(ML_Xtraining_data,ML_Ytraining_data)
rf_score=rf.score(ML_Xtraining_data,ML_Ytraining_data)
print("rf_训练集准确率得分：",rf_score)
rf_test_pred=rf.predict(ML_Xtesting_data)
rf_test_pred_proba=rf.predict_proba(ML_Xtesting_data)
print("finished RandomForest training")
print("rf_testing_acc:{:.4f}".format(accuracy_score(ML_Ytesting_data,rf_test_pred)))
print("rf__F1_Score:{:.4f}".format(f1_score(ML_Ytesting_data,rf_test_pred)))
ls_F1_Methods.append("RandomForest")
ls_F1_Score.append(f1_score(ML_Ytesting_data,rf_test_pred))
ls_training_acc.append(rf_score)
ls_testing_acc.append(accuracy_score(ML_Ytesting_data,rf_test_pred))

#XGBoost--xgb_
print("XGBoost training")
XGB=xgb.XGBClassifier(objective='binary:logistic')
XGB.fit(ML_Xtraining_data,ML_Ytraining_data)
xgb_score=XGB.score(ML_Xtraining_data,ML_Ytraining_data)
print("xgb_训练集准确率得分：",xgb_score)
xgb_test_pred_proba=XGB.predict_proba(ML_Xtesting_data)
xgb_test_pred=XGB.predict(ML_Xtesting_data)
print("finished XGBoost training")
print("XGBoost_testing_acc:{:.4f}".format(accuracy_score(ML_Ytesting_data,xgb_test_pred)))
print("xgb_F1_Score:{:.4f}".format(f1_score(ML_Ytesting_data,xgb_test_pred)))
ls_F1_Methods.append("XGBoost")
ls_F1_Score.append(f1_score(ML_Ytesting_data,xgb_test_pred))
ls_training_acc.append(xgb_score)
ls_testing_acc.append(accuracy_score(ML_Ytesting_data,xgb_test_pred))

#SVM--svm_
print("SVM training")
svm=svm.SVC(C=1,gamma="scale",degree=3,decision_function_shape='ovr',max_iter=-1,kernel="rbf",probability=True)
svm.fit(ML_Xtraining_data,ML_Ytraining_data)
svm_score=svm.score(ML_Xtraining_data,ML_Ytraining_data)
print("svm_训练集准确率得分：",svm_score)
svm_test_pred_proba=svm.predict_proba(ML_Xtesting_data)
svm_test_pred=svm.predict(ML_Xtesting_data)
print("finished SVM training")
print("SVM_testing_acc:{:.4f}".format(accuracy_score(ML_Ytesting_data,svm_test_pred)))
print("svm_F1_Score:{:.4f}".format(f1_score(ML_Ytesting_data,svm_test_pred)))
ls_F1_Methods.append("SVM")
ls_F1_Score.append(f1_score(ML_Ytesting_data,svm_test_pred))
ls_training_acc.append(svm_score)
ls_testing_acc.append(accuracy_score(ML_Ytesting_data,svm_test_pred))

#AdaBoost--adab_
print("AdaBoost training")
AdaB = AdaBoostClassifier(learning_rate=1,n_estimators=600,algorithm="SAMME.R",random_state=42)
AdaB.fit(ML_Xtraining_data,ML_Ytraining_data)
adab_score=AdaB.score(ML_Xtraining_data,ML_Ytraining_data)
print("adab_训练集准确率得分：",adab_score)
adab_test_pred=AdaB.predict(ML_Xtesting_data)
adab_test_pred_proba=AdaB.predict_proba(ML_Xtesting_data)
print("finished AdaBoost training")
print("AdaB_testing_acc:{:.4f}".format(accuracy_score(ML_Ytesting_data,adab_test_pred)))
print("adab_F1_Score:{:.4f}".format(f1_score(ML_Ytesting_data,adab_test_pred)))
ls_F1_Methods.append("AdaBoost")
ls_F1_Score.append(f1_score(ML_Ytesting_data,adab_test_pred))
ls_training_acc.append(adab_score)
ls_testing_acc.append(accuracy_score(ML_Ytesting_data,adab_test_pred))

print("methods",ls_F1_Methods)
print("F1-score",ls_F1_Score)
print("training_acc",ls_training_acc)
print("testing_acc",ls_testing_acc)
#AUC曲线--分别计算每个机器学习模型的TPR，FPR，AUC
#GBDT--gbdt_
gbdt_fpr=[]
gbdt_tpr=[]
gbdt_fpr,gbdt_tpr,thresholds = metrics.roc_curve(ML_Ytesting_data,gbdt_test_pred_proba[:,1])
np.savetxt("GBDT_fpr.txt",gbdt_fpr)
np.savetxt("GBDT_tpr.txt",gbdt_tpr)
gbdt_roc_auc=metrics.auc(gbdt_fpr,gbdt_tpr)

#LR--lr_
lr_fpr=[]
lr_tpr=[]
lr_fpr,lr_tpr,thresholds = metrics.roc_curve(ML_Ytesting_data,lr_test_pred_proba[:,1])
np.savetxt("LR_fpr.txt",lr_fpr)
np.savetxt("LR_tpr.txt",lr_tpr)
lr_roc_auc=metrics.auc(lr_fpr,lr_tpr)

#RF--rf_
rf_fpr=[]
rf_tpr=[]
rf_fpr,rf_tpr,thresholds = metrics.roc_curve(ML_Ytesting_data,rf_test_pred_proba[:,1])
np.savetxt("RF_fpr.txt",rf_fpr)
np.savetxt("RF_tpr.txt",rf_tpr)
rf_roc_auc=metrics.auc(rf_fpr,rf_tpr)

#XGBoost--xgb_
xgb_fpr=[]
xgb_tpr=[]
xgb_fpr,xgb_tpr,thresholds = metrics.roc_curve(ML_Ytesting_data,xgb_test_pred_proba[:,1])
np.savetxt("XGBoost_fpr.txt",xgb_fpr)
np.savetxt("XGBoost_tpr.txt",xgb_tpr)
xgb_roc_auc=metrics.auc(xgb_fpr,xgb_tpr)

#SVM--svm_
svm_fpr=[]
svm_tpr=[]
svm_fpr,svm_tpr,thresholds = metrics.roc_curve(ML_Ytesting_data,svm_test_pred_proba[:,1])
np.savetxt("SVM_fpr.txt",svm_fpr)
np.savetxt("SVM_tpr.txt",svm_tpr)
svm_roc_auc=metrics.auc(svm_fpr,svm_tpr)

#AdaBoost--adab_
adab_fpr=[]
adab_tpr=[]
adab_fpr,adab_tpr,thresholds = metrics.roc_curve(ML_Ytesting_data,adab_test_pred_proba[:,1])
np.savetxt("AdaBoost_fpr.txt",adab_fpr)
np.savetxt("AdaBoost_tpr.txt",adab_tpr)
adab_roc_auc=metrics.auc(adab_fpr,adab_tpr)

#画AUC曲线--根据计算得到的每个机器学习模型的TPR，FPR，AUC
plt.figure(0).clf()
lw=2
##GBDT--gbdt_
plt.plot(gbdt_fpr,gbdt_tpr,color="r",linestyle="--",lw=lw,label='gbdt_ROC curve(area=%0.4f)'%gbdt_roc_auc)
#plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
##LR--lr_
plt.plot(lr_fpr,lr_tpr,color="g",linestyle="--",lw=lw,label='lr_ROC curve(area=%0.4f)'%lr_roc_auc)
##RF--rf_
plt.plot(rf_fpr,rf_tpr,color="b",linestyle="--",lw=lw,label='rf_ROC curve(area=%0.4f)'%rf_roc_auc)
##XGBoost--xgb_
plt.plot(xgb_fpr,xgb_tpr,color="c",linestyle="--",lw=lw,label='xgb_ROC curve(area=%0.4f)'%xgb_roc_auc)
##SVM--svm_
plt.plot(svm_fpr,svm_tpr,color="m",linestyle="--",lw=lw,label='svm_ROC curve(area=%0.4f)'%svm_roc_auc)
#AdaBoost--adab_
plt.plot(adab_fpr,adab_tpr,color="y",linestyle="--",lw=lw,label='adab_ROC curve(area=%0.4f)'%adab_roc_auc)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.savefig("Firgue_GM12878_AUC.svg")
plt.show()

#绘制P-R曲线--计算每个ML模型的precision和recall以及p_r值
##GBDT--gbdt_
gbdt_precision=[]
gbdt_recall=[]
gbdt_precision,gbdt_recall,thresholds_pr=precision_recall_curve(ML_Ytesting_data,gbdt_test_pred_proba[:,1])
np.savetxt("GBDT_precision.txt",gbdt_precision)
np.savetxt("GBDT_recall.txt",gbdt_recall)
gbdt_p_r=metrics.auc(gbdt_recall,gbdt_precision)

##LR--lr_
lr_precision=[]
lr_recall=[]
lr_precision,lr_recall,thresholds_pr=precision_recall_curve(ML_Ytesting_data,lr_test_pred_proba[:,1])
np.savetxt("LR_precision.txt",lr_precision)
np.savetxt("LR_recall.txt",lr_recall)
lr_p_r=metrics.auc(lr_recall,lr_precision)
##RF--rf_
rf_precision=[]
rf_recall=[]
rf_precision,rf_recall,thresholds_pr=precision_recall_curve(ML_Ytesting_data,rf_test_pred_proba[:,1])
np.savetxt("RF_precision.txt",rf_precision)
np.savetxt("RF_recall.txt",rf_recall)
rf_p_r=metrics.auc(rf_recall,rf_precision)

##XGBoost--xgb_
xgb_precision=[]
xgb_recall=[]
xgb_precision,xgb_recall,thresholds_pr=precision_recall_curve(ML_Ytesting_data,xgb_test_pred_proba[:,1])
np.savetxt("XGBoost_precision.txt",xgb_precision)
np.savetxt("XGBoost_recall.txt",xgb_recall)
xgb_p_r=metrics.auc(xgb_recall,xgb_precision)

##SVM--svm
svm_precision=[]
svm_recall=[]
svm_precision,svm_recall,thresholds_pr=precision_recall_curve(ML_Ytesting_data,svm_test_pred_proba[:,1])
np.savetxt("SVM_precision.txt",svm_precision)
np.savetxt("SVM_recall.txt",svm_recall)
svm_p_r=metrics.auc(svm_recall,svm_precision)

#AdaBoost--adab_
adab_precision=[]
adab_recall=[]
adab_precision,adab_recall,thresholds_pr=precision_recall_curve(ML_Ytesting_data,adab_test_pred_proba[:,1])
np.savetxt("AdaBoost_precision.txt",adab_precision)
np.savetxt("AdaBoost_recall.txt",adab_recall)
adab_p_r=metrics.auc(adab_recall,adab_precision)

plt.figure(0).clf()
##GBDT--gbdt_
plt.plot(gbdt_recall,gbdt_precision,color="r",linestyle="--",lw=lw,label='gbdt_P_R curve(area=%0.4f)'%gbdt_p_r)
##LR--lr_
plt.plot(lr_recall,lr_precision,color="g",linestyle="--",lw=lw,label='lr_P_R curve(area=%0.4f)'%lr_p_r)
##RF--rf_
plt.plot(rf_recall,rf_precision,color="b",linestyle="--",lw=lw,label='rf_P_R curve(area=%0.4f)'%rf_p_r)
##XGBoost--xgb_
plt.plot(xgb_recall,xgb_precision,color="c",linestyle="--",lw=lw,label='xgb_P_R curve(area=%0.4f)'%xgb_p_r)
##SVM--svm_
plt.plot(svm_recall,svm_precision,color="m",linestyle="--",lw=lw,label='svm_P_R curve(area=%0.4f)'%svm_p_r)
#AdaBoost--adab_
plt.plot(adab_recall,adab_precision,color="y",linestyle="--",lw=lw,label='adab_P_R curve(area=%0.4f)'%adab_p_r)
plt.title("Precision/Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower right")
plt.savefig("Firgue_GM12878_AUPRC.svg")
plt.show()
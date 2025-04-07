import numpy as np
import pandas as pd
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
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix,f1_score,precision_recall_curve
import matplotlib.pyplot as plt
from pandas import DataFrame as df

#加载训练集合和测试集--ML_Xtraining_data,ML_Ytraining_data;ML_Xtesting_data,ML_Ytesting_data
Enhancer_Promoter_Histone_fetures=np.loadtxt("Enhancer_Promoter_Histone_fetures.txt")
#加载分好的训练集和测试集
ls_ML_Xtraining_data=[]
ls_ML_Ytraining_data=[]
Model_K562_training_pairs=pd.read_excel("ALL_Method_training.xlsx",header=None)
print(Model_K562_training_pairs.head())
print(Model_K562_training_pairs.values)
for i in range(Model_K562_training_pairs.values.shape[0]):
    ls_ML_Xtraining_data.append(Enhancer_Promoter_Histone_fetures[Model_K562_training_pairs.values[i][0]])
    ls_ML_Ytraining_data.append(Model_K562_training_pairs.values[i][1])
print(len(ls_ML_Xtraining_data))
print(len(ls_ML_Ytraining_data))
# print(ls_ML_Xtraining_data)
# print(ls_ML_Ytraining_data)
ls_ML_Xtesting_data=[]
ls_ML_Ytesting_data=[]
Model_K562_testing_pairs=pd.read_excel("ALL_Method_testing.xlsx",header=None)
print(Model_K562_testing_pairs.head())
for i in range(Model_K562_testing_pairs.values.shape[0]):
    ls_ML_Xtesting_data.append(Enhancer_Promoter_Histone_fetures[Model_K562_testing_pairs.values[i][0]])
    ls_ML_Ytesting_data.append(Model_K562_testing_pairs.values[i][1])
print(len(ls_ML_Xtesting_data))
print(len(ls_ML_Ytesting_data))
# print(ls_ML_Xtesting_data)
# print(ls_ML_Ytesting_data)
ML_Xtraining_data=df(ls_ML_Xtraining_data)
print("TX",ML_Xtraining_data)
ML_Ytraining_data=df(ls_ML_Ytraining_data)
print("TY",ML_Ytraining_data)
ML_Xtesting_data=df(ls_ML_Xtesting_data)
print(ML_Xtesting_data)
ML_Ytesting_data=df(ls_ML_Ytesting_data)
print(ML_Ytesting_data)
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
print("finished GBDT training ")
print("gbdt_F1_Score:{:.4f}".format(f1_score(ML_Ytesting_data,gbdt_test_pred)))

#LR--lr_
print("LogisticRegression training")
LR=LogisticRegression()
LR.fit(ML_Xtraining_data,ML_Ytraining_data)
lr_score=LR.score(ML_Xtraining_data,ML_Ytraining_data)
print("lr_训练集准确率得分：",lr_score)
lr_test_pred_proba=LR.predict_proba(ML_Xtesting_data)
lr_test_pred=LR.predict(ML_Xtesting_data)
print("finished LogisticRegression training ")
print("lr__F1_Score:{:.4f}".format(f1_score(ML_Ytesting_data,lr_test_pred)))

#RF--rf_
print("RandomForest training")
rf=RandomForestClassifier(n_estimators=50)
rf.fit(ML_Xtraining_data,ML_Ytraining_data)
rf_score=rf.score(ML_Xtraining_data,ML_Ytraining_data)
print("rf_训练集准确率得分：",rf_score)
rf_test_pred=rf.predict(ML_Xtesting_data)
rf_test_pred_proba=rf.predict_proba(ML_Xtesting_data)
print("finished RandomForest training")
print("rf__F1_Score:{:.4f}".format(f1_score(ML_Ytesting_data,rf_test_pred)))

#XGBoost--xgb_
print("XGBoost training")
XGB=xgb.XGBClassifier(objective='binary:logistic')
XGB.fit(ML_Xtraining_data,ML_Ytraining_data)
xgb_score=XGB.score(ML_Xtraining_data,ML_Ytraining_data)
print("xgb_训练集准确率得分：",xgb_score)
xgb_test_pred_proba=XGB.predict_proba(ML_Xtesting_data)
xgb_test_pred=XGB.predict(ML_Xtesting_data)
print("finished XGBoost training")
print("xgb_F1_Score:{:.4f}".format(f1_score(ML_Ytesting_data,xgb_test_pred)))

#SVM--svm_
print("SVM training")
svm=svm.SVC(C=1,gamma="scale",degree=3,decision_function_shape='ovr',max_iter=-1,kernel="rbf",probability=True)
svm.fit(ML_Xtraining_data,ML_Ytraining_data)
svm_score=svm.score(ML_Xtraining_data,ML_Ytraining_data)
print("svm_训练集准确率得分：",svm_score)
svm_test_pred_proba=svm.predict_proba(ML_Xtesting_data)
svm_test_pred=svm.predict(ML_Xtesting_data)
print("finished SVM training")
print("svm_F1_Score:{:.4f}".format(f1_score(ML_Ytesting_data,svm_test_pred)))

#AdaBoost--adab_
print("AdaBoost training")
AdaB = AdaBoostClassifier(learning_rate=1,n_estimators=600,algorithm="SAMME.R",random_state=42)
AdaB.fit(ML_Xtraining_data,ML_Ytraining_data)
adab_score=AdaB.score(ML_Xtraining_data,ML_Ytraining_data)
print("adab_训练集准确率得分：",adab_score)
adab_test_pred=AdaB.predict(ML_Xtesting_data)
adab_test_pred_proba=AdaB.predict_proba(ML_Xtesting_data)
print("finished AdaBoost training")
print("adab_F1_Score:{:.4f}".format(f1_score(ML_Ytesting_data,adab_test_pred)))

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
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
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
plt.show()
import numpy as np
from pandas import DataFrame as df
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix,f1_score,precision_recall_curve
import matplotlib.pyplot as plt

EPI_Enhancer=np.loadtxt("./data/K562_training_EPI_Enhancer_histonefeature_Matrix.txt")
EPI_Promoter=np.loadtxt("./data/K562_training_EPI_Promoter_histonefeature_Matrix.txt")
EPI_Enhancer_Promoter=np.concatenate((EPI_Enhancer,EPI_Promoter),axis=1)
EPI_Enhancer_Promoter=df(EPI_Enhancer_Promoter)

EPI_pairs = []
with open("./data/K562_training_pairs_DNase.bed")as f:
    for line in f:
        EPI_pairs.append(line.strip().split())

EPI_pairs=df(EPI_pairs,columns=['list','chr','p_start','p_stop','e_start','e_stop','labels'])
EPI_pairs=EPI_pairs.drop(columns=["list"],axis=1)
EPI_pairs=EPI_pairs.drop([0])
EPI_pairs=EPI_pairs.reset_index(drop=True)

EPI_Labels=EPI_pairs["labels"]
EPI_Labels=EPI_Labels.map({'1':1,"-1":0}).astype(int)

EPI_data=EPI_Enhancer_Promoter.join(EPI_Labels)



X_train,X_test,y_train,y_test=train_test_split(EPI_Enhancer_Promoter,EPI_Labels,test_size=0.3,random_state=42)
np.savetxt("./data/X_train.txt",X_train,delimiter=",")
np.savetxt("./data/y_train.txt",y_train,delimiter=",")
np.savetxt("./data/X_test.txt",X_test,delimiter=",")
np.savetxt("./data/y_test.txt",y_test,delimiter=",")
gbdt = GradientBoostingClassifier(loss='log_loss', learning_rate=0.1, n_estimators=50, subsample=1
                                  , min_samples_split=2, min_samples_leaf=1, max_depth=3
                                  , init=None, random_state=None, max_features=None
                                  , verbose=0, max_leaf_nodes=None, warm_start=False
                                  )
gbdt.fit(X_train,y_train)
test_pred_proba=gbdt.predict_proba(X_test)
test_pred=gbdt.predict(X_test)
#混淆矩阵
c_m=confusion_matrix(y_test,test_pred)
plt.matshow(c_m,cmap=plt.cm.Reds)
plt.colorbar()

for i in range(len(c_m)):
    for j in range(len(c_m)):
        plt.annotate(c_m[j,i],xy=(i,j),horizontalalignment='center',verticalalignment='center')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print("F1_Score:{:.4f}".format(f1_score(y_test,test_pred)))
#AUC曲线
fpr=[]
tpr=[]
fpr,tpr,thresholds = metrics.roc_curve(y_test,test_pred_proba[:,1])
roc_auc=metrics.auc(fpr,tpr)
plt.figure()
lw=2
plt.plot(fpr,tpr,lw=lw,label='ROC curve(area=%0.2f)'%roc_auc)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()

#绘制P-R曲线
plt.figure("P-R Curve")
precision=[]
recall=[]
precision,recall,thresholds_pr=precision_recall_curve(y_test,test_pred_proba[:,1])
p_r=metrics.auc(recall,precision)
plt.plot(recall,precision,label='P_R curve(area=%0.2f)'%p_r)
plt.title("Precision/Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower right")
plt.show()
#print(test_pred)

'''
print(X_train)
print(y_train)
'''

'''
print(EPI_Enhancer.shape)
print(EPI_Promoter.shape)
print(EPI_Enhancer_Promoter)
print(EPI_Labels)
print(EPI_data)
'''
# 延伸增强子坐标以统一增强子的强度
import pandas as pd
#增强子原始坐标
#,names=["chr","Enhancer_start","Enhancer_end","pair_rows","EPI_labels"],header=None
ENhancer_rawcor_aboutpairs=pd.read_excel("K562_training_Enhancercor12.xlsx",header=None,names=["chr","Enhancer_start","Enhancer_end","pair_rows","EPI_labels"])
print(ENhancer_rawcor_aboutpairs.head())
ENhancer_rawcor=ENhancer_rawcor_aboutpairs[["Enhancer_start","Enhancer_end"]]

print(ENhancer_rawcor.head())
#用原始数据坐标计算新的数据坐标def comnewcor(x,y)&enhancerlengh enlen
#bed文件的格式为从0开始，且不包含最后一个位置：个人理解左闭右开，form 0 start
def comnewcor(old_x,old_y,enlen):
    #原始序列中点坐标
    centercor=(old_x+old_y)/2
    newX=int(centercor-enlen/2)
    newY=int(centercor+enlen/2)
    return newX,newY
#暂定为enlen=1000
ls_tmp_cor=[]
enlen=1000
for i in range(ENhancer_rawcor.values.shape[0]):
    enhancernewX,enhancernewY=comnewcor(ENhancer_rawcor.values[i][0], ENhancer_rawcor.values[i][1], enlen)
    #print(enhancernewX,enhancernewY)
    #将输出保存到TXT文件中
    ls_tmp_cor.append([enhancernewX,enhancernewY])
# with open("Enhancercor_Extended.txt","w") as f:
#     for i in ls_tmp_cor:
#         f.write(str(i))
#写入csv
tmp=pd.DataFrame(data=ls_tmp_cor)
tmp.to_csv("Enhancercor_Extended.csv",header=None)
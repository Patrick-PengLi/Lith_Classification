# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 22:26:18 2023

@author: 86138
"""

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap
from tqdm import tnrange, tqdm_notebook  
import itertools
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.gridspec as gridspec


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(123456)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# import data
df = pd.read_csv('data.csv')

# data processing
train_top = 15242
train_bot = 19105
Depth_train = df['Depth'][train_top:train_bot].to_numpy()
GR_train = df['GR'][train_top:train_bot].to_numpy()
AF_train = df['AF90'][train_top:train_bot].to_numpy() 
HCAL_train = df['HCAL'][train_top:train_bot].to_numpy()
Rho_train = df['RHOZ'][train_top:train_bot].to_numpy()
DT_train = df['DT'][train_top:train_bot].to_numpy()
NPHI_train = df['NPHI'][train_top:train_bot].to_numpy()
PR_train = df['PR_FAST'][train_top:train_bot].to_numpy()
VpVs_train = df['VPVS_FAST'][train_top:train_bot].to_numpy()
LITH_CORE_train = df['LITH_CORE'][train_top:train_bot].to_numpy()

def shape_trans(x):
    x_use = x.reshape(len(x),1)
    return x_use

x_train0 = torch.from_numpy(shape_trans(Depth_train)).float()
x_train1 = torch.from_numpy(shape_trans(GR_train)).float()
x_train2 = torch.from_numpy(shape_trans(AF_train)).float()
x_train3 = torch.from_numpy(shape_trans(HCAL_train)).float()
x_train4 = torch.from_numpy(shape_trans(Rho_train)).float()
x_train5 = torch.from_numpy(shape_trans(DT_train)).float()
x_train6 = torch.from_numpy(shape_trans(NPHI_train)).float()
x_train7 = torch.from_numpy(shape_trans(PR_train)).float()
x_train8 = torch.from_numpy(shape_trans(VpVs_train)).float()
x_train = torch.cat((x_train1,x_train2,x_train3,x_train4,x_train5,x_train6,x_train7,x_train8),1)
y_train = torch.from_numpy(shape_trans(LITH_CORE_train)).float()

train = torch.cat((x_train0,x_train1,x_train2,x_train3,x_train4,x_train5,x_train6,x_train7,x_train8,y_train),1)
train = train[~torch.any(train.isnan(),dim=1)]

all_df=pd.DataFrame(train.numpy(),columns=['Depth','GR','AF90','HCAL','RHOZ','DT','NPHI','PR_FAST','VPVS_FAST','LITH_CORE'])
all_df =all_df.sort_values('Depth')
Facies = all_df['LITH_CORE'].to_numpy()

# best subset regression:
def Linear_reg(X,Y):
    #Fit linear regression model and return RSS and R squared values
    model_k = linear_model.LinearRegression(fit_intercept = True)
    model_k.fit(X,Y)
    RSS = mean_squared_error(Y,model_k.predict(X)) * len(Y)
    R_squared = model_k.score(X,Y)
    return RSS, R_squared

X_train_select = train[:,1:9]
Y_train_select = train[:,9:]

X_bst = pd.DataFrame(X_train_select.numpy())
X_bst.columns = ['GR','AF90','HCAL','RHOZ','DT','NPHI','PR_FAST','VPVS_FAST']
Y_bst = pd.DataFrame(Y_train_select.numpy())
Y_bst.columns = ['lith_core']
k = 8

RSS_value = []
R2_value = []
Feature_value = []
num_features = []

for i in tnrange(1,len(X_bst.columns) + 1):
    for multi_var in itertools.combinations(X_bst.columns,i):
        regsubsets = Linear_reg(X_bst[list(multi_var)],Y_bst)        #multi linear regression, y = b1x1 + b2x2 + … + bnxn + c.
        RSS_value.append(regsubsets[0])                 
        R2_value.append(regsubsets[1])
        Feature_value.append(multi_var)
        num_features.append(len(multi_var))   

df_bst = pd.DataFrame({'RSS': RSS_value,'R2':R2_value,'features':Feature_value,'num_features': num_features })

df_bst_min = df_bst[df_bst.groupby('num_features')['RSS'].transform(min) == df_bst['RSS']]
df_bst_max = df_bst[df_bst.groupby('num_features')['R2'].transform(max) == df_bst['R2']]
df_bst_max.head(8)

df_bst['min_RSS'] = df_bst.groupby('num_features')['RSS'].transform(min)
df_bst['max_R2'] = df_bst.groupby('num_features')['R2'].transform(max)
df_bst.head()

# confusion matrix
def plot_confusion_matrix(y_true,y_pred):
  labels = unique_labels(y_true)
  table = pd.DataFrame(confusion_matrix(y_true,y_pred,normalize='true'),
                       )
  ax = sns.heatmap(table,annot=True,fmt = '.2g',cmap = 'viridis')
  ax.set(xlabel="Predicted lithology", ylabel="True lithology")

  figure = ax.get_figure()    
  figure.savefig('ax_conf.tif', dpi=400)
  return ax

# Random forest with different input size
train_acc = []
test_acc = []
num_variables = []

# 8 variables
xtrain_tree8,xtest_tree8,ytrain_tree8,ytest_tree8 = train_test_split(X_bst,Y_bst,test_size=0.2,random_state=123456) 
clf_tree8 = RandomForestClassifier(criterion='entropy',random_state=123456,max_depth=10,min_samples_leaf=10,min_samples_split=5) 
clf_tree8 = clf_tree8.fit(xtrain_tree8,ytrain_tree8) 
y_pred_tree8 = clf_tree8.predict(xtest_tree8)
train_score8 = clf_tree8.score(xtrain_tree8,ytrain_tree8)
test_score8 = clf_tree8.score(xtest_tree8,ytest_tree8)   #Return the mean accuracy on the given test data and labels.

train_acc = np.append(train_acc,train_score8)
test_acc = np.append(test_acc,test_score8)
num_variables = np.append(num_variables,xtrain_tree8.shape[1])
print('train score:',train_score8.round(3))
print('test score:',test_score8.round(3))
# plot_confusion_matrix(ytest_tree8,y_pred_tree8)

# 7 variables
df_tree7 = pd.concat([pd.DataFrame(X_bst),pd.DataFrame(Y_bst)],axis=1) 
df_tree7 = df_tree7.drop(['NPHI','lith_core'], axis=1)
print(df_tree7)

xtrain_tree7,xtest_tree7,ytrain_tree7,ytest_tree7 = train_test_split(df_tree7,Y_bst,test_size=0.2,random_state=123456)
clf_tree7 = RandomForestClassifier(criterion='entropy',random_state=123456,max_depth=10,min_samples_leaf=10,min_samples_split=5,n_estimators=100) 
clf_tree7 = clf_tree7.fit(xtrain_tree7,ytrain_tree7) 
y_pred_tree7 = clf_tree7.predict(xtest_tree7)
train_score7 = clf_tree7.score(xtrain_tree7,ytrain_tree7)
test_score7 = clf_tree7.score(xtest_tree7,ytest_tree7) 

train_acc = np.append(train_acc,train_score7)
test_acc = np.append(test_acc,test_score7)
num_variables = np.append(num_variables,xtrain_tree7.shape[1])
print('train score:' ,train_score7.round(3))
print('test score:' ,test_score7.round(3))
# plot_confusion_matrix(ytest_tree7,y_pred_tree7)

# 6 variables
df_tree6 = pd.concat([pd.DataFrame(X_bst),pd.DataFrame(Y_bst)],axis=1) 
df_tree6 = df_tree6.drop(['AF90','NPHI','lith_core'], axis=1)
print(df_tree6)

xtrain_tree6,xtest_tree6,ytrain_tree6,ytest_tree6 = train_test_split(df_tree6,Y_bst,test_size=0.2,random_state=123456)
clf_tree6 = RandomForestClassifier(criterion='entropy',random_state=123456,max_depth=10,min_samples_leaf=10,min_samples_split=5,n_estimators=100) 
clf_tree6 = clf_tree6.fit(xtrain_tree6,ytrain_tree6) 
y_pred_tree6 = clf_tree6.predict(xtest_tree6)
train_score6 = clf_tree6.score(xtrain_tree6,ytrain_tree6)
test_score6 = clf_tree6.score(xtest_tree6,ytest_tree6) 

train_acc = np.append(train_acc,train_score6)
test_acc = np.append(test_acc,test_score6)
num_variables = np.append(num_variables,xtrain_tree6.shape[1])
print('train score:' ,train_score6.round(3))
print('test score:' ,test_score6.round(3))
# plot_confusion_matrix(ytest_tree6,y_pred_tree6)

# 5 variables
df_tree5 = pd.concat([pd.DataFrame(X_bst),pd.DataFrame(Y_bst)],axis=1) 
df_tree5 = df_tree5.drop(['AF90','HCAL','NPHI','lith_core'], axis=1)
print(df_tree5)

xtrain_tree5,xtest_tree5,ytrain_tree5,ytest_tree5 = train_test_split(df_tree5,Y_bst,test_size=0.2,random_state=123456)
clf_tree5 = RandomForestClassifier(criterion='entropy',random_state=123456,max_depth=10,min_samples_leaf=10,min_samples_split=5,n_estimators=100) 
clf_tree5 = clf_tree5.fit(xtrain_tree5,ytrain_tree5) 
y_pred_tree5 = clf_tree5.predict(xtest_tree5)
train_score5 = clf_tree5.score(xtrain_tree5,ytrain_tree5)
test_score5 = clf_tree5.score(xtest_tree5,ytest_tree5) 

train_acc = np.append(train_acc,train_score5)
test_acc = np.append(test_acc,test_score5)
num_variables = np.append(num_variables,xtrain_tree5.shape[1])
print('train score:',train_score5.round(3))
print('test score:',test_score5.round(3))
# plot_confusion_matrix(ytest_tree5,y_pred_tree5)

#4 variables
df_tree4 = pd.concat([pd.DataFrame(X_bst),pd.DataFrame(Y_bst)],axis=1) 
df_tree4 = df_tree4.drop(['AF90','NPHI','PR_FAST','VPVS_FAST','lith_core'], axis=1)
print(df_tree4)

xtrain_tree4,xtest_tree4,ytrain_tree4,ytest_tree4 = train_test_split(df_tree4,Y_bst,test_size=0.2,random_state=123456)
clf_tree4 = RandomForestClassifier(criterion='entropy',random_state=123456,max_depth=10,min_samples_leaf=10,min_samples_split=5,n_estimators=100) 
clf_tree4 = clf_tree4.fit(xtrain_tree4,ytrain_tree4) 
y_pred_tree4 = clf_tree4.predict(xtest_tree4)
train_score4 = clf_tree4.score(xtrain_tree4,ytrain_tree4)
test_score4 = clf_tree4.score(xtest_tree4,ytest_tree4) 

train_acc = np.append(train_acc,train_score4)
test_acc = np.append(test_acc,test_score4)
num_variables = np.append(num_variables,xtrain_tree4.shape[1])
print('train score:',train_score4.round(3))
print('test score:',test_score4.round(3))
# plot_confusion_matrix(ytest_tree4,y_pred_tree4)

#3 variables
df_tree3 = pd.concat([pd.DataFrame(X_bst),pd.DataFrame(Y_bst)],axis=1) 
df_tree3 = df_tree3.drop(['AF90','HCAL','NPHI','PR_FAST','VPVS_FAST','lith_core'], axis=1)
print(df_tree3)

xtrain_tree3,xtest_tree3,ytrain_tree3,ytest_tree3 = train_test_split(df_tree3,Y_bst,test_size=0.2,random_state=123456)
clf_tree3 = RandomForestClassifier(criterion='entropy',random_state=123456,max_depth=10,min_samples_leaf=10,min_samples_split=5,n_estimators=100) 
clf_tree3 = clf_tree3.fit(xtrain_tree3,ytrain_tree3) 
y_pred_tree3 = clf_tree3.predict(xtest_tree3)
train_score3 = clf_tree3.score(xtrain_tree3,ytrain_tree3)
test_score3 = clf_tree3.score(xtest_tree3,ytest_tree3) 

train_acc = np.append(train_acc,train_score3)
test_acc = np.append(test_acc,test_score3)
num_variables = np.append(num_variables,xtrain_tree3.shape[1])
print('train score:',train_score3.round(3))
print('test score:',test_score3.round(3))
# plot_confusion_matrix(ytest_tree3,y_pred_tree3)

#2 variables
df_tree2 = pd.concat([pd.DataFrame(X_bst),pd.DataFrame(Y_bst)],axis=1) 
df_tree2 = df_tree2.drop(['AF90','HCAL','NPHI','lith_core','GR','PR_FAST','VPVS_FAST'], axis=1)
print(df_tree2)

xtrain_tree2,xtest_tree2,ytrain_tree2,ytest_tree2 = train_test_split(df_tree2,Y_bst,test_size=0.2,random_state=123456)
clf_tree2 = RandomForestClassifier(criterion='entropy',random_state=123456,max_depth=10,min_samples_leaf=10,min_samples_split=5,n_estimators=100) 
clf_tree2 = clf_tree2.fit(xtrain_tree2,ytrain_tree2) 
y_pred_tree2 = clf_tree2.predict(xtest_tree2)
train_score2 = clf_tree2.score(xtrain_tree2,ytrain_tree2)
test_score2 = clf_tree2.score(xtest_tree2,ytest_tree2) 

train_acc = np.append(train_acc,train_score2)
test_acc = np.append(test_acc,test_score2)
num_variables = np.append(num_variables,xtrain_tree2.shape[1])
print('train score:', train_score2.round(2))
print('test score:',test_score2.round(2))
# plot_confusion_matrix(ytest_tree2,y_pred_tree2)

#1 variable
df_tree1 = pd.concat([pd.DataFrame(X_bst),pd.DataFrame(Y_bst)],axis=1) 
df_tree1 = df_tree1.drop(['AF90','HCAL','NPHI','lith_core','RHOZ','PR_FAST','VPVS_FAST','GR'], axis=1)
print(df_tree1)

xtrain_tree1,xtest_tree1,ytrain_tree1,ytest_tree1 = train_test_split(df_tree1,Y_bst,test_size=0.2,random_state=123456)
clf_tree1 = RandomForestClassifier(criterion='entropy',random_state=123456,max_depth=10,min_samples_leaf=10,min_samples_split=5,n_estimators=100) 
clf_tree1 = clf_tree1.fit(xtrain_tree1,ytrain_tree1) 
y_pred_tree1 = clf_tree1.predict(xtest_tree1)
train_score1 = clf_tree1.score(xtrain_tree1,ytrain_tree1)
test_score1 = clf_tree1.score(xtest_tree1,ytest_tree1) 

train_acc = np.append(train_acc,train_score1)
test_acc = np.append(test_acc,test_score1)
num_variables = np.append(num_variables,xtrain_tree1.shape[1])
print('train score:',train_score1.round(3))
print('test score:',test_score1.round(3))
# plot_confusion_matrix(ytest_tree1,y_pred_tree1)


# Plotting
nw =40
Facies_true=np.repeat(np.expand_dims(Facies,1), nw, 1)
cmap = ListedColormap(['#bebebe','#7cfc00', '#ffff00','#80ffff','#8080ff','#ef138a'])

fs=12
lw=1.5

color=['#bebebe','#bebebe','#7cfc00','#7cfc00','#ffff00','#ffff00','#80ffff','#80ffff','#8080ff','#8080ff','#ef138a','#ef138a']
fig1, ax = plt.subplots(figsize=(12,6), facecolor='w')
cnts, values, bars = ax.hist(train[:,9], edgecolor='k')
for i, (cnt, value, bar) in enumerate(zip(cnts, values, bars)):
    bar.set_facecolor(color[i % len(color)])
plt.xticks([])
plt.xlabel('Lithology',fontsize=fs+4)
plt.ylabel('Frequency',fontsize=fs+4)
fig1.tight_layout()
plt.show()
# fig1.savefig('Hist.tiff',dpi=400)

fig2 = plt.figure(figsize = (15,5))
plt.subplot(1, 2, 1)
plt.scatter(df_bst.num_features,df_bst.RSS,  color = 'lightblue' )
plt.xlabel('Subset size',fontsize=fs)
plt.ylabel('$RSS$',fontsize=fs)
plt.plot(df_bst.num_features,df_bst.min_RSS,'^-',color = 'k', label = 'Best subset',markersize =fs-2)
plt.legend(fontsize = fs)
plt.xticks(fontsize = fs-2)
plt.yticks(fontsize = fs-2)
plt.grid()
plt.subplot(1, 2, 2)
plt.scatter(df_bst.num_features,df_bst.R2,  color = 'lightblue' )
plt.plot(df_bst.num_features,df_bst.max_R2,'o-',color = 'k', label = 'Best subset',markersize =fs-2)
plt.xlabel('Subset size',fontsize=fs)
plt.ylabel('$R^2$',fontsize = fs)
plt.legend(fontsize = fs)
plt.xticks(fontsize = fs-2)
plt.yticks(fontsize = fs-2)
plt.grid()
fig2.tight_layout()
plt.show()
# fig2.savefig('bestsub.tiff',dpi=400)

fig3=plt.figure(figsize=(8,5))
plt.plot(num_variables,train_acc,marker='v',color = 'b',label ='Train score',linewidth =lw)
plt.plot(num_variables,test_acc,marker='d',color = 'r',label ='Test score',linewidth =lw)
plt.grid(True, linestyle='--')
plt.ylim([0.6,0.95])
plt.legend(fontsize =fs)
plt.xlabel('Number of Variables',fontsize=fs)
plt.ylabel('Accuracy score',fontsize =fs)
plt.tight_layout
# fig3.savefig('Accuracy.tiff',dpi=400)

Facies_df = pd.DataFrame()

Facies_df['lith_core'] = ytest_tree1
Facies_df['pred1'] = y_pred_tree1.tolist()
Facies_df['pred2'] = y_pred_tree2.tolist()
Facies_df['pred3'] = y_pred_tree3.tolist()
Facies_df['pred4'] = y_pred_tree4.tolist()
Facies_df['pred5'] = y_pred_tree5.tolist()
Facies_df['pred6'] = y_pred_tree6.tolist()
Facies_df['pred7'] = y_pred_tree7.tolist()
Facies_df['pred8'] = y_pred_tree8.tolist()
Facies_df = Facies_df.sort_index()

nw =12
true_Facies = np.repeat(np.expand_dims(Facies_df['lith_core'],1), nw, 1)
pred_var1=np.repeat(np.expand_dims(Facies_df['pred1'],1), nw, 1)
pred_var2=np.repeat(np.expand_dims(Facies_df['pred2'],1), nw, 1)
pred_var3=np.repeat(np.expand_dims(Facies_df['pred3'],1), nw, 1)
pred_var4=np.repeat(np.expand_dims(Facies_df['pred4'],1), nw, 1)
pred_var5=np.repeat(np.expand_dims(Facies_df['pred5'],1), nw, 1)
pred_var6=np.repeat(np.expand_dims(Facies_df['pred6'],1), nw, 1)
pred_var7=np.repeat(np.expand_dims(Facies_df['pred7'],1), nw, 1)
pred_var8=np.repeat(np.expand_dims(Facies_df['pred8'],1), nw, 1)


fig4 =plt.figure(figsize=(16,8))
gs=gridspec.GridSpec(2,9,height_ratios=[2,0.6])
ax1=fig4.add_subplot(gs[0,0])
ax1.imshow(true_Facies,cmap = cmap, vmin=0, vmax=5,)
ax1.yaxis.set_tick_params(labelsize=fs-2)
plt.xticks([])
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = lw
plt.xlabel('True Facies',fontsize = fs)
plt.ylabel('Core samples',fontsize = fs)

ax2=fig4.add_subplot(gs[0,1])
ax2.imshow(pred_var1,cmap = cmap, vmin=0, vmax=5,)
ax2.yaxis.set_tick_params(labelsize=fs-2)
plt.xticks([])
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = lw
plt.xlabel('(1)',fontsize = fs)

ax3=fig4.add_subplot(gs[0,2])
ax3.imshow(pred_var2,cmap = cmap, vmin=0, vmax=5,)
ax3.yaxis.set_tick_params(labelsize=fs-2)
plt.xticks([])
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = lw
plt.xlabel('(2)',fontsize = fs)

ax4=fig4.add_subplot(gs[0,3])
ax4.imshow(pred_var3,cmap = cmap, vmin=0, vmax=5,)
ax4.yaxis.set_tick_params(labelsize=fs-2)
plt.xticks([])
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = lw
plt.xlabel('(3)',fontsize = fs)

ax5=fig4.add_subplot(gs[0,4])
ax5.imshow(pred_var4,cmap = cmap, vmin=0, vmax=5,)
ax5.yaxis.set_tick_params(labelsize=fs-2)
plt.xticks([])
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = lw
plt.xlabel('(4)',fontsize = fs)

ax6=fig4.add_subplot(gs[0,5])
ax6.imshow(pred_var5,cmap = cmap, vmin=0, vmax=5,)
ax6.yaxis.set_tick_params(labelsize=fs-2)
plt.xticks([])
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = lw
plt.xlabel('(5)',fontsize = fs)

ax7=fig4.add_subplot(gs[0,6])
ax7.imshow(pred_var6,cmap = cmap, vmin=0, vmax=5,)
ax7.yaxis.set_tick_params(labelsize=fs-2)
plt.xticks([])
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = lw
plt.xlabel('(6)',fontsize = fs)

ax8=fig4.add_subplot(gs[0,7])
ax8.imshow(pred_var7,cmap = cmap, vmin=0, vmax=5,)
ax8.yaxis.set_tick_params(labelsize=fs-2)
plt.xticks([])
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = lw
plt.xlabel('(7)',fontsize = fs)

ax9=fig4.add_subplot(gs[0,8])
ax9.imshow(pred_var8,cmap = cmap, vmin=0, vmax=5,)
ax9.yaxis.set_tick_params(labelsize=fs-2)
plt.xticks([])
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = lw
plt.xlabel('(8)',fontsize = fs)

ax10=fig4.add_subplot(gs[1,:])
xx=ax10.imshow(pred_var8,cmap = cmap, vmin=0, vmax=5,)
plt.gca().set_visible(False)
# cax = fig.add_axes([0.3, 1, 0.5, 1]) ,cax=cax
cbar = plt.colorbar(xx,orientation="horizontal")
cbar.set_ticklabels(['Shale','Siltstone','Sandstone','Carbonate','Anhydrite','Bentonite'])
cbar.ax.tick_params(labelsize=fs-4)

fig4.tight_layout()
plt.show()
# fig4.savefig('predicted.tiff',dpi=400)

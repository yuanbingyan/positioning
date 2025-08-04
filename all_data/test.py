import pandas as pd
import numpy as np
import sklearn.model_selection

from sklearn.model_selection import train_test_split
from All_data.Super_parameter_lat import get_parameter_lat
from All_data.Super_parameter_lng import get_parameter_lng
from All_data.多项式回归_lng import polynomial_regression_lng
from All_data.多项式回归_lat import polynomial_regression_lat
from All_data.adaboost import adaboost_main
from sklearn.metrics import mean_squared_error
from All_data.data_op import data_op_main
def rad(d):
    return d * np.pi / 180.0
def distance(lng1, lng2, lat1, lat2):
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * np.arcsin(np.sqrt(pow(np.sin(a / 2), 2) + np.cos(radLat1) * np.cos(radLat2) * pow(np.sin(b / 2), 2)))
    s = s * 6378.137 * 1000
    return s
def train(): ## 训练模型
    limli_scale_data=pd.read_csv('计算覆盖距离.csv',encoding='ANSI')
    data_with_GPS = pd.read_excel('华为已处理数据.xlsx')
    for index,row in limli_scale_data.iterrows():
        lte_id = row['lte_id']
        limli_scale = row['覆盖范围']
        if limli_scale<=500:
            limli_scale = 500
        BS_lng = data_with_GPS[data_with_GPS['lte_id'] == lte_id].经度.iloc[0]
        BS_lat = data_with_GPS[data_with_GPS['lte_id'] == lte_id].维度.iloc[0]
        lte_id_data = data_with_GPS[data_with_GPS['lte_id'] == lte_id][['ltescrsrp','ltescrsrq','ltescphr','ltencrsrp_1','ltencrsrq_1','ltencpci_1','ltencrsrp_2','ltencrsrq_2','ltencpci_2','longitude','latitude']]
        lte_id_data = np.array(lte_id_data)
        ###
        s = distance(BS_lng, lte_id_data[:,9], BS_lat, lte_id_data[:,10])
        num = []
        for j in range(len(s)):
            if s[j] > limli_scale:
                num.append(j)
        # print('500米范围占',1-len(num)/len(s))
        lte_id_data = np.delete(lte_id_data, num, axis=0)

        ###
        train_x, test_x, train_y, test_y = train_test_split(lte_id_data[:,0:9], lte_id_data[:,9:], test_size=0.2)
        print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
        lte_id_data = np.concatenate((train_x,train_y),axis=1)
        test_data  = np.concatenate((test_x,test_y),axis=1)
        np.savez('data/{}.npz'.format(lte_id),lte_id_data,test_data)
        print(lte_id_data.shape)
        # 获取超参数
        parameter_lat = get_parameter_lat(lte_id_data,BS_lng,BS_lat,limli_scale)
        parameter_lng = get_parameter_lng(lte_id_data,BS_lng,BS_lat,limli_scale)
        # 多项式回归法训练
        polynomial_regression_lng(lte_id_data,BS_lng,BS_lat,limli_scale,parameter_lng,lte_id)
        polynomial_regression_lat(lte_id_data,BS_lng,BS_lat,limli_scale,parameter_lat,lte_id)
        # adaboost 训练
        adaboost_main(lte_id_data,BS_lng,BS_lat,limli_scale,lte_id)
        print(lte_id)

def pred():  # 预测数据
    # data = pd.read_csv('没有经纬度的华为数据.csv', encoding='ANSI')
    # 准备格子
    grid = pd.read_csv('栅格的中心经纬度.csv',encoding='ANSI')
    grid_x = np.array(list(set(np.array(grid)[:,0])))
    grid_x.sort()
    grid_y = np.array(list(set(np.array(grid)[:,1])))
    grid_y.sort()
    limli_scale_data = pd.read_csv('计算覆盖距离.csv', encoding='ANSI')
    # data_with_GPS = pd.read_excel('华为已处理数据.xlsx')
    s_ada_ls, s_poly_ls, s_weight_ls, s_ada_poly_ls,s_knn_ada_ls,s_knn_ls = [],[],[],[],[],[]
    s_xgboost_ls,s_ada_xgboost_ls,s_ada_GBDT_ls,s_ada_bagging_ls,s_ada_rangom_forest_ls,s_GBDT_ls = [],[],[],[],[],[]
    s_bagging_ls,s_random_forest_ls = [],[]
    for index, row in limli_scale_data.iterrows():
        ## 准备训练数据
        lte_id = row['lte_id']
        data = np.load('data/{}.npz'.format(lte_id))
        lte_id_data = data['arr_0']
        # 准备预测数据
        predict_x = data['arr_1']
        y_lng_pred,y_lat_pred,s_ada,s_poly,s_weight,s_ada_poly,s_knn_ada,s_knn,s_xgboost,s_ada_xgboost,s_ada_GBDT,s_ada_bagging,s_ada_rangom_forest,s_GBDT,s_bagging,s_rangom_forest\
            = data_op_main(predict_x,lte_id_data,lte_id,grid_x,grid_y)
        s_ada_ls.append(s_ada)
        s_poly_ls.append(s_poly)
        s_weight_ls.append(s_weight)
        s_ada_poly_ls.append(s_ada_poly)
        s_knn_ada_ls.append(s_knn_ada)
        s_knn_ls.append(s_knn)
        s_xgboost_ls.append(s_xgboost)
        s_ada_xgboost_ls.append(s_ada_xgboost)
        s_ada_GBDT_ls.append(s_ada_GBDT)
        s_ada_bagging_ls.append(s_ada_bagging)
        s_ada_rangom_forest_ls.append(s_ada_rangom_forest)
        s_GBDT_ls.append(s_GBDT)
        s_bagging_ls.append(s_bagging)
        s_random_forest_ls.append(s_rangom_forest)
    print('=============================')
    print('ada',np.mean(s_ada_ls))
    print('poly',np.mean(s_poly_ls))
    print('weight',np.mean(s_weight_ls))
    print('ada_poly',np.mean(s_ada_poly_ls))
    print('ada_knn',np.mean(s_knn_ada_ls))
    print('knn',np.mean(s_knn_ls))
    print('xgboost',np.mean(s_xgboost_ls))
    print('ada_xgboost',np.mean(s_ada_xgboost_ls))
    print('ada_GBDT',np.mean(s_ada_GBDT_ls))
    print('ada_bagging',np.mean(s_ada_bagging_ls))
    print('ada_random_forest',np.mean(s_ada_rangom_forest_ls))
    print('GBDT',np.mean(s_GBDT_ls))
    print('bagging',np.mean(s_bagging_ls))
    print('random_forest',np.mean(s_random_forest_ls))
    # print('s_ada:',np.mean(s_ada_ls),'s_poly:',np.mean(s_poly_ls),'s_wight:',np.mean(s_weight_ls),'s_ada+poly:',np.mean(s_ada_poly_ls),)
    # print('ada',np.mean(s_ada_ls))
    # print('bagging',np.mean(s_bagging_ls))
    # print('random_forest',np.mean(s_random_forest_ls))
    # print('GBDT', np.mean(s_GBDT_ls))
    # print('ada_knn',np.mean(s_knn_ada_ls))
    # print('knn',np.mean(s_knn_ls))
    # print('xgboost',np.mean(s_xgboost_ls))
    # print('ada_xgboost',np.mean(s_ada_xgboost_ls))
    # print('ada_GBDT',np.mean(s_ada_GBDT_ls))
    # print('ada_bagging',np.mean(s_ada_bagging_ls))
    # print('ada_random_forest',np.mean(s_ada_rangom_forest_ls))
# a = np.load('data/183734-11.npz')
# print(a['arr_0'].shape,a['arr_1'].shape)
# train()
pred()

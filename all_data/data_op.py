import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
### 给出经纬度计算距离
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
k = 1
knn = KNeighborsRegressor(k)
###
xgboost_demo = xgb.XGBRegressor( max_depth=9,
                        learning_rate=0.1,
                        n_estimators=150,
                        min_child_weight=1,
                        subsample=0.8,
                        reg_alpha=0,
                        reg_lambda=1,
            gamma=0,)
gbm0 = GradientBoostingRegressor(n_estimators=200,learning_rate=0.1)
bagging = BaggingRegressor(base_estimator=DecisionTreeRegressor(criterion='friedman_mse',splitter='random'),
                       n_estimators=500, max_samples=1.0,
                        max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1, random_state=1)
rangom_forest = RandomForestRegressor(n_estimators=300,n_jobs=-1)

def data_op_main(predict_x,train_data,lte_id,grid_x,grid_y):
    predict_x = predict_x.T
    predict_y = predict_x[9:,]
    predict_x = predict_x[0:9,].T  ## 准备需要预测的数据
    train_data = np.array(train_data).T #(25721,19)
    train_x = train_data[0:9, ].T
    train_y = train_data[9:,]
    train_y_lng = train_y[0,:]
    train_y_lat = train_y[1,:]
    ### knn
    knn.fit(train_x, train_y_lng)
    knn_lng = knn.predict(predict_x)
    knn.fit(train_x, train_y_lat)
    knn_lat = knn.predict(predict_x)
    s_knn = distance(knn_lng, predict_y[0, :], knn_lat, predict_y[1, :])

    ### xgboost
    xgboost_demo.fit(train_x, train_y_lng)
    xgboost_lng = xgboost_demo.predict(predict_x)
    xgboost_demo.fit(train_x, train_y_lat)
    xgboost_lat = xgboost_demo.predict(predict_x)
    s_xgboost = distance(xgboost_lng, predict_y[0, :], xgboost_lat, predict_y[1, :])
    ###
    ### GBDT
    gbm0.fit(train_x, train_y_lng)
    GBDT_lng = gbm0.predict(predict_x)
    gbm0.fit(train_x, train_y_lat)
    GBDT_lat = gbm0.predict(predict_x)
    s_GBDT = distance(GBDT_lng, predict_y[0, :], GBDT_lat, predict_y[1, :])
    ###
    ### bagging
    bagging.fit(train_x, train_y_lng)
    bagging_lng = bagging.predict(predict_x)
    bagging.fit(train_x, train_y_lat)
    bagging_lat = bagging.predict(predict_x)
    s_bagging = distance(bagging_lng, predict_y[0, :], bagging_lat, predict_y[1, :])
    ###
    ###random_forest
    rangom_forest.fit(train_x, train_y_lng)
    rangom_forest_lng = rangom_forest.predict(predict_x)
    rangom_forest.fit(train_x, train_y_lat)
    rangom_forest_lat = rangom_forest.predict(predict_x)
    s_rangom_forest = distance(rangom_forest_lng, predict_y[0, :], rangom_forest_lat, predict_y[1, :])
    ###
    xx, yy = np.meshgrid(grid_x, grid_y)
    xxx, yyy = xx.flatten(), yy.flatten()
    pos = {}
    for i,j in zip(xxx,yyy):
        pos[(i,j)] = 0
    ##### 把训练数据对应到网格
    for i in range(len(train_y_lat)):
        form_lng = grid_x[np.argmin(np.abs(train_y_lng[i]-grid_x))]
        form_lat = grid_y[np.argmin(np.abs(train_y_lat[i] - grid_y))]
        train_y_lng[i] = form_lng
        train_y_lat[i] = form_lat
        pos[(form_lng,form_lat)] = 1
    ## 纬度预测
    (limli_scale,BS_lng,BS_lat,scaler_lat,poly_reg_lat,lin_reg_2_lat) = joblib.load('polynomial_regression_model/'+lte_id+'_lat.pkl')
    y_lat_pred = lin_reg_2_lat.predict(poly_reg_lat.fit_transform(scaler_lat.fit_transform(predict_x)))
    ## 经度预测
    (scaler_lng,poly_reg_lng,lin_reg_2_lng) = joblib.load('polynomial_regression_model/'+lte_id+'_lng.pkl')
    y_lng_pred = lin_reg_2_lng.predict(poly_reg_lng.fit_transform(scaler_lng.fit_transform(predict_x)))
    s_poly = distance(y_lng_pred,predict_y[0,:],y_lat_pred,predict_y[1,:])

    ###
    (scaler_adaboost, bdt_lng, bdt_lat) = joblib.load('adaboost_model/' + lte_id + '.pkl')
    adaboost_x = predict_x
    adaboost_x = scaler_adaboost.transform(adaboost_x)
    ada_lng = bdt_lng.predict(adaboost_x)
    train_x_addFeature = ada_lng.reshape(adaboost_x.shape[0], 1)
    train_x_addFeature = np.hstack((adaboost_x, train_x_addFeature))
    ada_lat = bdt_lat.predict(train_x_addFeature)
    s_ada = distance(ada_lng, predict_y[0, :], ada_lat, predict_y[1, :])

    w = 0.8
    s_weight = distance(w * ada_lng+(1-w)*y_lng_pred, predict_y[0, :], w*ada_lat +(1-w)*y_lat_pred, predict_y[1, :])
    s_knn_ada = distance(w * ada_lng+(1-w)*knn_lng, predict_y[0, :], w*ada_lat +(1-w)*knn_lat, predict_y[1, :])
    s_ada_xgboost = distance(w * ada_lng+(1-w)*xgboost_lng, predict_y[0, :], w*ada_lat +(1-w)*xgboost_lat, predict_y[1, :])
    s_ada_GBDT = distance(w * ada_lng+(1-w)*GBDT_lng, predict_y[0, :], w*ada_lat +(1-w)*GBDT_lat, predict_y[1, :])
    s_ada_bagging = distance(w * ada_lng+(1-w)*bagging_lng, predict_y[0, :], w*ada_lat +(1-w)*bagging_lat, predict_y[1, :])
    s_ada_rangom_forest = distance(w * ada_lng+(1-w)*rangom_forest_lng, predict_y[0, :], w*ada_lat +(1-w)*rangom_forest_lat, predict_y[1, :])
    ###
    num = []
    for i in range(len(y_lng_pred)):
        form_lng = grid_x[np.argmin(np.abs(y_lng_pred[i]-grid_x))]
        form_lat = grid_y[np.argmin(np.abs(y_lat_pred[i] - grid_y))]
        if distance(y_lng_pred[i],BS_lng,y_lat_pred[i],BS_lat)>limli_scale or pos[(form_lng,form_lat)] == 1:
            num.append(i)
    if len(num)!=0:
        (scaler_adaboost, bdt_lng, bdt_lat) = joblib.load('adaboost_model/'+lte_id+'.pkl')
        adaboost_x = predict_x[num,:]
        adaboost_x = scaler_adaboost.transform(adaboost_x)
        ada_lng = bdt_lng.predict(adaboost_x)
        train_x_addFeature = ada_lng.reshape(adaboost_x.shape[0],1)
        train_x_addFeature = np.hstack((adaboost_x,train_x_addFeature))
        ada_lat = bdt_lat.predict(train_x_addFeature)
        y_lng_pred[num] = ada_lng
        y_lat_pred[num] = ada_lat
    s_ada_poly = distance(y_lng_pred, predict_y[0, :], y_lat_pred, predict_y[1, :])
    s_ada = np.mean(s_ada)
    s_poly= np.mean(s_poly)
    s_weight= np.mean(s_weight)
    s_ada_poly= np.mean(s_ada_poly)
    s_knn_ada= np.mean(s_knn_ada)
    s_ada_xgboost = np.mean(s_ada_xgboost)
    s_ada_GBDT = np.mean(s_ada_GBDT)
    s_ada_bagging = np.mean(s_ada_bagging)
    s_ada_rangom_forest = np.mean(s_ada_rangom_forest)
    s_knn= np.mean(s_knn)
    s_xgboost = np.mean(s_xgboost)
    s_GBDT = np.mean(s_GBDT)
    s_bagging = np.mean(s_bagging)
    s_rangom_forest = np.mean(s_rangom_forest)
    # s_ada = np.sqrt(mean_squared_error(s_ada,np.array([0]*len(s_ada))))
    # s_poly= np.sqrt(mean_squared_error(s_poly,np.array([0]*len(s_poly))))
    # s_weight= np.sqrt(mean_squared_error(s_weight,np.array([0]*len(s_weight))))
    # s_ada_poly= np.sqrt(mean_squared_error(s_ada_poly,np.array([0]*len(s_ada_poly))))
    # s_knn_ada= np.sqrt(mean_squared_error(s_knn_ada,np.array([0]*len(s_knn_ada))))
    # s_ada_xgboost = np.sqrt(mean_squared_error(s_ada_xgboost,np.array([0]*len(s_ada_xgboost))))
    # s_ada_GBDT = np.sqrt(mean_squared_error(s_ada_GBDT,np.array([0]*len(s_ada_GBDT))))
    # s_ada_bagging = np.sqrt(mean_squared_error(s_ada_bagging,np.array([0]*len(s_ada_bagging))))
    # s_ada_rangom_forest = np.sqrt(mean_squared_error(s_ada_rangom_forest,np.array([0]*len(s_ada_rangom_forest))))
    # s_knn= np.sqrt(mean_squared_error(s_knn,np.array([0]*len(s_knn))))
    # s_xgboost = np.sqrt(mean_squared_error(s_xgboost,np.array([0]*len(s_xgboost))))
    # s_GBDT = np.sqrt(mean_squared_error(s_GBDT,np.array([0]*len(s_GBDT))))
    # s_bagging = np.sqrt(mean_squared_error(s_bagging,np.array([0]*len(s_bagging))))
    # s_rangom_forest = np.sqrt(mean_squared_error(s_rangom_forest,np.array([0]*len(s_rangom_forest))))
    print('ada', s_ada)
    print('poly',s_poly)
    print('s_weight',s_weight)
    print('xgboost',s_xgboost)
    print('GBDT',s_GBDT)
    print('ada_xgboost',s_ada_xgboost)
    print('ada_bagging',s_ada_bagging)
    print('ada_random_forest',s_ada_rangom_forest)
    print('ada_GBDT',s_ada_GBDT)
    print('knn',s_knn)
    print('knn_ada',s_knn_ada)
    # print('s_train_knn',s_train_knn)
    # print('poly',s_poly)
    # print('weight', s_weight)
    # print('ada+poly',s_ada_poly )
    return y_lng_pred,y_lat_pred,s_ada,s_poly,s_weight,s_ada_poly,s_knn_ada,s_knn,s_xgboost,\
           s_ada_xgboost,s_ada_GBDT,s_ada_bagging,s_ada_rangom_forest,s_GBDT,s_bagging,s_rangom_forest

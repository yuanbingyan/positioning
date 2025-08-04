import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import joblib
import os

def knn_main(data,lte_id):
    # data = pd.read_excel('180757-1.xlsx')
    data = np.array(data)  # (25721,19)
    data = data.T  # (19, 25721)
    x = data[0:9, :]  # (17, 25721)
    y = data[9:, ]  # (2, 25721)
    ############ 数据处理
    x = x.T
    y = y.T
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.0001, random_state=0,shuffle=True)
    train_y_lng =  train_y[:, 0]
    train_y_lat = train_y[:, 1]
    test_y_lng = test_y[:, 0]
    test_y_lat = test_y[:, 1]
    ## knn
    k = 1
    knn_lat = KNeighborsRegressor(k, metric='manhattan')
    knn_lat.fit(train_x, train_y_lat)
    train_y_lat_pred = knn_lat.predict(train_x)
    test_y_lat_pred = knn_lat.predict(test_x)
    ###
    knn_lng = KNeighborsRegressor(k, metric='manhattan')
    knn_lng.fit(train_x,train_y_lng)
    train_y_lng_pred = knn_lng.predict(train_x)
    test_y_lng_pred = knn_lng.predict(test_x)
    #############计算误差
    print(lte_id,'knn纬度训练误差',np.mean(np.abs(train_y_lat-train_y_lat_pred)))
    print(lte_id,'knn纬度测试误差',np.mean(np.abs(test_y_lat-test_y_lat_pred)))
    print(lte_id,'knn经度训练误差',np.mean(np.abs(train_y_lng-train_y_lng_pred)))
    print(lte_id,'knn经度测试误差',np.mean(np.abs(test_y_lng-test_y_lng_pred)))
    ####保存模型
    if not os.path.exists('knn_model/'):
        os.makedirs('knn_model/')
    joblib.dump((knn_lng,knn_lat),'knn_model/'+str(lte_id)+'.pkl')


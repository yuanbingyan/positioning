import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from knn_model import knn_main
from adaboost import adaboost_main
from data_op import data_op_main
def train(): ## 训练模型
    limli_scale_data=pd.read_csv('计算覆盖距离.csv',encoding='ANSI')
    data_with_GPS = pd.read_csv('华为已处理数据.csv',encoding='ANSI')
    for index,row in limli_scale_data.iterrows():
        lte_id = row['lte_id']
        limli_scale = row['覆盖范围']
        BS_lng = data_with_GPS[data_with_GPS['lte_id'] == lte_id].经度.iloc[0]
        BS_lat = data_with_GPS[data_with_GPS['lte_id'] == lte_id].纬度.iloc[0]
        lte_id_data = data_with_GPS[data_with_GPS['lte_id'] == lte_id][['ltescrsrp','ltescrsrq','ltescphr','ltencrsrp_1','ltencrsrq_1','ltencpci_1','ltencrsrp_2','ltencrsrq_2','ltencpci_2','longitude','latitude']]
        # knn训练
        knn_main(lte_id_data,lte_id)
        # adaboost 训练
        adaboost_main(lte_id_data,BS_lng,BS_lat,limli_scale,lte_id)
        print(lte_id,'完成')

def pred():  # 预测数据
    data = pd.read_csv('没有经纬度的华为数据.csv', encoding='ANSI')
    limli_scale_data = pd.read_csv('计算覆盖距离.csv', encoding='ANSI')
    data_with_GPS = pd.read_csv('华为已处理数据.csv',encoding='ANSI')
    for index, row in limli_scale_data.iterrows():
        ## 准备训练数据
        lte_id = row['lte_id']
        lte_id_data = data_with_GPS[data_with_GPS['lte_id'] == lte_id][
            ['ltescrsrp', 'ltescrsrq', 'ltescphr', 'ltencrsrp_1', 'ltencrsrq_1', 'ltencpci_1', 'ltencrsrp_2',
             'ltencrsrq_2', 'ltencpci_2', 'longitude', 'latitude']]
        # 准备预测数据
        predict_x = data[data['lte_id'] == lte_id][
            ['ltescrsrp', 'ltescrsrq', 'ltescphr', 'ltencrsrp_1', 'ltencrsrq_1', 'ltencpci_1', 'ltencrsrp_2',
             'ltencrsrq_2', 'ltencpci_2']]
        GPS_info = data[data['lte_id'] == lte_id][['mmeues1apid','lte_timestamp','lte_id',]]
        y_lng_pred,y_lat_pred = data_op_main(predict_x,lte_id_data,lte_id)
        pred_GPS = np.concatenate((GPS_info,y_lng_pred.reshape(len(y_lng_pred), 1), y_lat_pred.reshape(len(y_lat_pred), 1)), axis=1)
        # print(pred_GPS.shape)
        pd.DataFrame(pred_GPS).to_csv('GPS.csv',header=False,mode='a',index=None)
        print(lte_id,'完成')

train()
pred()
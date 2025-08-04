from All_data.挖洞.挖洞_多项式回归 import train_poly
from All_data.挖洞.挖洞_adaboost import train_adaboost
from All_data.挖洞.挖洞_AdaBoost_KNN import train_ada_knn
# file = '183053-1.xlsx'
# lng_1 = 103.7045
# lat_1 = 36.075
# lng_2 = 103.7037
# lat_2 = 36.074
# lng_3 = 103.7035
# lat_3 = 36.075

file = '180757-1.xlsx'
lng_1 = 103.7005
lat_1 = 36.0755
lng_2 = 103.7043
lat_2 = 36.0772
lng_3 = 103.7035
lat_3 = 36.075
train_adaboost(file,lng_1,lat_1,lng_2,lat_2,lng_3,lat_3)
print('==========================')
train_ada_knn(file,lng_1,lat_1,lng_2,lat_2,lng_3,lat_3)
# train_poly(file,lng,lat)
# '182610-2.xlsx'
# '182610-2.xlsx'
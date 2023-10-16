import numpy as np
import pandas as pd
import pulp
import itertools
import folium
import openrouteservice
from geopy.distance import great_circle
from geopy.geocoders import Nominatim
from branca.element import Figure
from openrouteservice import convert
import matplotlib.pyplot as plt
import japanize_matplotlib
import sys
import time
import csv

##前準備


#座標取得
geo_list = []
geo = pd.read_csv("geo_point.csv")
point = geo.to_numpy().tolist()
geolocator = Nominatim(user_agent="user-id")
for chimei in range(len(geo)):
    location = geolocator.geocode(geo.iloc[chimei,0]+","+geo.iloc[chimei,1])
    print(location)
    geo_list.append([location.address.split(",")[0],location.latitude, location.longitude,geo.iloc[chimei,2]])
zahyo = pd.DataFrame(geo_list, columns=["地名","latitude","longitude","stay_time"])
line = zahyo.to_numpy().tolist()

color_list = ["red","green","purple","orange","darkred","lightred","beige","darkblue","darkgreen","cadetblue","darkpurple","white","pink","lightblue","lightgreen","gray","black","lightgray","blue"]
points_a = []
for i in range(len(zahyo)):
    points_a.append([zahyo.iloc[i,1],zahyo.iloc[i,2]])
    

key = "5b3ce3597851110001cf624804ffaeec7cd246038d01eb4d3a32f633"
client = openrouteservice.Client(key=key)

datalist = []

for i in range(len(line)-2):
    p1 = float(line[i+1][1]), float(line[i+1][2])
    for j in range(i,len(line)-2):
        p2 = float(line[j+2][1]), float(line[j+2][2])
        p1r = tuple(reversed(p1))
        p2r = tuple(reversed(p2))
        mean_lat = (p1[0] + p2[0]) / 2
        mean_long = (p1[1] + p2[1]) / 2

        # 経路計算 (Directions V2)
        routedict = client.directions((p1r, p2r),profile="foot-walking")
        geom = routedict["routes"][0]["geometry"]
        decoded = convert.decode_polyline(geom)
        datalist.append([i,j+1,float(routedict["routes"][0]["summary"]["duration"])])
idou_jikan = pd.DataFrame(datalist, columns=["出発地点","行先","移動時間[s]"])
loc_loc_time = idou_jikan.to_numpy().tolist()

##ここから最適化

num_places_time = len(idou_jikan)
customer_count = len(zahyo) #場所の数（id=0はdepot）
lim_day_count = 100 #何日かlim
lim_time_capacity = 60*60*6 #一日に使える時間

cost = [[0 for i in range(customer_count)] for j in range(customer_count)]
for i in range(num_places_time):
    cost[int(loc_loc_time[i][0])][int(loc_loc_time[i][1])] = float(loc_loc_time[i][2])
    cost[int(loc_loc_time[i][1])][int(loc_loc_time[i][0])] = float(loc_loc_time[i][2])
cost = np.array(cost)
print(cost)

visit = [[0 for i in range(customer_count)] for j in range(customer_count)]
for i in range(num_places_time):
    visit[int(loc_loc_time[i][0])][int(loc_loc_time[i][1])] = float(point[int(loc_loc_time[i][1])+1][2])*60
    visit[int(loc_loc_time[i][1])][int(loc_loc_time[i][0])] = float(point[int(loc_loc_time[i][0])+1][2])*60
visit = np.array(visit)
print(visit)



for lim_day_count in range(lim_day_count+1):
    opt_TripY = pulp.LpProblem("CVRP", pulp.LpMinimize)
    
    #変数定義
    
    X_ijk = [[[pulp.LpVariable("X%s_%s,%s"%(i,j,k), cat="Binary") if i != j else None for k in range(lim_day_count)]
            for j in range(customer_count)] for i in range(customer_count)]
    Y_ijk = [[[pulp.LpVariable("Y%s_%s,%s"%(i,j,k), cat="Binary") if i != j else None for k in range(lim_day_count)]
            for j in range(customer_count)] for i in range(customer_count)]

    #制約条件

    for j in range(1, customer_count):
        opt_TripY += pulp.lpSum(X_ijk[i][j][k] if i != j else 0 for i in range(customer_count) for k in range(lim_day_count)) == 1 

    for k in range(lim_day_count):
        opt_TripY += pulp.lpSum(X_ijk[0][j][k] for j in range(1,customer_count)) == 1
        opt_TripY += pulp.lpSum(X_ijk[i][0][k] for i in range(1,customer_count)) == 1

    for k in range(lim_day_count):
        for j in range(customer_count):
            opt_TripY += pulp.lpSum(X_ijk[i][j][k] if i != j else 0 for i in range(customer_count)) -  pulp.lpSum(X_ijk[j][i][k] for i in range(customer_count)) == 0
    for k in range(lim_day_count):
        opt_TripY += pulp.lpSum(visit[i][j] * X_ijk[i][j][k] + cost[i][j] * X_ijk[i][j][k] if i != j else 0 for i in range(customer_count) for j in range (customer_count)) <= lim_time_capacity
        
    #目的関数
    
    opt_TripY += pulp.lpSum(visit[i][j] * X_ijk[i][j][k] + cost[i][j] * X_ijk[i][j][k] if i != j else 0 for k in range(lim_day_count) for j in range(customer_count) for i in range (customer_count))
    
    #部分巡回路除去制約
    
    subtours = []
    for i in range(2,customer_count):
        subtours += itertools.combinations(range(1,customer_count), i)
    for s in subtours:
        opt_TripY += pulp.lpSum(X_ijk[i][j][k] if i !=j else 0 for i, j in itertools.permutations(s,2) for k in range(lim_day_count)) <= len(s) - 1
        
    if opt_TripY.solve() == 1:
        time_start = time.time()
        status = opt_TripY.solve()
        time_stop = time.time()
        print(f'ステータス:{pulp.LpStatus[status]}')
        print('日数:', lim_day_count)
        print('目的関数値:',pulp.value(opt_TripY.objective))
        print('使用時間:',f"{int(pulp.value(opt_TripY.objective)//3600)}時間{int(pulp.value(opt_TripY.objective)%3600//60)}分{pulp.value(opt_TripY.objective)%3600%60:.3}秒")
        print(f'計算時間:{(time_stop - time_start):.3}(秒)')
        break
if not(opt_TripY.solve()) == 1:
    print("日数が足りません. プランを立て直してください.")
    sys.exit()
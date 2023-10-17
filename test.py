import numpy as np
import pandas as pd
import pulp
import itertools
import folium
import openrouteservice
from geopy.geocoders import Nominatim
from branca.element import Figure
from openrouteservice import convert
import sys
import time


geo = pd.read_csv("geo_point.csv")
key = "5b3ce3597851110001cf624804ffaeec7cd246038d01eb4d3a32f633"
client = openrouteservice.Client(key=key)
lim_time_capacity = 60*60*8 #一日に使える時間



#座標取得
geo_list = []
geolocator = Nominatim(user_agent="user-id")
for chimei in range(len(geo)):
    location = geolocator.geocode(geo.iloc[chimei,0]+","+geo.iloc[chimei,1])
    print(location)
    geo_list.append([location.address.split(",")[0],location.latitude, location.longitude,geo.iloc[chimei,2]])
location_time = pd.DataFrame(geo_list, columns=["地名","latitude","longitude","stay_time"])



datalist = []
line = location_time.to_numpy().tolist()
for i in range(len(line)):
    p1 = float(line[i][1]), float(line[i][2])
    for j in range(i,len(line)-1):
        p2 = float(line[j+1][1]), float(line[j+1][2])
        p1r = tuple(reversed(p1))
        p2r = tuple(reversed(p2))
        mean_lat = (p1[0] + p2[0]) / 2
        mean_long = (p1[1] + p2[1]) / 2
        
        # 経路計算 (Directions V2)
        routedict = client.directions((p1r, p2r),profile="foot-walking")
        datalist.append([i,j+1,float(routedict["routes"][0]["summary"]["duration"])])
transfer_time = pd.DataFrame(datalist, columns=["出発地点","行先","移動時間[s]"])


num_places_time = len(transfer_time)
customer_count = len(location_time) #場所の数（id=0はdepot）
lim_day_count = 100 #何日かlim

loc_loc_time = transfer_time.to_numpy().tolist()
point = geo.to_numpy().tolist()

cost = [[0 for i in range(customer_count)] for j in range(customer_count)]
for i in range(num_places_time):
    cost[int(loc_loc_time[i][0])][int(loc_loc_time[i][1])] = float(loc_loc_time[i][2])
    cost[int(loc_loc_time[i][1])][int(loc_loc_time[i][0])] = float(loc_loc_time[i][2])
cost = np.array(cost)
print(cost)

visit = [[0 for i in range(customer_count)] for j in range(customer_count)]
for i in range(num_places_time):
    visit[int(loc_loc_time[i][0])][int(loc_loc_time[i][1])] = float(point[int(loc_loc_time[i][1])][2])*60
    visit[int(loc_loc_time[i][1])][int(loc_loc_time[i][0])] = float(point[int(loc_loc_time[i][0])][2])*60
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
    



color_list = ["red","green","purple","orange","darkred","lightred","beige","darkblue","darkgreen","cadetblue","darkpurple","white","pink","lightblue","lightgreen","gray","black","lightgray","blue"]

points_a = []
for i in range(len(location_time)):
    points_a.append([location_time.iloc[i,1],location_time.iloc[i,2]])




def route_view(points_a):
    loc_place = []
    for chimei in range(len(points_a)-1):
        p1 = points_a[chimei]
        p2 = points_a[chimei+1]
        p1r = tuple(reversed(p1))
        p2r = tuple(reversed(p2))

        # 経路計算 (Directions V2)
        routedict = client.directions((p1r, p2r),profile="foot-walking")
        geom = routedict["routes"][0]["geometry"]
        decoded = convert.decode_polyline(geom)
        for i in range(len(decoded["coordinates"])):
            loc_place.append(decoded["coordinates"][i])
    return loc_place

def reverse_lat_long(list_of_lat_long):
    return [(p[1], p[0]) for p in list_of_lat_long]


ave_lat = sum(p[0] for p in points_a)/len(points_a)
ave_lon = sum(p[1] for p in points_a)/len(points_a)
fig = Figure(width=800, height=400)

my_map = folium.Map(
    location=[ave_lat, ave_lon], 
    zoom_start=12
)

basyo_num_list = []
hiduke_judg_list = []

for k in range(lim_day_count):
    for i in range(customer_count):
        for j in range(customer_count):
            if i != j and pulp.value(X_ijk[i][j][k]) == 1:
                #print("日付：",k)
                #print("地点：",i)
                #print("目的：",j,"\n")
                basyo_num_list.append(i)
                hiduke_judg_list.append(k)
                
print(basyo_num_list,hiduke_judg_list)
day_trip_zahyo = []
hiduke_hantei = 0
bangou_1 = 0

for aaaa in hiduke_judg_list:
    bangou_2 = basyo_num_list[bangou_1]
    if not(aaaa==hiduke_hantei):
        day_trip_zahyo.append([location_time.iloc[0,1],location_time.iloc[0,2]])
        print(day_trip_zahyo)
        def_routeview = route_view(day_trip_zahyo)
        coord = reverse_lat_long(def_routeview)
        folium.vector_layers.PolyLine(coord,
                                        color=color_list[aaaa], 
                                        weight=2.5, 
                                        opacity=1
                                        ).add_to(my_map)
        for each in range(len(day_trip_zahyo)-2):
            folium.Marker(
                    location=day_trip_zahyo[each+1],
                    icon = folium.Icon(color=color_list[aaaa])
                ).add_to(my_map)
        day_trip_zahyo = []
        day_trip_zahyo.append([location_time.iloc[bangou_2,1],location_time.iloc[bangou_2,2]])
        hiduke_hantei += 1
    else:
        day_trip_zahyo.append([location_time.iloc[bangou_2,1],location_time.iloc[bangou_2,2]])
    bangou_1 += 1
    
#最終日ルート
day_trip_zahyo.append([location_time.iloc[0,1],location_time.iloc[0,2]])
print(day_trip_zahyo)
def_routeview = route_view(day_trip_zahyo)
coord = reverse_lat_long(def_routeview)
folium.vector_layers.PolyLine(coord,
                                color=color_list[0], 
                                weight=2.5, 
                                opacity=1
                                ).add_to(my_map)
for each in range(len(day_trip_zahyo)-2):
    folium.Marker(
            location=day_trip_zahyo[each+1],
            icon = folium.Icon(color=color_list[0])
        ).add_to(my_map)
folium.Marker(
    location=[location_time.iloc[0,1],location_time.iloc[0,2]],
    popup=location_time.iloc[0,0]
).add_to(my_map)

#最後日に赤ルートならばどの順番で回っても大丈夫

my_map.save(outfile="map.html")

fig.add_child(my_map)
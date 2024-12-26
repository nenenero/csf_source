import pandas as pd 
import os
import json
import datetime
import math
import csv

csv_path = 'CSV_data_1011\\2018-06-26\\2018-06-26.csv'
day_hold = 26
time_csv_path = 'count_csv'
flight_id = []
flight_nums = []
hour_hold = 0
minute_hold = 0
time_hold = -10
#day_hold = 2
data_csv = pd.read_csv(csv_path)
flight_count = 0
visit_count = 0
not_in_circle =0
weekday = 'y'
count_csv_path = csv_path.rsplit('\\',1)[1]
Week_dict= {'Monday':1, 'Tuesday':2 , 'Wednesday':3, 'Thursday':4, 'Friday': 5, 'Saturday':6, 'Sunday': 7}
count_csv_path = count_csv_path.rsplit('.')[0]
#day_hold = count_csv_path[-1]
#day_hold = int(day_hold)
#print('dayhold: ',str(day_hold-2))
count_csv_path = time_csv_path + '\\' + count_csv_path
#os.makedirs(count_csv_path)
index_hold = 0
last_time = 5

#time_hold += 15
print('now_time_hold: ', time_hold)
flight_nums.append(0)
for index, rows in data_csv.iloc[index_hold:].iterrows():
        
    id = rows['Id']
    isInCircle = rows['isInCircle']
    month = rows['Month']
    day = rows['Day']
    hour = rows['Hour']
    minute = rows['Minute']
    weekday = rows['WeekDay']
    weekday = Week_dict[weekday]
    #time_hold = hour_hold * 60 + minute_hold
    time_here = hour * 60 + minute

    if(isInCircle == 0):
        not_in_circle += 1
        print('hour: '+ str(hour) + '  minute: ' + str(minute) + '  id: '+ str(id) )
        continue

    if(day != day_hold or time_here < 5):
        visit_count += 1
        continue

    elif(time_here > time_hold +15 and isInCircle ==1 ):
        #if(time_here>time_hold+30):
        #    print('time hold : ' + str(time_hold) + ' time here: '+ str(time_here))
        #    flight_id = []
        #    flight_nums[-1]=0
        #    break
        print('time hold : ' + str(time_hold) + ' time here: '+ str(time_here))
        visit_count += 1
        #print('before clear : '+ str(flight_id))
        flight_id = []
        #print('after clear : '+ str(flight_id))
        flight_id.append(id)
        time_hold += 15
        flight_nums.append(flight_count)
        flight_count = 1
        #index_hold = int(index)+1
        #break
        continue

        
    elif(time_here - time_hold <= 15 and isInCircle ==1 ):
        if id in flight_id :
            visit_count += 1
            #print('id exist ' + str(time_hold) + ': '+ str(id))
            continue
        else:
            visit_count += 1
            flight_count += 1
            flight_id.append(id)

print(flight_nums)
print('visit_count = '+ str(visit_count))
print('not in circle = '+str(not_in_circle))

count_csv_rows = ['Day', 'TimePeriod', 'WeekDay', 'Flight_nums']
time_hold = -10
count_csv_path = count_csv_path + '.csv'
with open(count_csv_path,"w", newline='') as count_csv:
    writer = csv.writer(count_csv)
    writer.writerow(count_csv_rows)
    for Fn in flight_nums:
        writer.writerow((day_hold,time_hold,weekday,Fn))
        time_hold += 15






        

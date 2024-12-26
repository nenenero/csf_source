import pandas as pd 
import os
import json
import datetime
import math
json_path = 'D:\\homeworks\\csf_project_learn\\ADS-B数据\\ADS-B数据\\5周\\2018-06-30 Frankfurt Airport\\'
toward_path= 'D:\\homeworks\\csf_project_learn\\CSV_data_1011\\'
col_to_keep = ['Id','Alt','GAlt','Lat','Long','PosTime','Spd','Trak','TT','Cos']
col_to_keep_dup = ['Id','Alt','GAlt','Lat','Long','PosTime','Spd','Trak','TT']
flag = 0
Lat = 50.033333
Long = 8.570556
Lat = math.radians(Lat)
Long = math.radians(Long)
feet2km = 0.0003048
#def location(Lat, Long):
r_avg =6371
newdir = json_path.rsplit('\\',2)[1]
newdir = newdir[:10]
os.makedirs(toward_path + newdir + '\\')
toward_path = toward_path + newdir + '\\'
def distance(loc):
    lat1 = loc[0]
    long1 = loc [1]
    lat1 = math.radians(lat1)
    long1 = math.radians(long1)
    d_test = 2 * r_avg *math.asin( math.sqrt( ((math.sin((lat1 - Lat)/2))**2) + math.cos(Lat) * math.cos(lat1) * ((math.sin((long1 - Long)/2))**2)) )
    #print('lat1:'+str(lat1)+'  '+ 'long1:' + str(long1)+ '  d_test:' + str(d_test))
    return d_test

for file_name in os.listdir(json_path):
    if(file_name.endswith('.json')):
        print(file_name)
        file_path = os.path.join(json_path,file_name)
        f = open(file_path,'r',encoding='utf-8')
        file_json = json.load(f)
        df = pd.json_normalize(file_json,record_path='acList')
        df_filtered = df[col_to_keep]
        for index, row in df_filtered.iterrows():
            TT = row['TT']
    #print(str(index)+TT)
            if((TT == 'a') or (TT == 's')):
                new_cos = row['Cos'][0:4]
                df_filtered.at[index,'Cos'] = new_cos
            else:
                new_cos = row['Cos'][0:3]
                df_filtered.at[index,'Cos'] = new_cos

          
        df_filtered_nodup = df_filtered.drop_duplicates(subset=col_to_keep_dup)

        de_na_col = ['Alt','Lat','Long','PosTime','Trak','TT','Cos','GAlt']

        df_filtered_nodup_denan = df_filtered_nodup.dropna(subset = de_na_col)



        df_filtered_nodup_denan_sortbytime = df_filtered_nodup_denan.sort_values(by='PosTime')
        df_filtered_nodup_denan_sortbytime['Year']='2018'
        df_filtered_nodup_denan_sortbytime['Month'] = '06'
        df_filtered_nodup_denan_sortbytime['Day'] = '01'
        df_filtered_nodup_denan_sortbytime['Hour']='000'
        df_filtered_nodup_denan_sortbytime['Minute']='00'
        #df_filtered_nodup_denan_sortbytime['X']='0'
        #df_filtered_nodup_denan_sortbytime['Y']='0'
        df_filtered_nodup_denan_sortbytime['Distance'] = '0'
        df_filtered_nodup_denan_sortbytime['isInCircle'] = '0'
        df_filtered_nodup_denan_sortbytime['GAltInKm'] = '0'
        df_filtered_nodup_denan_sortbytime['WeekDay'] = 'X'
        for index, row in df_filtered_nodup_denan_sortbytime.iterrows():
            PosTime = row['PosTime']
            utc_time = datetime.datetime.utcfromtimestamp(PosTime/1000)
            new_Year = utc_time.strftime('%Y')
            new_Month = utc_time.strftime('%m')
            new_Day = utc_time.strftime('%d')
            new_Hour = utc_time.strftime('%H')
            new_Minute = utc_time.strftime('%M')
            new_WeekDay = utc_time.strftime('%A')
            #new_X = 
            lat1 = row['Lat']
            long1 = row['Long']
            loc = [lat1, long1]
            #print(loc)
            new_distance = distance(loc)
            new_GAltInKm = feet2km * row['GAlt']
            if(new_distance<=300.0 and new_GAltInKm <= 300.0):
                df_filtered_nodup_denan_sortbytime.at[index,'isInCircle'] = '1'
            df_filtered_nodup_denan_sortbytime.at[index,'GAltInKm'] = new_GAltInKm
            df_filtered_nodup_denan_sortbytime.at[index,'Year']=new_Year
            df_filtered_nodup_denan_sortbytime.at[index,'Month']=new_Month
            df_filtered_nodup_denan_sortbytime.at[index,'Day'] = new_Day
            df_filtered_nodup_denan_sortbytime.at[index,'Hour']=new_Hour
            df_filtered_nodup_denan_sortbytime.at[index,'Minute']=new_Minute
            df_filtered_nodup_denan_sortbytime.at[index,'Distance']=new_distance
            df_filtered_nodup_denan_sortbytime.at[index,'WeekDay'] = new_WeekDay
        #print(df_filtered_nodup)
        csv_file_name = file_name.replace('json','csv')
        #os.mkdir(toward_path+csv_file_name)
        csv_path = toward_path+csv_file_name
        df_filtered_nodup_denan_sortbytime.to_csv(csv_path,index=False)
        #df_filtered.to_csv(csv_path,index=False)





        
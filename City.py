from head import *
import time

li=["zhangjiakou","Tianjin","sjiazhuang"]
keys=["AQI","PM2.5","PM10","SO2","NO2","CO","O3"]
year=[2014,2015,2016,2017]
month=["1","2","3","4","5","6","7","8","9","10","11","12"]
file_path="/root/R 语言大作业/airdata_zhangjiakouAll.xlsx"
file_path2="/root/R 语言大作业/airdata_tianjinAll.xlsx"
file_path3="/root/R 语言大作业/airdata_sjzhuangAll.xlsx"
file_path_lists=[file_path,file_path2,file_path3]
df_zhangjiakou=df_read(file=file_path)
df_sjiahzhuang=df_read(file=file_path2)
df_Tianjin=df_read(file=file_path3)

df_zhangjikou=df_zhangjiakou.loc[:,"AQI":]
df_sjzhuang=df_sjiahzhuang.loc[:,"AQI":"O3"]
df_Tianjn=df_Tianjin.loc[:,"AQI":'O3']

Se_zhangjikou_med=medians(df_zhangjikou)
Se_sjiazhuang_med=medians(df_sjzhuang)
Se_Tianjin_med=medians(df_Tianjn)

Se_zhangjikou_me=means(df_zhangjikou)
Se_sjiazhuang_me=means(df_sjzhuang)
Se_Tianjin_me=means(df_Tianjn)

Se_zhangjiakou_std=stds(df_zhangjikou)
Se_sjiazhuang_std=stds(df_sjzhuang)
Se_Tianjin_std=stds(df_Tianjn)

Se_zhangjiakou_max=MAXS(df_zhangjikou)
Se_sjiazhuang_max=MAXS(df_sjzhuang)
Se_Tianjin_max=MAXS(df_Tianjn)

Se_zhangjiakou_min=MINS(df_zhangjikou)
Se_sjiazhuang_min=MINS(df_sjzhuang)
Se_Tianjin_min=MINS(df_Tianjn)

Se_zhangjiakou_skew=SKEWS(df_zhangjikou)
Se_shijiazhuang_skew=SKEWS(df_sjzhuang)
Se_Tianjin_skew=SKEWS(df_Tianjn)

df_me=pd.DataFrame({
    "zhangjiakou":Se_zhangjikou_me,
    "sjiazhuang":Se_sjiazhuang_me,
    "Tianjin":Se_Tianjin_me
})
df_med=pd.DataFrame({
    "zhangjiakou":Se_sjiazhuang_med,
    "sjiazhuang":Se_sjiazhuang_med,
    "Tianjin":Se_Tianjin_med
})
df_std=pd.DataFrame({
    "zhangjiakou":Se_zhangjiakou_std,
    "sjiazhuang":Se_sjiazhuang_std,
    "Tianjin":Se_Tianjin_std
})
df_skew=pd.DataFrame({
    "":["AQI","PM2.5","PM10","SO2","NO2","CO","O3"],
    "zhangjiakou":Se_zhangjiakou_skew,
    "sjiazhuang":Se_shijiazhuang_skew,
    "Tianjin":Se_Tianjin_skew
})

df_max=pd.DataFrame({
    "zhangjiakou":Se_zhangjiakou_max,
    "sjiazhuang":Se_sjiazhuang_max,
    "Tianjin":Se_Tianjin_max
})
df_min=pd.DataFrame({
    "zhangjiakou":Se_zhangjiakou_min,
    "sjiazhuang":Se_sjiazhuang_min,
    "Tianjin":Se_Tianjin_min
})

def average_year():
    tmp2=[2014,2015,2016,2017,2018]
    for i in keys:
    
        year_average_z=df_zhangjiakou.groupby(pd.Grouper(freq='Y'))[i].mean().tolist()
        year_average_s=df_sjiahzhuang.groupby(pd.Grouper(freq='Y'))[i].mean().tolist()
        year_average_t=df_Tianjin.groupby(pd.Grouper(freq='Y'))[i].mean().tolist()
        year_average_each_year={
            "zhangjiakou":year_average_z,
            "sjiazhuang":year_average_s,
            "Tianjin":year_average_t
            }
        year_average_each_year=pd.DataFrame(data=year_average_each_year,index=tmp2)
        ax=create_multi_plot(x=tmp2,y=li,df=year_average_each_year,xlabel=i,title="each year")
        print(year_average_each_year)
        plt.show()
    


def CITY(file:str):
    zhangjiakou=df_read(file=file)
    zhangjiakou_simplify=df_simplify_average(zhangjiakou,"AQI")
    ax_simplify=create_multi_plot(x=month,y=year,title="zhangjiakou month 2014-2017 AQI",xlabel="month",df=zhangjiakou_simplify)
    zhangjiakou_years={}
    plt.show()
    for i in year:
        tmp=df_time_split(df_zhangjiakou,i)
        zhangjiakou_years.update({i:tmp})
    num=int(input("输入要分析的年份 :-) "))
    tmp={}
    for i in keys:
        zhangjiakou_year=zhangjiakou_years[num].groupby(pd.Grouper(freq='M'))[i].mean().tolist()
        tmp.update({i:zhangjiakou_year})
    zhangjiakou_year_average=pd.DataFrame(tmp,index=month)
    ax_multi_bar=create_multi_bar2(month,zhangjiakou_year_average,xlabel="month",ylabel="value",title=f"the month average of {num}",width=0.13)
# 散点图分析aqi和各个变量的关系
    time.sleep(5)
    for i in keys[2:]:
        print(f"{i}:")
        ax3=create_scatter_with_regression(labels=zhangjiakou_years[num][i],data=zhangjiakou_years[num]["AQI"],xlabel=i,ylabel='AQI',title=f"SCATTER:{i} and AQI")
        
    plt.show()
    # 进行回归分析以及预测
    print("回归分析 1.单个数据 2.多个数据")
    num=int(input("root@kali#:"))
    match num:
        
        case 1:
            s=input("请输入训练变量:")
            X_train, X_test, y_train, y_test=datapro(x=s,y="AQI",df=zhangjiakou)
            model=model_init()
            model_training(model=model,xt_train=X_train,y_train=y_train)
            model_judge(model=model,X_test=X_test,y_test=y_test)
            
            newdata=np.array([float(input("输入值:"))]).reshape(1,-1)
            predict_data=model_predict(model=model,data=newdata)
            print("预测结果$:",end="")
            print(predict_data)
        case 2:
            
            X_train, X_test, y_train, y_test=datapro_s(x=keys[1:],y="AQI",df=zhangjiakou)
            model=model_init()
            model_training(model=model,xt_train=X_train,y_train=y_train)
            model_judge(model=model,X_test=X_test,y_test=y_test)
            zhangjiakou.index=pd.to_datetime(zhangjiakou.index)
            print("由于在2018年的数据较少,所以选择2018的数据进行预测 :-)")
            newdata=zhangjiakou[zhangjiakou.index.year==2018].loc[:,"PM2.5":"O3"]
            predict_data=model_predict(model=model,data=newdata)
            print("预测结果")
            print(predict_data)
            print("单个数据预测")
        

def CITYS():
    # 每个城市2014-2017各个值平均值,中位数,最大值,最小值,方差,均差
    print("每个城市2014-2017各个城市平均值,中位数,最大值,最小值,方差,均差如图:")
    ax1=create_multi_bar(keys,df_me,xlabel="keys",ylabel="value",title="zhangjiakou-Tianjin 2014-2017 AQI-O3 aver values",width=0.12)
    ax2=create_multi_bar(keys,df_med,xlabel="keys",ylabel="value",title="zhangjiakou-Tianjin 2014-2017 AQI-O3 med values",width=0.12)
    ax3=create_multi_bar(keys,df_max,xlabel="keys",ylabel="value",title="zhangjiakou-Tianjin 2014-2017 AQI-O3 max values",width=0.12)
    ax4=create_multi_bar(keys,df_min,xlabel="keys",ylabel="value",title="zhangjiakou-Tianjin 2014-2017 AQI-O3 min values",width=0.12)
    ax5=create_multi_bar(keys,df_std,xlabel="keys",ylabel="value",title="zhangjiakou-Tianjin 2014-2017 AQI-O3 std values",width=0.12)
    ax6=create_multi_bar(keys,df_skew,xlabel="keys",ylabel="value",title="zhangjiakou-Tianjin 2014-2017 AQI-O3 skew values",width=0.12)
    # print("2014-2017总年平均值")
    print("三年每年平均值")
    average_year()
    plt.show()
    
def deal():
    while True:
        a=input("输入城市(break退出):")
        if a=="break":
            break
        k=str(input("key值:"))
        d_f={
            "zhangjiakou":df_zhangjiakou,
            "sjiazhuang":df_sjiahzhuang,
            "Tianjin":df_Tianjin
            }
        
        city_min=df_simplify_min(df=d_f[a],var=k)
        city_max=df_simplify_max(df=d_f[a],var=k)
        city_std=df_simplify_std(df=d_f[a],var=k)
        city_med=df_simplify_average(df=d_f[a],var=k)
        
        print("*********the min********")
        print(city_min)
        print("***********************")
        print("*********the max********")
        print(city_max)
        print("***********************")
        print("*********the std********")
        print(city_std)
        print("***********************")
        print("*********the average********")
        print(city_med)
        print("***********************")     
        
# 测试代码
# if __name__=="__main__":
#     while True:
#         print("1-单个城市分析")
#         print("2-多个城市分析")
#         print("3-相关数据结果")
#         print("break quit")
#         num=int(input("root@kali:~# :"))
#         match num:
#             case 1:
                
#                 while True:
#                     city=str(input("输入城市:"))
#                     match city:
#                         case "zhangjiakou":
#                             #zhangjiakou()
#                             CITY(file=file_path_lists[0])
#                         case "sjiazhuang":
#                             CITY(file_path_lists[1])
#                         case "Tianjin":
#                             CITY(file_path_lists[2])
#                         case "break":
#                             print("相关数据结果")
#                             break
                        
#             case 2:
#                 CITYS()
#             case 3:
#                 deal()           
#             case 4:
#                break
                
            
                
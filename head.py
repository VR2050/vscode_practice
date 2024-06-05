import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as se
from scipy.stats import skew
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error, r2_score 
# global vars
import pandas as pd  
from scipy.stats import linregress  
  
# 城市列表（注意：实际中可能不会在此处直接定义城市，但为示例提供）  
li=["zhangjiakou","Tianjin","sjiazhuang"]  
# 数据键列表  
keys=["AQI","PM2.5","PM10","SO2","NO2","CO","O3"]  
# 年份列表  
year=[2014,2015,2016,2017]  
# 月份列表（字符串格式）  
month=["1","2","3","4","5","6","7","8","9","10","11","12"]  
  
# 函数：读取数据帧  
def df_read(file:str):  
    """  
    读取CSV或Excel文件，并将其转换为Pandas DataFrame。  
    如果文件中有缺失值，则用0填充。  
  
    参数:  
        file (str): 文件路径和名称。  
  
    返回:  
        pd.DataFrame: 转换后的数据帧，以时间为索引。  
  
    异常处理:  
        如果读取文件时发生错误，则打印"read_error!"。  
    """  
    try:  
        if 'csv' in file:  
            # 如果文件是CSV格式  
            df=pd.read_csv(file,parse_dates=['time'],index_col='time')  
            df.fillna(0,inplace=True)  # 填充缺失值为0  
            return df  
        elif 'xlsx' in file:  
            # 如果文件是Excel格式  
            df=pd.read_excel(file,parse_dates=['time'],index_col='time')  
            df.fillna(0,inplace=True)  # 填充缺失值为0  
            return df  
    except:  
        print("read_error!")  # 读取文件时发生错误  
  
# 函数：根据时间筛选数据帧  
def df_time_split(df:pd.DataFrame,time:int):  
    """  
    根据给定的时间（年份或月份）筛选数据帧。  
  
    参数:  
        df (pd.DataFrame): 要筛选的数据帧。  
        time (int): 时间值，可以是年份（2014-2017）或月份（1-12）。  
  
    返回:  
        pd.DataFrame: 筛选后的数据帧。  
  
    注意:  
        如果time大于等于2014，则假设它是年份；否则，假设它是月份。  
    """  
    df.index = pd.to_datetime(df.index)  # 确保索引是datetime类型  
    if time >= 2014:  
        # 如果time是年份  
        mask = df.index.year == time  
    else:  
        # 如果time是月份  
        mask = df.index.month == time  
    return df[mask]  # 返回筛选后的数据帧

# 计算DataFrame中某一列的平均值  

def average(df: pd.DataFrame, key: str):  

    """  

    计算给定DataFrame中指定列的平均值。  

  

    参数:  

    df (pd.DataFrame): 要处理的DataFrame  

    key (str): 要计算平均值的列名  

  

    返回:  

    float: 指定列的平均值  

    """  

    return df[key].mean()  

  

# 计算DataFrame中所有列的平均值  

def averages(df: pd.DataFrame):  

    """  

    计算给定DataFrame中所有列的平均值。  

  

    参数:  

    df (pd.DataFrame): 要处理的DataFrame  

  

    返回:  

    pd.Series: 包含所有列平均值的Series  

    """  

    return df.mean()  

  

# 计算DataFrame中某一列的标准差  

def std(df: pd.DataFrame, key: str):  

    """  

    计算给定DataFrame中指定列的标准差。  

  

    参数:  

    df (pd.DataFrame): 要处理的DataFrame  

    key (str): 要计算标准差的列名  

  

    返回:  

    float: 指定列的标准差  

    """  

    return df[key].std()  

  

# 计算DataFrame中所有列的标准差  

def stds(df: pd.DataFrame):  

    """  

    计算给定DataFrame中所有列的标准差。  

  

    参数:  

    df (pd.DataFrame): 要处理的DataFrame  

  

    返回:  

    pd.Series: 包含所有列标准差的Series  

    """  

    return df.std()  

  

# 计算DataFrame中某一列的中位数  

def median(df: pd.DataFrame, key: str):  

    """  

    计算给定DataFrame中指定列的中位数。  

  

    参数:  

    df (pd.DataFrame): 要处理的DataFrame  

    key (str): 要计算中位数的列名  

  

    返回:  

    float: 指定列的中位数  

    """  

    return df[key].median()  

  

# 注意：means函数与averages函数功能重复，通常不需要两个函数做同样的事情  

# 但为了保持一致性，这里仍然保留means函数并添加注释  

def means(df: pd.DataFrame):  

    """  

    计算给定DataFrame中所有列的平均值（与averages函数功能相同）。  

  

    参数:  

    df (pd.DataFrame): 要处理的DataFrame  

  

    返回:  

    pd.Series: 包含所有列平均值的Series  

    """  

    return df.mean()  

  

# 计算DataFrame中所有列的中位数  

def medians(df: pd.DataFrame):  

    """  

    计算给定DataFrame中所有列的中位数。  

  

    参数:  

    df (pd.DataFrame): 要处理的DataFrame  

  

    返回:  

    pd.Series: 包含所有列中位数的Series  

    """  

    return df.median()  

  

# 计算DataFrame中某一列的偏度  

def SKEW(df: pd.DataFrame, key: str):  

    """  

    计算给定DataFrame中指定列的偏度。  

  

    参数:  

    df (pd.DataFrame): 要处理的DataFrame  

    key (str): 要计算偏度的列名  

  

    返回:  

    float: 指定列的偏度值  

    """  

    return skew(df[key])  

def MAXS(df:pd.DataFrame):
    return df.max() 

def MINS(df:pd.DataFrame):
    return df.min()

# 注意：SKEWS函数需要确保DataFrame的每一列都可以计算偏度，  

# 并且所有列都应该具有数值型数据。否则，会抛出异常。  

def SKEWS(df: pd.DataFrame):  

    """  

    计算给定DataFrame中所有列的偏度。  

  

    参数:  

    df (pd.DataFrame): 要处理的DataFrame  

  

    返回:  

    pd.Series: 包含所有列偏度值的Series  

    """  

    return df.apply(skew)

def create_figure(x:int,y:int):
    return plt.figure(figsize=(x,y),dpi=80)

# def create_plot(x:str,y:list,title:str,df:pd.DataFrame):
#     plt.plot(marker='o')
#     plt.title(title,fontsize=14)
#     for i in y:
#         plt.plot(df[x],df[i],label=str(i))
#     plt.xlabel(str(x),color="black",fontsize=14)
#     plt.ylabel(str(x),color="black",fontsize=14)
#     plt.legend()
#     plt.show()

  
# 自动在柱状图上标注数值  
def autolabel(rects, ax: plt.Axes):  
    """  
    在柱状图的每个柱子上方添加一个文本标签，显示其高度。  
  
    参数:  
        rects (list): 包含matplotlib.patches.Rectangle对象的列表，代表柱状图中的柱子。  
        ax (plt.Axes): 绘图的Axes对象。  
  
    返回值:  
        无返回值，该函数直接在图上添加文本标签。  
    """  
    for rect in rects:  
        height = rect.get_height()  # 获取柱子的高度  
        ax.annotate('{:.2f}'.format(height),  # 格式化高度为两位小数  
                    xy=(rect.get_x() + rect.get_width() / 2, height),  # 文本位置为柱子中心上方  
                    xytext=(0, 3),  # 文本相对于指定位置的偏移量（垂直方向）  
                    textcoords="offset points",  # 文本坐标类型为偏移点  
                    ha='center', va='bottom')  # 文本水平和垂直对齐方式  
  
# 创建一个多折线图  
def create_multi_plot(x, y: list, title: str, df: pd.DataFrame, xlabel: str):  
    """  
    根据给定的x轴数据、y轴数据列表、标题、数据框和x轴标签，创建一个多折线图。  
  
    参数:  
        x (array-like): x轴的数据。  
        y (list): y轴数据列表，其中每个元素对应数据框中的一列。  
        title (str): 图表的标题。  
        df (pd.DataFrame): 包含数据的Pandas DataFrame。  
        xlabel (str): x轴的标签。  
  
    返回:  
        ax (matplotlib.axes.Axes): 绘图的Axes对象。  
    """  
    fig, ax = plt.subplots()  
    ax.set_title(title, fontsize=14)  # 设置图表标题和字体大小  
    for i in y:  
        ax.plot(x, df[i], label=str(i), marker='o')  # 绘制折线并添加标记  
    ax.set_xlabel(xlabel, color="black", fontsize=14)  # 设置x轴标签和字体样式  
    ax.set_ylabel("Value", color="black", fontsize=14)  # 设置y轴标签和字体样式  
    ax.legend()  # 添加图例  
    return ax  
  
# 创建一个简单的折线图  
def create_plot(x, y, title: str, xlabel: str, ylabel: str):  
    """  
    根据给定的x轴数据、y轴数据、标题、x轴标签和y轴标签，创建一个简单的折线图。  
  
    参数:  
        x (array-like): x轴的数据。  
        y (array-like): y轴的数据。  
        title (str): 图表的标题。  
        xlabel (str): x轴的标签。  
        ylabel (str): y轴的标签。  
  
    返回:  
        ax (matplotlib.axes.Axes): 绘图的Axes对象。  
    """  
    fig, ax = plt.subplots()  
    ax.plot(x, y)  # 绘制折线  
    ax.set_title(title)  # 设置图表标题  
    ax.set_xlabel(xlabel, color="black", fontsize=14)  # 设置x轴标签和字体样式  
    ax.set_ylabel(ylabel, color="black", fontsize=14)  # 设置y轴标签和字体样式  
    return ax
    
  
# 创建一个柱状图  
def create_bar(labels, data, title: str, xlabel: str, ylabel: str):  
    """  
    根据给定的标签、数据、标题、x轴标签和y轴标签，创建一个柱状图。  
  
    参数:  
        labels (list): 柱状图的x轴标签列表。  
        data (list): 柱状图的数据列表，与labels一一对应。  
        title (str): 图表的标题。  
        xlabel (str): x轴的标签。  
        ylabel (str): y轴的标签。  
  
    返回:  
        ax (matplotlib.axes.Axes): 绘图的Axes对象。  
    """  
    fig, ax = plt.subplots()  
    ax.bar(labels, data)  # 绘制柱状图  
    ax.set_title(title, fontsize=14)  # 设置图表标题和字体大小  
    ax.set_xlabel(xlabel, color='black', fontsize=14)  # 设置x轴标签和字体样式  
    ax.set_ylabel(ylabel, color="black", fontsize=14)  # 设置y轴标签和字体样式  
    return ax  
  
# 创建一个散点图  
import matplotlib.pyplot as plt  
from scipy.stats import linregress  
import numpy as np  
  
def create_scatter_with_regression(labels, data, title: str, xlabel: str, ylabel: str):  
  
    """  
  
    根据给定的标签、数据、标题、x轴标签和y轴标签，创建一个带有线性回归线的散点图。  
  
    参数:  
  
        labels (list): 散点图的x轴数据列表。  
  
        data (list): 散点图的y轴数据列表，与labels一一对应。  
  
        title (str): 图表的标题。  
  
        xlabel (str): x轴的标签。  
  
        ylabel (str): y轴的标签。  
  
    返回:  
  
        ax (matplotlib.axes.Axes): 绘图的Axes对象。  
  
    """  
  
    fig, ax = plt.subplots()  
    ax.scatter(labels, data)  # 绘制散点图  
    ax.set_title(title, fontsize=14)  # 设置图表标题和字体大小  
    ax.set_xlabel(xlabel, color='black', fontsize=14)  # 设置x轴标签和字体样式  
    ax.set_ylabel(ylabel, color="black", fontsize=14)  # 设置y轴标签和字体样式  
  
    # 检查labels是否全部相同  
    if len(set(labels)) == 1:  
        print(f"Warning: All {xlabel} values are identical. Cannot calculate a linear regression.")  
    else:  
        # 计算线性回归  
        slope, intercept, r_value, p_value, std_err = linregress(labels, data)  
  
        # 创建回归线的x值（通常与原始数据的x值范围相同）  
        x_fit = np.linspace(min(labels), max(labels), 100)  
        y_fit = slope * x_fit + intercept  
  
        # 在散点图上绘制回归线  
        ax.plot(x_fit, y_fit, 'r-', label=f'y = {slope:.2f}x + {intercept:.2f}')  
        ax.legend()  # 显示图例  
  
    return ax
  
# 创建一个多散点图  
def create_multiscatter(labels: list, xlabel: str, ylabel: str, title: str, df: pd.DataFrame):  
    """  
    根据给定的标签列表、x轴标签、y轴标签、标题和数据框，创建一个多散点图。  
  
    参数:  
        labels (list): 包含列名的列表，用于绘制多个散点图。  
        xlabel (str): x轴的标签，通常是时间或其他连续变量。  
        ylabel (str): y轴的标签。  
        title (str): 图表的标题。  
        df (pd.DataFrame): 包含数据的Pandas DataFrame，其中'time'列作为x轴数据。  
  
    返回:  
        ax (matplotlib.axes.Axes): 绘图的Axes对象。  
    """  
    fig, ax = plt.subplots()  
    ax.set_title(title, fontsize=14)  # 设置图表标题和字体大小  
    for label in labels:  # 遍历标签列表  
        ax.scatter(df['time'], df[label], label=label)  # 绘制散点图并添加标签  
    ax.set_xlabel(xlabel, color="black", fontsize=14)  # 设置x轴标签和字体样式  
    ax.set_ylabel(ylabel, color="black", fontsize=14)  # 设置y轴标签和字体样式  
    ax.legend()  # 添加图例  
    return ax

def create_time_multi_bar(labels: list, data: pd.DataFrame, xlabel: str, ylabel: str, title: str, width: float = 0.2):  
    """  
    根据给定的标签列表、数据框、x轴标签、y轴标签、标题和时间间隔宽度，  
    绘制一个时间序列的多个柱状图。  
  
    参数:  
    labels (list): 柱状图的x轴标签列表。  
    data (pd.DataFrame): 包含数据的Pandas DataFrame，其中列名代表年份。  
    xlabel (str): x轴的标签。  
    ylabel (str): y轴的标签。  
    title (str): 图表的标题。  
    width (float, optional): 每个柱状图的宽度。默认为0.2。  
  
    返回:  
    ax (matplotlib.axes.Axes): 绘图的Axes对象。  
    """  
    # 创建一个新的图形和Axes对象  
    fig, ax = plt.subplots()  
  
    # 初始化柱状图的偏移量  
    bar_offset = 0  
  
    # 遍历年份，绘制柱状图  
    for year in range(2014, 2018):  
        # 获取当前年份的数据  
        current_data = data[year]  
        # 绘制柱状图，并更新偏移量  
        bars = ax.bar([x + bar_offset for x in range(len(labels))], current_data, width=width, label=str(year))  
        autolabel(bars, ax=ax)  
        bar_offset += width  
  
    # 设置x轴刻度位置在柱状图的中间，并设置x轴标签  
    ax.set_xticks([x + bar_offset / 2 - (len(range(2014, 2018)) - 1) * width / 2 for x in range(len(labels))])  
    ax.set_xticklabels(labels)  
  
    # 添加图例  
    ax.legend()  
  
    # 设置标题、x轴和y轴的标签  
    ax.set_title(title, fontsize=14)  
    ax.set_xlabel(xlabel, color='black', fontsize=14)  
    ax.set_ylabel(ylabel, color="black", fontsize=14)  
  
    # 返回Axes对象以便进一步操作（如果需要）  
    return ax  
        

def create_multi_bar(labels: list, data: pd.DataFrame, xlabel: str, ylabel: str, title: str,width:float=0.2):  
    """  
    绘制多个城市的柱状图  
  
    参数:  
    labels (list): 柱状图的x轴标签列表  
    data (pd.DataFrame): 包含数据的Pandas DataFrame，其中列名代表不同的城市  
    xlabel (str): x轴的标签  
    ylabel (str): y轴的标签  
    title (str): 图表的标题  
  
    返回:  
    ax (matplotlib.axes.Axes): 绘图的Axes对象  
    """  
    fig, ax = plt.subplots()  # 创建一个新的图形和Axes对象  
    bar_width = 0.3  # 每个柱状图的宽度  
    index = list(range(len(labels)))  # 索引列表，用于柱状图的x位置  
    bar_positions = []  # 用于存储每个城市的柱状图位置  
  
    # 遍历城市名称和数据列  
    for i, city in enumerate(li):  
        # 绘制柱状图  
        bars = ax.bar([x + (i+1) * bar_width for x in index], data[city], width=bar_width, label=city)  
        autolabel(bars, ax=ax)  # 添加数值标签  
        bar_positions.append(i * bar_width)  # 存储位置以便设置x轴刻度  
  
    # 计算x轴刻度的中间位置  
    tick_positions = [x + sum(bar_positions) / len(bar_positions) for x in index]  
    # 设置x轴刻度位置和标签  
    ax.set_xticks(tick_positions)  
    ax.set_xticklabels(labels)  
  
    # 添加图例  
    ax.legend()  
  
    # 设置标题、x轴和y轴的标签  
    ax.set_title(title, fontsize=14)  
    ax.set_xlabel(xlabel, color='black', fontsize=14)  
    ax.set_ylabel(ylabel, color="black", fontsize=14)  
  
    # 返回Axes对象以便进一步操作（如果需要）  
    return ax 

def create_multi_bar2(labels: list, data: pd.DataFrame, xlabel: str, ylabel: str, title: str, width: float):  
    # 创建一个新的图形和坐标轴  
    fig, ax = plt.subplots()  
      
    # 设置条形图的宽度  
    bar_width = width  
      
    # 创建一个与labels长度相同的索引列表  
    index = list(range(len(labels)))  
      
    # 初始化一个变量，用于在循环中移动条形图的起始位置  
    tmp = 1  
      
    # 遍历DataFrame的每一列  
    for i in data.columns:  
        # 在当前索引和tmp计算的基础上绘制条形图  
        # 注意这里使用了列表推导式来为每个条形图计算x位置  
        a = ax.bar([x + bar_width * tmp for x in index], data[i], width=bar_width, label=i)  
          
        # 在条形图上添加数据标签  
        autolabel(a, ax=ax)  
          
        # 增加tmp的值，以便下一个条形图可以放置在正确的位置  
        tmp += 1  
      
    # 设置x轴的刻度位置，使其位于条形图的中间  
    # 注意这里我们除以6（可能是一个为了美观的调整，但通常我们除以2）  
    ax.set_xticks([x + bar_width / 2 for x in index])  # 通常使用 / 2 而不是 / 6  
      
    # 设置x轴的标签为传入的labels列表  
    ax.set_xticklabels(labels)  
      
    # 显示图例  
    ax.legend()  
      
    # 设置图形的标题  
    ax.set_title(title, fontsize=14)  
      
    # 设置x轴的标签和字体样式  
    ax.set_xlabel(xlabel, color='black', fontsize=14)  
      
    # 设置y轴的标签和字体样式  
    ax.set_ylabel(ylabel, color="black", fontsize=14)  
    
    # 返回坐标轴对象，以便在外部可以进一步修改或保存图形  
    return ax

def df_simplify_average(df: pd.DataFrame, var: str):  

    """  

    计算给定 DataFrame 中每个年份每个月的指定变量的平均值。  

  

    参数:  

    df (pd.DataFrame): 输入的 DataFrame。  

    var (str): 需要计算平均值的列名。  

  

    返回:  

    pd.DataFrame: 一个 DataFrame，其中行索引为月份（但此处应传入正确的月份列表），  

                   列标签为年份，值为对应月份和年份的平均值。  

    """  

    tmp = {}  

    for i in year:  # 遍历年份  

        df_year = df_time_split(df=df, time=i)  # 根据年份分割数据  

        month_avg_aqi = df_year.groupby(pd.Grouper(freq='M'))[var].mean().tolist()  # 计算每月的平均值  

        tmp.update({i: month_avg_aqi})  # 将结果添加到字典中  

    tmp = pd.DataFrame(tmp, index=month)  # 将字典转换为 DataFrame  

    return tmp 

def df_simplify_median(df:pd.DataFrame,var:str):
    tmp={}
    for i in year:
        df_year=df_time_split(df=df,time=i)
        month_avg_aqi=df_year.groupby(pd.Grouper(freq='M'))[var].median().tolist()
        tmp.update({i:month_avg_aqi})
    tmp=pd.DataFrame(tmp,index=month)
    return tmp
        
def df_simplify_std(df:pd.DataFrame,var:str):
    tmp={}
    for i in year:
        df_year=df_time_split(df=df,time=i)
        month_avg_aqi=df_year.groupby(pd.Grouper(freq='M'))[var].std().tolist()
        tmp.update({i:month_avg_aqi})
    tmp=pd.DataFrame(tmp,index=month)
    return tmp

def df_simplify_max(df:pd.DataFrame,var:str):
    tmp={}
    for i in year:
        df_year=df_time_split(df=df,time=i)
        month_avg_aqi=df_year.groupby(pd.Grouper(freq='M'))[var].max().tolist()
        tmp.update({i:month_avg_aqi})
    tmp=pd.DataFrame(tmp,index=month)
    return tmp

def df_simplify_min(df:pd.DataFrame,var:str):
    tmp={}
    for i in year:
        df_year=df_time_split(df=df,time=i)
        month_avg_aqi=df_year.groupby(pd.Grouper(freq='M'))[var].min().tolist()
        tmp.update({i:month_avg_aqi})
    tmp=pd.DataFrame(tmp,index=month)
    return tmp



# 数据回归分析
# 数据处理函数（单个特征）  
def datapro(x: str, y: str, df: pd.DataFrame) -> tuple:  
    """  
    处理单个特征，划分训练集和测试集。  
  
    参数:  
    x (str): 要使用的特征名  
    y (str): 目标列名  
    df (pd.DataFrame): 数据框  
  
    返回:  
    tuple: 包含四个元素的元组 (X_train, X_test, y_train, y_test)  
    """  
    X = df[[x]].copy()  # 从数据框中复制特征x的数据  
    Y = df[y]         # 从数据框中获取目标列y的数据  
    X.fillna(0, inplace=True)  # 将特征X中的缺失值填充为0  
    Y.fillna(0, inplace=True)  # 将目标列Y中的缺失值填充为0  
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)  # 划分训练集和测试集  
    return X_train, X_test, y_train, y_test  
  
# 数据处理函数（多个特征）  
def datapro_s(x: list, y: str, df: pd.DataFrame) -> tuple:  
    """  
    处理多个特征，划分训练集和测试集。  
  
    参数:  
    x (list): 要使用的特征名列表  
    y (str): 目标列名  
    df (pd.DataFrame): 数据框  
  
    返回:  
    tuple: 包含四个元素的元组 (X_train, X_test, y_train, y_test)  
    """  
    X = df[x]  # 从数据框中获取特征列表x的数据  
    Y = df[y]  # 从数据框中获取目标列y的数据  
    X.fillna(0, inplace=True)  # 将特征X中的缺失值填充为0  
    Y.fillna(0, inplace=True)  # 将目标列Y中的缺失值填充为0  
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)  # 划分训练集和测试集  
    return X_train, X_test, y_train, y_test

# 处理多个特征数据，并划分训练集和测试集  
def datapro_s(x: list, y: str, df: pd.DataFrame) -> tuple:  
    """  
    从DataFrame中选取多个特征，并划分训练集和测试集。  
  
    参数:  
    x (list): 特征名列表  
    y (str): 目标列名  
    df (pd.DataFrame): 输入的DataFrame  
  
    返回:  
    tuple: 包含四个元素的元组 (X_train, X_test, y_train, y_test)  
    """  
    X = df[x].copy()  # 选取DataFrame中的特征  
    Y = df[y].copy()  # 选取DataFrame中的目标列  
    X.fillna(0, inplace=True)  # 将特征X中的缺失值填充为0  
    Y.fillna(0, inplace=True)  # 将目标列Y中的缺失值填充为0  
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)  # 划分数据集  
    return X_train, X_test, y_train, y_test  
  
# 初始化线性回归模型  
def model_init() -> LinearRegression:  
    """  
    初始化并返回一个LinearRegression模型实例。  
  
    返回:  
    LinearRegression: 线性回归模型实例  
    """  
    model = LinearRegression()  
    return model  
  
# 使用训练数据训练模型  
def model_training(model: LinearRegression, xt_train, y_train):  
    """  
    使用给定的训练数据来训练线性回归模型。  
  
    参数:  
    model (LinearRegression): 要训练的线性回归模型实例  
    xt_train: 训练数据的特征  
    y_train: 训练数据的目标值  
  
    返回:  
    None  
    """  
    model.fit(xt_train, y_train)  
  
# 评估模型在测试集上的性能  
def model_judge(model: LinearRegression, X_test, y_test):  
    """  
    评估模型在测试集上的性能，并打印均方误差(MSE)和R^2分数。  
  
    参数:  
    model (LinearRegression): 已训练的线性回归模型实例  
    X_test: 测试数据的特征  
    y_test: 测试数据的目标值  
  
    返回:  
    None  
    """  
    y_pred = model.predict(X_test)  # 预测测试集的结果  
    mse = mean_squared_error(y_test, y_pred)  # 计算均方误差  
    r2 = r2_score(y_test, y_pred)  # 计算R^2分数  
    print(f'Mean Squared Error: {mse}')  # 打印均方误差  
    print(f'R2 Score: {r2}')  # 打印R^2分数  
  
# 使用模型进行预测  
def model_predict(model: LinearRegression, data: any) -> any:  
    """  
    使用已训练的模型对新的数据进行预测。  
  
    参数:  
    model (LinearRegression): 已训练的线性回归模型实例  
    data: 要进行预测的数据  
  
    返回:  
    any: 预测结果  
    """  
    predicted_result = model.predict(data)  # 预测结果  
    return predicted_result



    

    
    
    


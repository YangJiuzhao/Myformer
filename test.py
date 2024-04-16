import os
import numpy as np
import pandas as pd

# root_path = './datasets/'
# data_path = 'machine_usage.csv'

# df_raw = pd.read_csv(os.path.join(root_path,data_path),header=None)#,index_col=['machine_id','time','cpu_util_percent','mem_util_percent','mem_gps','mkpi','net_in','net_out','disk_io_percent'])
# df = df_raw[['machine_id','cpu_util_percent','mem_util_percent','net_in','net_out','disk_io_percent']]

# target_machineID = 'm_1999'
# target_select = df['machine_id'].str.contains(target_machineID)
# target = df[target_select]
# target.to_csv(target_machineID+'.csv',mode='a',header=None)
# print()

df = pd.read_csv('m1999.csv',header=None)
df = df[[2,3,4,5,6,7]]
df['time'] = pd.to_timedelta(df[2], unit='s')
# 将时间列设置为索引
df = df.set_index('time')

# 设置时间间隔，这里假设为1分钟
interval = '1H'

# 使用resample函数进行时间重采样，并使用插值方法填充缺失值
df_resampled = df.resample(interval).interpolate()

# 重置索引，恢复时间间隔为列
df_resampled = df_resampled.reset_index()

# 打印处理后的数据框
print(df_resampled)
print(df_resampled)
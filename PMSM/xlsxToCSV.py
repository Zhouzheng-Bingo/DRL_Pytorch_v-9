import pandas as pd

# # 读取xlsx文件
# xlsx_file = 'data/电机转动角度(弧度).xlsx'
# df = pd.read_excel(xlsx_file)
#
# # 将数据保存为csv文件
# csv_file = 'data/电机转动角度(弧度).csv'
# df.to_csv(csv_file, index=False)
#
# # 读取xlsx文件
# xlsx_file = 'data/电机转速(转秒).xlsx'
# df = pd.read_excel(xlsx_file)
#
# # 将数据保存为csv文件
# csv_file = 'data/电机转速(转秒).csv'
# df.to_csv(csv_file, index=False)
#
# # 读取xlsx文件
# xlsx_file = 'data/电流id.xlsx'
# df = pd.read_excel(xlsx_file)
#
# # 将数据保存为csv文件
# csv_file = 'data/电流id.csv'
# df.to_csv(csv_file, index=False)
#
# # 读取xlsx文件
# xlsx_file = 'data/电流iq.xlsx'
# df = pd.read_excel(xlsx_file)
#
# # 将数据保存为csv文件
# csv_file = 'data/电流iq.csv'
# df.to_csv(csv_file, index=False)

import pandas as pd

# 读取xlsx文件
xlsx_file = 'data/电机转动角度(弧度).xlsx'
df = pd.read_excel(xlsx_file)

# 截取前1万行数据
df_subset = df.head(10000)

# 将截取的数据保存为新的csv文件
csv_file = 'data/电机转动角度(弧度)_subset.csv'
df_subset.to_csv(csv_file, index=False)

# 读取xlsx文件
xlsx_file = 'data/电机转速(转秒).xlsx'
df = pd.read_excel(xlsx_file)

# 截取前1万行数据
df_subset = df.head(10000)

# 将截取的数据保存为新的csv文件
csv_file = 'data/电机转速(转秒)_subset.csv'
df_subset.to_csv(csv_file, index=False)

# 读取xlsx文件
xlsx_file = 'data/电流id.xlsx'
df = pd.read_excel(xlsx_file)

# 截取前1万行数据
df_subset = df.head(10000)

# 将截取的数据保存为新的csv文件
csv_file = 'data/电流id_subset.csv'
df_subset.to_csv(csv_file, index=False)

# 读取xlsx文件
xlsx_file = 'data/电流iq.xlsx'
df = pd.read_excel(xlsx_file)

# 截取前1万行数据
df_subset = df.head(10000)

# 将截取的数据保存为新的csv文件
csv_file = 'data/电流iq_subset.csv'
df_subset.to_csv(csv_file, index=False)

print('Done!')

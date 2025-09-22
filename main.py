from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, kruskal

# Dataset
train_ds = read_csv('Datasets/train.csv', sep=';')
test_ds = read_csv('Datasets/test.csv', sep=';')


train_ds['y'] = train_ds['y'].map({'yes': 1, 'no': 0})

client_stats_columns = [
    'job',
    'marital',
    'education',
    'default',
    'housing',
    'loan',
]

train_ds['age_range'] = pd.cut(
    train_ds['age'],
    bins=[17,30,40,50,60,100],
    labels=['18-30','31-40','41-50','51-60','60+']
)

print('--------------------')
print('Conversion rate per age')
print(train_ds.groupby('age_range').agg(
    conversion_rate=('y', 'mean')
).sort_values('conversion_rate', ascending=False))
print('--------------------')

for column in client_stats_columns:
    print('--------------------')
    print(f"Conversion rate by '{column}':")
    stats = train_ds.groupby(column).agg(
        total_clients=("y", 'size'),
        conversion_rate=("y", 'mean'),
        avg_age=("age", "mean"),
        avg_balance=("balance", "mean")
    ).sort_values("conversion_rate", ascending=False)
    print(stats)
    print('--------------------')

    # Barplot de tasa de conversión por categoría
    plt.figure(figsize=(8,4))
    sns.barplot(x=stats.index, y=stats['conversion_rate'], palette="viridis")
    plt.title(f"Tasa de conversión por {column}")
    plt.xticks(rotation=45)
    plt.show()

train_ds['day_ranges'] = pd.cut(
    train_ds['day'],
    bins=[0, 7, 15, 23, 31],
    labels=['1-7','8-15','16-23','24-31']
)

print('--------------------')
print('Conversion rate per day')
print(train_ds.groupby('day_ranges').agg(
    conversion_rate=('y', 'mean')
).sort_values('day_ranges', ascending=True))
print('--------------------')

contact_statistics_columns = [
    'contact',
    'month',
]

for column in contact_statistics_columns:
    print('--------------------')
    print(f'Conversion rate by {column}')
    stats = train_ds.groupby(column).agg(
        conversion_rate=('y', 'mean'),
    ).sort_values('conversion_rate', ascending=False)
    print(stats)
    print('--------------------')

    # Pie chart de distribución de contactos
    plt.figure(figsize=(5,5))
    train_ds[column].value_counts().plot.pie(
        autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel")
    )
    plt.title(f"Distribución de {column}")
    plt.ylabel("")
    plt.show()

# 1) Histograma de edad
plt.figure(figsize=(6,4))
sns.histplot(train_ds['age'], bins=30, kde=True)
plt.title("Distribución de edad")
plt.show()

# 2) Boxplot de balance (para ver outliers)
plt.figure(figsize=(6,4))
sns.boxplot(x=train_ds['balance'])
plt.title("Boxplot de balance")
plt.show()

# 3) Scatter: relación entre edad y balance, coloreado por resultado
plt.figure(figsize=(6,4))
sns.scatterplot(x='age', y='balance', hue='y', data=train_ds, alpha=0.5, palette="Set1")
plt.title("Edad vs Balance por resultado (y)")
plt.show()

group_yes = train_ds[train_ds['housing'] == 'yes']['y']
group_no = train_ds[train_ds['housing'] == 'no']['y']

t_stat, p_val = ttest_ind(group_yes, group_no, equal_var=False)
print("T-test Housing loan vs No loan")
print("t = {:.3f}, p = {:.5f}".format(t_stat, p_val))

groups = [group['y'].values for name, group in train_ds.groupby('education')]
h_stat, p_val = kruskal(*groups)
print("Kruskal-Wallis Education")
print("H = {:.3f}, p = {:.5f}".format(h_stat, p_val))
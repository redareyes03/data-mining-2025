from pandas import read_csv
import pandas as pd

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
    print("Conversion rate by '{}':".format(column))
    print(train_ds.groupby(column).agg(
        total_clients=("y", 'size'),
        conversion_rate=("y", 'mean'),
        avg_age=("age", "mean"),
        avg_balance=("balance", "mean")
    ).sort_values("conversion_rate", ascending=False))
    print('--------------------')

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
    print('Conversion rate by {}'.format(column))
    print(train_ds.groupby(column).agg(
        conversion_rate=('y', 'mean'),
    ).sort_values('conversion_rate', ascending=False))
    print('--------------------')

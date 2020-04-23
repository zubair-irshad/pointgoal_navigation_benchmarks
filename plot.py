from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd

df_train = pd.read_csv('/home/mirshad7/Downloads/run-.-tag-Loss.csv')
df_val = pd.read_csv('/home/mirshad7/Downloads/run-.-tag-Mean_Val_Loss.csv')

fig = make_subplots(rows=1, cols=2)

fig.add_trace(
    go.Scatter(x=df_train['Step'], y=df_train['Value']),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=df_val['Step'], y=df_val['Value']),
    row=1, col=2
)
fig.show()
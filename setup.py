import sqlalchemy as db
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import pandas as pd
from config import password
import datetime as dt
import psycopg2

DATABASE_CONNECTION = f"postgresql://postgres:{password}@127.0.0.1:5432/Portfolio"
engine = create_engine(DATABASE_CONNECTION)
session = Session(engine)

a = db.MetaData()
AAPL_imp = db.Table('aapl', a, autoload=True, autoload_with=engine)
DOG_imp = db.Table('dog', a, autoload=True, autoload_with=engine)
FB_imp = db.Table('fb', a, autoload=True, autoload_with=engine)
GOOGL_imp = db.Table('googl', a, autoload=True, autoload_with=engine)
KO_imp = db.Table('ko', a, autoload=True, autoload_with=engine)
MSFT_imp = db.Table('msft', a, autoload=True, autoload_with=engine)
PG_imp = db.Table('pg', a, autoload=True, autoload_with=engine)
SPXS_imp = db.Table('spxs', a, autoload=True, autoload_with=engine)
SPY_imp = db.Table('spy', a, autoload=True, autoload_with=engine)
T_imp = db.Table('t', a, autoload=True, autoload_with=engine)
TLT_imp = db.Table('tlt', a, autoload=True, autoload_with=engine)

q1 = db.select([AAPL_imp])
q2 = db.select([DOG_imp])
q3 = db.select([FB_imp])
q4 = db.select([GOOGL_imp])
q5 = db.select([KO_imp])
q6 = db.select([MSFT_imp])
q7 = db.select([PG_imp])
q8 = db.select([SPXS_imp])
q9 = db.select([SPY_imp])
q10 = db.select([T_imp])
q11 = db.select([TLT_imp])

rp1= session.execute(q1)
rp2= session.execute(q2)
rp3= session.execute(q3)
rp4= session.execute(q4)
rp5= session.execute(q5)
rp6= session.execute(q6)
rp7= session.execute(q7)
rp8= session.execute(q8)
rp9= session.execute(q9)
rp10= session.execute(q10)
rp11= session.execute(q11)

rs1 = rp1.fetchall()
rs2 = rp2.fetchall()
rs3 = rp3.fetchall()
rs4 = rp4.fetchall()
rs5 = rp5.fetchall()
rs6 = rp6.fetchall()
rs7 = rp7.fetchall()
rs8 = rp8.fetchall()
rs9 = rp9.fetchall()
rs10 = rp10.fetchall()
rs11 = rp11.fetchall()

AAPL = pd.DataFrame(rs1)
AAPL.columns = rs1[0].keys()
DOG = pd.DataFrame(rs2)
DOG.columns = rs2[0].keys()
FB = pd.DataFrame(rs3)
FB.columns = rs3[0].keys()
GOOGL = pd.DataFrame(rs4)
GOOGL.columns = rs4[0].keys()
KO = pd.DataFrame(rs5)
KO.columns = rs5[0].keys()
MSFT = pd.DataFrame(rs6)
MSFT.columns = rs6[0].keys()
PG = pd.DataFrame(rs7)
PG.columns = rs7[0].keys()
SPXS = pd.DataFrame(rs8)
SPXS.columns = rs8[0].keys()
SPY = pd.DataFrame(rs9)
SPY.columns = rs9[0].keys()
T = pd.DataFrame(rs10)
T.columns = rs10[0].keys()
TLT = pd.DataFrame(rs11)
TLT.columns = rs11[0].keys()

AAPL_DF = AAPL.rename(columns={"adj_close": "Apple"})
DOG_DF = DOG.rename(columns={"adj_close": "Pro Shares Dow Short"})
FB_DF = FB.rename(columns={"adj_close": "Facebook"})
GOOGL_DF = GOOGL.rename(columns={"adj_close": "Google"})
KO_DF = KO.rename(columns={"adj_close": "Coca Cola"})
MSFT_DF = MSFT.rename(columns={"adj_close": "Microsoft"})
PG_DF = PG.rename(columns={"adj_close": "Proctor & Gamble"})
SPXS_DF = SPXS.rename(columns={"adj_close": "Direxion S&P 500 Short"})
SPY_DF = SPY.rename(columns={"adj_close": "S&P 500 ETF"})
T_DF = T.rename(columns={"adj_close": "AT&T"})
TLT_DF = TLT.rename(columns={"adj_close": "Treasury Yield ETF"})

Raw_DF = pd.concat([AAPL_DF, DOG_DF, FB_DF, GOOGL_DF, KO_DF, MSFT_DF, PG_DF, SPXS_DF, SPY_DF, T_DF, TLT_DF], axis = 1)

Main_DF= Raw_DF[['Apple','Pro Shares Dow Short','Facebook','Google','Coca Cola','Microsoft','Proctor & Gamble','Direxion S&P 500 Short','S&P 500 ETF','AT&T','Treasury Yield ETF']].dropna()
Main_DF

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

y = Main_DF['S&P 500 ETF']
x = Main_DF.drop(columns = 'S&P 500 ETF')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,
   y, random_state=0)

model = LinearRegression()

model.fit(x, y)

y_pred = model.predict(x)
print(y_pred.shape)

print(model.coef_)
print(model.intercept_)

colors = ['Positive' if c > 0 else 'Negative' for c in model.coef_]

fig = px.bar(
    x=x.columns, y=model.coef_, color=colors,
    color_discrete_sequence=['blue', 'red'],
    labels=dict(x='Feature', y='Linear coefficient'),
    title='Multiple Regression'
)
fig.show()

fig1 = px.scatter(x=y, y=y_pred, labels={'x': 'actual', 'y': 'prediction'}, title = 'Actual vs Predicted')
fig1.add_shape(
    type="line", line=dict(dash='dash'),
    x0=y.min(), y0=y.min(),
    x1=y.max(), y1=y.max(),
)
fig1.show()

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import hvplot.pandas

Main_DF_Scaled = StandardScaler().fit_transform(Main_DF)
Main_DF_Scaled 

pca = PCA(n_components=2)

Main_DF_pca = pca.fit_transform(Main_DF_Scaled)
Main_DF_pca

Main_DF_pca_DF = pd.DataFrame(data = Main_DF_pca, columns = ['PC1','PC2'])
Main_DF_pca_DF

pca.explained_variance_ratio_

# Find the best value for K
inertia = []
k = list(range(1, 11))

# Calculate the inertia for the range of K values
for i in k:
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(Main_DF_pca_DF)
    inertia.append(km.inertia_)

# Create the elbow curve
elbow_data = {"k": k, "inertia": inertia}
df_elbow = pd.DataFrame(elbow_data)
df_elbow.hvplot.line(x="k", y="inertia", xticks=k, title="Elbow Curve")

# Initialize the K-means model
model = KMeans(n_clusters=3, random_state=0)

# Fit the model
model.fit(Main_DF_pca_DF)

# Predict clusters
predictions = model.predict(Main_DF_pca_DF)

# Add the predicted class columns
Main_DF_pca_DF["class"] = model.labels_
Main_DF_pca_DF

Main_DF_pca_DF.hvplot.scatter(
    x="PC1",
    y="PC2",
    hover_cols=["class"],
    by="class",
)




import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors

df = Main_DF['S&P 500 ETF']
X = Main_DF['Apple']
X_train, X_test, y_train, y_test = train_test_split(
X, df.tip, random_state=42)

models = {'Regression': linear_model.LinearRegression,
          'Decision Tree': tree.DecisionTreeRegressor,
          'k-NN': neighbors.KNeighborsRegressor}

app = dash.Dash(__name__)

app.layout = html.Div([
    html.P("Select Model:"),
    dcc.Dropdown(
        id='model-name',
        options=[{'label': x, 'value': x} 
                 for x in models],
        value='Regression',
        clearable=False
    ),
    dcc.Graph(id="graph"),
])

@app.callback(
    Output("graph", "figure"), 
    [Input('model-name', "value")])
def train_and_display(name):
    model = models[name]()
    model.fit(X_train, y_train)

    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train, 
                   name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, 
                   name='test', mode='markers'),
        go.Scatter(x=x_range, y=y_range, 
                   name='prediction')
    ])

    return fig

app.run_server(debug=True)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import scipy.stats as stats
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
import statsmodels.graphics.gofplots as smg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


import plotly.graph_objects as go
from plotly.subplots import make_subplots

county_opts = ["All"] + [str(x) for x in range(16)]
product_type_opts = ["All"] + [str(x) for x in range(4)]
is_business_opts = ["All"] + [str(x) for x in range(2)]
is_consumption_opts = ["All"] + [str(x) for x in range(2)]
installed_capacity_opts = ["Show", "Hide"]




# col_pca_list = ['target','eic_count', 'installed_capacity',
#     'euros_per_mwh', 'temperature_fcast_mean', 'dewpoint_fcast_mean',
#     'cloudcover_total_fcast_mean', 'direct_solar_radiation_fcast_mean',
#     'surface_solar_radiation_downwards_fcast_mean', 'snowfall_fcast_mean',
#     'temperature_hist_mean', 'rain_hist_mean', 'snowfall_hist_mean',
#     'shortwave_radiation_hist_mean', 'diffuse_radiation_hist_mean',
#     'windspeed_10m_hist_mean_by_county',
#     'direct_solar_radiation_hist_mean_by_county']



selected_columns = ['county', 'datetime', 'is_business', 'product_type', 'target',
       'is_consumption', 'prediction_unit_id', 'month', 'day', 'hour',
       'dayofweek', 'dayofyear', 'eic_count', 'installed_capacity',
       'euros_per_mwh', 'temperature_fcast_mean', 'dewpoint_fcast_mean',
       'cloudcover_total_fcast_mean', 'direct_solar_radiation_fcast_mean',
       'surface_solar_radiation_downwards_fcast_mean', 'snowfall_fcast_mean',
       'temperature_hist_mean', 'rain_hist_mean', 'snowfall_hist_mean',
       'shortwave_radiation_hist_mean', 'diffuse_radiation_hist_mean',
       'windspeed_10m_hist_mean_by_county',
       'direct_solar_radiation_hist_mean_by_county']



selected_columns_pca = ['eic_count', 'installed_capacity',
       'euros_per_mwh', 'temperature_fcast_mean', 'dewpoint_fcast_mean',
       'cloudcover_total_fcast_mean', 'direct_solar_radiation_fcast_mean',
       'surface_solar_radiation_downwards_fcast_mean', 'snowfall_fcast_mean',
       'temperature_hist_mean', 'rain_hist_mean', 'snowfall_hist_mean',
       'shortwave_radiation_hist_mean', 'diffuse_radiation_hist_mean',
       'windspeed_10m_hist_mean_by_county',
       'direct_solar_radiation_hist_mean_by_county']







client = pd.read_csv(r'C:\Users\Asus\Downloads\client_df.csv', parse_dates=True)
train = pd.read_csv(r'C:\Users\Asus\Downloads\training.csv', parse_dates=True)

client = client[:20000]
train = train[:20000]

train = train.pivot_table(index="datetime", columns=["county", "product_type", "is_business", "is_consumption"], values="target")
client = client.pivot_table(index="date", columns=["county", "product_type", "is_business"], values="installed_capacity")

df = pd.read_csv(r'C:\Users\Asus\Downloads\enefit_project_train.csv')




df = df.fillna(0)

df_test = df[:5000]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Initialize the Dash app
app = dash.Dash(name='Energy imbalance', external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H2("Energy Imbalance of Prosumers", style={'textAlign': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Data Visualization', children=[
            dcc.Dropdown(
                id='column-dropdown',
                options=[{'label': col, 'value': col} for col in df.columns],
                value=df.columns[3]  # Default value for the column dropdown
            ),
            html.Br(),
            dcc.Dropdown(
                id='plot-type-dropdown',
                options=[
                    {'label': 'KDE Plot', 'value': 'kde'},
                    {'label': 'Box Plot', 'value': 'box'},
                    {'label': 'Histogram Plot', 'value': 'hist'},
                    {'label': 'QQ Plot', 'value': 'qqplot'},
                    {'label': 'Regression Plot', 'value': 'regplot'},
                    {'label': 'Line Plot', 'value': 'lineplot'},
                    {'label': 'Pie Chart', 'value': 'pie'}
                ],
                value='kde'  # Default value for the plot type dropdown
            ),
            dcc.Graph(id='plot-output')
        ]),
        dcc.Tab(label='Normality Test', children=[
            html.H2("Normality Test"),
            html.Div([
                html.Label("Select Column(s):"),
                dcc.Dropdown(
                    id='test-columns-dropdown',
                    options=[{'label': col, 'value': col} for col in df.columns],
                    value=[df.columns[0]],  # Default value for the test column dropdown
                    multi=True  # Allow multiple selections
                ),
            ]),
            html.Br(),
            html.Div([
                html.Label("Select Test Type:"),
                dcc.Dropdown(
                    id='test-type-dropdown',
                    options=[
                        {'label': 'Shapiro-Wilk Test', 'value': 'shapiro'},
                        {'label': 'D’Agostino’s K^2 Test', 'value': 'dagostino'},
                        {'label': 'Kolmogorov-Smirnov Test', 'value': 'ks'}
                    ],
                    value='shapiro',  # Default value for the test type dropdown
                ),
            ]),
            html.Div(id='normality-test-result')
        ]),
        dcc.Tab(label='Consumption Analysis', children=[
            dcc.Loading(
                id="loading",
                type="default",
                children=html.Div(id="loading-output")
            ),
            html.Div(className="row", children=[
                html.Div(className="column", children=[
                    html.Label("county"),
                    dcc.Dropdown(
                        id='county-dropdown',
                        options=[{'label': opt, 'value': opt} for opt in county_opts],
                        value="All",
                        clearable=False
                    )], style={'width': '24%', 'display': 'inline-block', 'padding': '0 5px'}),
                html.Div(className="column", children=[
                    html.Label("product_type"),
                    dcc.Dropdown(
                        id='product-type-dropdown',
                        options=[{'label': opt, 'value': opt} for opt in product_type_opts],
                        value="All",
                        clearable=False
                    )], style={'width': '24%', 'display': 'inline-block', 'padding': '0 5px'}),
                html.Div(className="column", children=[
                    html.Label("is_business"),
                    dcc.Dropdown(
                        id='is-business-dropdown',
                        options=[{'label': opt, 'value': opt} for opt in is_business_opts],
                        value="All",
                        clearable=False
                    )], style={'width': '24%', 'display': 'inline-block', 'padding': '0 5px'}),
                html.Div(className="column", children=[
                    html.Label("is_consumption"),
                    dcc.Dropdown(
                        id='is-consumption-dropdown',
                        options=[{'label': opt, 'value': opt} for opt in is_consumption_opts],
                        value="All",
                        clearable=False
                    )], style={'width': '24%', 'display': 'inline-block', 'padding': '0 5px'}),
                html.Div(className="column", children=[
                    html.Label("installed_capacity"),
                    dcc.Dropdown(
                        id='installed-capacity-dropdown',
                        options=[{'label': opt, 'value': opt} for opt in installed_capacity_opts],
                        value="Show",
                        clearable=False
                    )], style={'width': '24%', 'display': 'inline-block', 'padding': '0 5px'})
            ], style={'display': 'flex'}),
            html.Div(className="row", children=[
                html.Div([
                    dcc.Graph(id="graph")
                ])
            ]),
        ]),
        dcc.Tab(label='PCA Analysis', children=[
            html.Div([
                html.H3('PCA Analysis'),
                html.Label('Select range of components:'),
                dcc.RangeSlider(
                    id='num-components-slider',
                    min=1,
                    max=len(selected_columns_pca),
                    step=1,
                    value=[1, len(selected_columns_pca)],  # Default value to select all components
                    marks={i: str(i) for i in range(1, len(selected_columns_pca) + 1)}
                ),
                dcc.Graph(id='pca-plot')
            ])
        ]),
        
        # dcc.Tab(label='Outlier Correction', children=[
        #     html.Div([
        #         html.H3("Outlier Correction"),
        #         html.Label("Select column for outlier correction:"),
        #         dcc.Dropdown(
        #             id='outlier-column-dropdown',
        #             options=[{'label': col, 'value': col} for col in selected_columns_pca],
        #             value=selected_columns_pca[0]  # Default value for the column dropdown
        #         ),
        #         dcc.Graph(id='boxplot-before'),
        #         dcc.Graph(id='boxplot-after')
        #     ])
        # ]),
        
        dcc.Tab(label='Know Your Data', children=[
            html.H3("Know Your Data"),
            html.Div([
                html.Label("Select what you want to know about your data:"),
                dcc.RadioItems(
                    id='data-info-radio',
                    options=[
                        {'label': 'Number of Columns', 'value': 'num_columns'},
                        {'label': 'Number of Rows', 'value': 'num_rows'},
                        {'label': 'Number of NaN Values', 'value': 'num_nan'},
                        {'label': 'Column Statistics', 'value': 'column_stats'}
                    ],
                    value='num_columns',
                    labelStyle={'display': 'block'}
                )
            ]),
            html.Br(),
            html.Div(id='data-info-output'),
            html.Br(),
            html.Button('Download CSV', id='download-button'),
            dcc.Download(id="download")
        ])
    ])
])










@app.callback(
    Output('data-info-output', 'children'),
    [Input('data-info-radio', 'value')])
def provide_data_info(selected_info):
    if selected_info == 'num_columns':
        return html.Div(f"Number of Columns: {len(df.columns)}")
    elif selected_info == 'num_rows':
        return html.Div(f"Number of Rows: {len(df)}")
    elif selected_info == 'num_nan':
        nan_count = df.isnull().sum().sum()
        return html.Div(f"Number of NaN Values: {nan_count}")
    elif selected_info == 'column_stats':
        stats = df.describe().reset_index().melt(id_vars='index', var_name='column', value_name='statistic')
        return html.Table([
            html.Thead(
                html.Tr([html.Th(col) for col in stats.columns])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(stats.iloc[i][col]) for col in stats.columns
                ]) for i in range(len(stats))
            ])
        ])







# @app.callback(
#     Output('data-info-output', 'children'),
#     [Input('data-info-radio', 'value')])


# def provide_data_info(selected_info):
#     if selected_info == 'num_columns':
#         return html.Div(f"Number of Columns: {len(df.columns)}")
#     elif selected_info == 'num_rows':
#         return html.Div(f"Number of Rows: {len(df)}")
#     elif selected_info == 'num_nan':
#         nan_count = df.isnull().sum().sum()
#         return html.Div(f"Number of NaN Values: {nan_count}")


@app.callback(
    Output("download", "data"),
    [Input("download-button", "n_clicks")])


def download_data(n_clicks):
    if n_clicks:
        return dcc.send_data_frame(df.to_csv, "training.csv")



# Define callback to update the plot based on user selection
@app.callback(
    Output('plot-output', 'figure'),
    [Input('column-dropdown', 'value'),
     Input('plot-type-dropdown', 'value')]
)
def update_plot(selected_column, plot_type):
    # Plot selected plot type
    if plot_type == 'kde':
        kde_values = stats.gaussian_kde(df[selected_column])(np.linspace(df[selected_column].min(), df[selected_column].max(), 100))
        fig = go.Figure(data=go.Scatter(x=np.linspace(df[selected_column].min(), df[selected_column].max(), 100), y=kde_values, mode='lines', line=dict(color='red')))
        fig.update_layout(title=f'KDE Plot of {selected_column}', xaxis_title=selected_column, yaxis_title='Density')
    elif plot_type == 'box':
        fig = px.box(df, y=selected_column, title='Box Plot')
    elif plot_type == 'hist':
        fig = px.histogram(df, x=selected_column, title='Histogram Plot')
    elif plot_type == 'qqplot':
        fig = px.scatter(x=df[selected_column],y=stats.norm.ppf((df[selected_column].rank() - 0.5) / len(df)), trendline='ols')
        fig.update_layout(title=f'Scatter Plot with Trendline for {selected_column}', 
                      xaxis_title=selected_column, 
                      yaxis_title='Normal Quantiles')
    elif plot_type == 'regplot':
        fig = px.scatter(x=df[selected_column], y=np.random.normal(size=len(df[selected_column])), title='Regression Plot', trendline='ols')
        fig.update_layout(title=f'Regression Plot of {selected_column}', xaxis_title=selected_column, yaxis_title='Residuals')
    elif plot_type == 'lineplot':
        fig = px.line(df, x=df.datetime, y=selected_column, title='Line Plot')
        fig.update_layout(title=f'Line Plot of {selected_column}', xaxis_title='datetime', yaxis_title=selected_column)
    elif plot_type == 'pie':
        min_val = df[selected_column].min()
        max_val = df[selected_column].max()
        labels = df[selected_column].unique()
        values = [df[selected_column].tolist().count(label) for label in labels]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig.update_layout(title=f'Pie Chart of {selected_column}')

    return fig

# Define callback to perform normality test and display result
@app.callback(
    Output('normality-test-result', 'children'),
    [Input('test-columns-dropdown', 'value'),
     Input('test-type-dropdown', 'value')]
)
def perform_normality_test(selected_columns, test_type):
    results = []
    if selected_columns and test_type:
        for column in selected_columns:
            if test_type == 'shapiro':
                statistic, p_value = stats.shapiro(df_test[column])
                interpretation = ""
                if p_value > 0.01:
                    interpretation = "The data follows a normal distribution."
                else:
                    interpretation = "The data does not follow a normal distribution."
                results.append(html.Div([
                    html.H5(f'Shapiro-Wilk Test Result for {column}:'),
                    html.P(f'Statistic: {statistic}'),
                    html.P(f'P-value: {p_value}'),
                    html.P(f'Interpretation: {interpretation}')
                ]))
            elif test_type == 'dagostino':
                statistic, p_value = stats.normaltest(df_test[column])
                interpretation = ""
                if p_value > 0.01:
                    interpretation = "The data follows a normal distribution."
                else:
                    interpretation = "The data does not follow a normal distribution."
                results.append(html.Div([
                    html.H5(f'D’Agostino’s K^2 Test Result for {column}:'),
                    html.P(f'Statistic: {statistic}'),
                    html.P(f'P-value: {p_value}'),
                    html.P(f'Interpretation: {interpretation}')
                ]))
            elif test_type == 'ks':
                statistic, p_value = stats.kstest(df_test[column], 'norm')
                interpretation = ""
                if p_value > 0.01:
                    interpretation = "The data follows a normal distribution."
                else:
                    interpretation = "The data does not follow a normal distribution."
                results.append(html.Div([
                    html.H5(f'Kolmogorov-Smirnov Test Result for {column}:'),
                    html.P(f'Statistic: {statistic}'),
                    html.P(f'P-value: {p_value}'),
                    html.P(f'Interpretation: {interpretation}')
                ]))
    return results

@app.callback(
    Output('graph', 'figure'),
    Output('loading-output', 'children'),
    Input('county-dropdown', 'value'),
    Input('product-type-dropdown', 'value'),
    Input('is-business-dropdown', 'value'),
    Input('is-consumption-dropdown', 'value'),
    Input('installed-capacity-dropdown', 'value')
)
def update_graph(county, product_type, is_business, is_consumption, installed_capacity):
    if county == "All":
        county = slice(None)
    else:
        county = int(county)
    if product_type == "All":
        product_type = slice(None)
    else:
        product_type = int(product_type)
    if is_business == "All":
        is_business = slice(None)
    else:
        is_business = int(is_business)
    if is_consumption == "All":
        is_consumption = slice(None)
    else:
        is_consumption = int(is_consumption)
    if installed_capacity == "Show":
        installed_capacity = True
    else:
        installed_capacity = False

    train_ = train.xs((county, product_type, is_business, is_consumption), level=["county", "product_type", "is_business", "is_consumption"], axis=1)
    train_.columns = [str(x) for x in train_.columns.values]
    if installed_capacity:
        client_ = client.xs((county, product_type, is_business), level=["county", "product_type", "is_business"], axis=1)
        client_.columns = [str(x) for x in client_.columns.values]
        subfig = make_subplots(specs=[[{"secondary_y": True}]])
        fig1 = px.scatter(train_)
        fig1.update_traces(marker=dict(size=4))
        fig2 = px.line(client_)
        subfig.add_traces(fig1.data + fig2.data)
        subfig.layout.yaxis2.title = "installed_capacity"
        subfig.for_each_trace(lambda trace: trace.update(line=dict(color=trace.marker.color)))
    else:
        subfig = px.scatter(train_)
        #set marker size
        subfig.update_traces(marker=dict(size=4))
    subfig.layout.xaxis.title = "datetime"
    subfig.layout.yaxis.title = "target"

    return subfig, ""



@app.callback(
    Output('pca-plot', 'figure'),
    [Input('num-components-slider', 'value')]
)



def update_pca_plot(num_components_range):
    
    
    selected_columns_pca = ['eic_count', 'installed_capacity',
       'euros_per_mwh', 'temperature_fcast_mean', 'dewpoint_fcast_mean',
       'cloudcover_total_fcast_mean', 'direct_solar_radiation_fcast_mean',
       'surface_solar_radiation_downwards_fcast_mean', 'snowfall_fcast_mean',
       'temperature_hist_mean', 'rain_hist_mean', 'snowfall_hist_mean',
       'shortwave_radiation_hist_mean', 'diffuse_radiation_hist_mean',
       'windspeed_10m_hist_mean_by_county',
       'direct_solar_radiation_hist_mean_by_county']
    
    
    
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[selected_columns_pca])
    
    num_components = list(range(num_components_range[0], num_components_range[1] + 1))
    pca = PCA(n_components=num_components[-1])
    pca.fit(scaled_data)
    explained_variance_ratio = pca.explained_variance_ratio_
    

    fig = go.Figure(
        data=go.Scatter(x=list(range(1, len(explained_variance_ratio) + 1)), y=explained_variance_ratio),
    )
    fig.update_layout(
        title='PCA Explained Variance Ratio',
        xaxis=dict(title='Principal Component'),
        yaxis=dict(title='Explained Variance Ratio'),
    )
    
    return fig
    
# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
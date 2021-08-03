# %%
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
# %%
app = dash.Dash()

app.layout = html.Div(children=[
    html.H1(children='Welcome to Price4Car'), # app name
    html.Br(), 
    html.Br(),
  
    html.Label('Please choose your car make'), # drop down for car make
        dcc.Dropdown(
        id='make',
        options=[
            {'label': 'MERCEDES', 'value': 'T1'},
            {'label': 'AUDI', 'value': 'T1'},
            {'label': 'PEUGEOT', 'value': 'T1'},
            {'label': 'JAGUAR', 'value': 'T1'},
            {'label': 'LEXUS', 'value': 'T1'},
            {'label': 'PORSCHE', 'value': 'T1'},
            {'label': 'LAND ROVER', 'value': 'T1'},
            {'label': 'BENTLEY', 'value': 'T1'},
            {'label': 'MASERATI', 'value': 'T1'},
            {'label': 'INFINITI', 'value': 'T1'},
            {'label': 'ROLLS ROYCE', 'value': 'T1'},
            {'label': 'FERRARI', 'value': 'T1'},
            {'label': 'ASTON MARTIN', 'value': 'T1'},
            {'label': 'MCLAREN', 'value': 'T1'},
            {'label': 'VOLKSWAGEN', 'value': 'T1'},
            {'label': 'VOLVO', 'value': 'T2'},
            {'label': 'MINI', 'value': 'T2'},
            {'label': 'MITSUBISHI', 'value': 'T2'},
            {'label': 'HONDA', 'value': 'T2'},
            {'label': 'JEEP', 'value': 'T2'},
            {'label': 'DS', 'value': 'T2'},
            {'label': 'RENAULT', 'value': 'T2'},
            {'label': 'SUBARU', 'value': 'T2'},
            {'label': 'ALFA ROMEO', 'value': 'T2'},
            {'label': 'SAAB', 'value': 'T2'},
            {'label': 'DODGE', 'value': 'T2'},
            {'label': 'FORD', 'value': 'T3'},
            {'label': 'VAUXHALL', 'value': 'T3'},
            {'label': 'DODGE', 'value': 'T3'},
            {'label': 'NISSAN', 'value': 'T3'},
            {'label': 'TOYOTA', 'value': 'T3'},
            {'label': 'HYUNDAI', 'value': 'T3'},
            {'label': 'SEAT', 'value': 'T3'},
            {'label': 'SMART', 'value': 'T3'},
            {'label': 'CITROEN', 'value': 'T3'},
            {'label': 'ABARTH', 'value': 'T3'},
            {'label': 'KIA', 'value': 'T3'},
            {'label': 'MAZDA', 'value': 'T3'},
            {'label': 'SUZUKI', 'value': 'T3'},
            {'label': 'DACIA', 'value': 'T3'},
            {'label': 'MG', 'value': 'T3'},
            {'label': 'CHEVROLET', 'value': 'T3'},
            {'label': 'SSANGYONG', 'value': 'T3'},
            {'label': 'CHRYSLER', 'value': 'T3'},
            {'label': 'PERODUA', 'value': 'T3'},
            {'label': 'DAIHATSU' , 'value': 'T3'}
    
                   
        ],

    ),
    html.Br(),  # sapce 
    html.Label('Please input your car mileage'), # collect car mileage info
    html.Br(),
    dcc.Input(id='mileage', value='', type='text'),
    html.Br(),
    html.Br(),

    html.Label('Please choose your car fuel type'), # drop down for car fuel type
        dcc.Dropdown(
        id='fuel',
        options=[
            {'label': 'Diesel', 'value': 'Diesel'},
            {'label': 'Petrol', 'value': 'Petrol'},
            {'label': 'Hybrid', 'value': 'Hybrid'},
            {'label': 'Electric', 'value': 'Electric'}
        ],
        
    ),
    html.Br(),  # sapce 

    html.Label('Please choose your car transmission'), # drop down for car transmission
        dcc.Dropdown(
        id='transmission',
        options=[
            {'label': 'Manual', 'value': 'Manual'},
            {'label': 'Automatic', 'value': 'Automatic'},
            {'label': 'Semi-auto', 'value': 'Semi-auto'}
        ],
        
    ),
    html.Br(),

    html.Label('Please choose your car body type'), # drop down for car body type
        dcc.Dropdown(
        id='type',
        options=[
            {'label': 'Estate', 'value': 'Estate'},
            {'label': 'Hatchback', 'value': 'Hatchback'},
            {'label': 'Mpv', 'value': 'Mpv'},
            {'label': 'Coupe', 'value': 'Coupe'},
            {'label': 'Convertible', 'value': 'Convertible'}
        ],
       
    ),
    html.Br(),

    html.Label('Please choose your car engine size'), # drop down for car engine size
        dcc.Dropdown(
        id='engine',
        options=[
            {'label': '<=1L', 'value': '0-1L'},
            {'label': '1L< size <=2L', 'value': '1-2L'},
            {'label': '2L< size <=3L', 'value': '2-3L'},
            {'label': '>3L', 'value': '>3L'}
        ],
        
    ),
    html.Br(),

    html.Label('Please choose your car made year'), # drop down for car year
        dcc.Dropdown(
        id='year',
        options=[
            {'label': '<2010', 'value': '<2010'},
            {'label': '2010-2014', 'value': '2010-2014'},
            {'label': '2015-2019', 'value': '2015-2019'},
            {'label': '>2020', 'value': '>2020'}
        ],
    ),
    html.Br(),
    html.Button(id='submit-button-state', children='Submit', n_clicks=0),
    html.Br(),
    html.Div(id='output-state')

])

@app.callback(Output('output-state', 'children'),
              [Input('submit-button-state', 'n_clicks')],
              [State('make', 'value'),
               State('mileage', 'value'),
               State('fuel', 'value'),
               State('transmission', 'value'),
               State('type', 'value'),
               State('engine', 'value'),
               State('year', 'value')])
def update_output(n_clicks, make, mileage, fuel, transmission, btype, engine, year):
    return 'This is your {} submit, your can information is {}, {}, {}, {}, {}, {}, {}; the estimated price is £'.format(n_clicks, make, mileage, fuel, transmission, btype, engine, year)



# %%
if __name__ == '__main__':
    app.run_server(debug=True)

# %%

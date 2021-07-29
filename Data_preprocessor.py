import pandas as pd
import numpy as np

def data_preprocessor():
    """pre-process data collected from AA"""

    # Import AA data
    df_car = pd.read_csv('Output Test.csv', index_col=0)
    df = df_car[['license_plate', 'price', 'make', 'mileage', 'year', 'fuel_type', 'transmission', 'body_type', 'engine_size']]
    df.set_index('license_plate', inplace=True) #index with licence plates

    # Clean data and re-category categorical features
    df['fuel_type'].replace(dict.fromkeys(['Petrol/electric', 'Plug_in_hybrid', 'Hybrid electric', 'Petrol hybrid', 'Petrol/plugin elec h', 'Petrol / electric hy', 'Petrol phev','Petrol plug-in hybri', 'Diesel hybrid', 'Plug-in hybrid petro','Petrol/electric hybr', 'Diesel/plugin elec h', 'Petrol plugin hybrid'], 'Hybrid'), inplace=True)
    df['fuel_type'].replace(dict.fromkeys([' ', 'N/a'], np.NaN), inplace=True)
    df.dropna(subset=['fuel_type'], inplace=True) # fuel_type col

    df['transmission'].replace(dict.fromkeys(['Semi auto', 'Semiauto', 'Cvt','Semiautomatic', 'Semi automatic', 'Semi-automatic', 'Semi'], 'Semi-auto'), inplace=True)
    df['transmission'].replace(dict.fromkeys(['Not specified', 'N/a', 'Unknown'], np.NaN), inplace=True)
    df.dropna(subset=['transmission'], inplace=True) # transmission col

    df['body_type'].replace(dict.fromkeys(['Other', 'N/a', 'Na', 'Temperature controlled', 'Cars', 'car', 'Truck', 'Limousine'], np.NaN), inplace=True)
    df['body_type'].replace(dict.fromkeys(['Window van', 'Standard roof minibus', 'Van with side windows', 'High roof minibus', 
    'Medium vans', 'Camper van motorhome', 'Large vans', 'Krew cab', 'Crew bus', 'Crew cab', 'Double cab van', 'Dropside', 'Tipper',
    'Chassis cab', 'Double cab tipper', 'Hook loader', 'High volume/high roof van', 'Luton', 'Curtainside', 'Swb panel van', 'Van swb', 
    'Van - swb', 'Panel van lwb', 'Mwb panel van', 'Lwb panel van', 'Van lwb', 'Transit custom', 'Double cab pick-up', 'Pick up', 
    'Dropside pickups', 'Crewcab pickup', 'Medium roof van','Medium vans',  'All terrain', 'Minibus', 'Van', 'Pick-up', 'Medium van', 'Pickup',
    'Combi van', 'Campervan', 'Station wagon', 'Panel van'], 'van'), inplace=True)
    df['body_type'].replace(dict.fromkeys(['Sports tourer' ], 'Estate'), inplace=True)
    df['body_type'].replace(dict.fromkeys(['Compact saloon', '4 door saloon'], 'Saloon'), inplace=True)
    df['body_type'].replace(dict.fromkeys(['People carrier', 'Commercial', 'Car derived van'], 'Mpv'), inplace=True)
    df['body_type'].replace(dict.fromkeys(['Cabriolet', 'Coupe-convertible', 'Convertibles', 'Cabriolet/roadster', 'Sports car', 'Sports', 'Roadster'], 'Convertible'), inplace=True)
    df['body_type'].replace(dict.fromkeys(['Off-roader', 'Light 4x4 utility', 'Four wheel drive', 'Crossover', '4x4'], 'Suv'), inplace=True)
    df['body_type'].replace(dict.fromkeys(['5 door hatchback', 'Hatch', '3 door hatchback'], 'Hatchback'), inplace=True)
    df.dropna(subset=['body_type'], inplace=True)
    index_btype = df[df['body_type'].str.contains('van')].index
    df.drop(index_btype, inplace = True)  # body_type col

    df['make'] = df['make'].apply(lambda x: x.upper())
    df['make'].replace(dict.fromkeys(['MERCEDES-BENZ', 'MERCEDES BENZ'], 'MERCEDES'), inplace=True)
    df['make'].replace('DS AUTOMOBILES', 'DS', inplace=True)
    df['make'].replace('MG MOTOR UK', 'MG', inplace=True)  # make col

    df['car_tier'] = df['make']
    df['car_tier'].replace(dict.fromkeys(['MERCEDES', 'AUDI', 'PEUGEOT', 'LAND ROVER', 'JAGUAR', 'LEXUS', 'PORSCHE', 'BENTLEY', 'MASERATI', 'INFINITI', 
    'ROLLS ROYCE', 'FERRARI', 'ASTON MARTIN', 'MCLAREN', 'BMW'], 'T1'), inplace=True)
    df['car_tier'].replace(dict.fromkeys(['VOLKSWAGEN', 'VOLVO', 'MINI', 'MITSUBISHI', 'HONDA', 'JEEP', 'DS', 'RENAULT', 'SUBARU', 'ALFA ROMEO', 'SAAB', 'DODGE'], 'T2'), inplace=True)
    df['car_tier'].replace(dict.fromkeys(['FORD', 'VAUXHALL', 'NISSAN', 'SKODA', 'TOYOTA', 'HYUNDAI', 'SEAT', 'SMART', 'CITROEN', 
    'FIAT', 'ABARTH', 'KIA', 'MAZDA', 'SUZUKI', 'DACIA', 'MG', 'ABARTH', 'CHEVROLET', 'SSANGYONG', 'CHRYSLER', 'PERODUA', 'DAIHATSU'], 'T3'), inplace=True) #categorize make into tires


    df.dropna(subset=['engine_size'], inplace=True) # engine_size col

    df['year'].replace(19, 2019, inplace=True)
    df['year'].replace(21, 2021, inplace=True)
    df['year'].replace(70, 1970, inplace=True)
    df['year'].replace(69, 1969, inplace=True)
    df['year'].replace(18, 2018, inplace=True)
    df['year'].replace(68, 1968, inplace=True)
    df['year'].replace(17, 2017, inplace=True)
    df['year'].replace(12, 2012, inplace=True)
    df['year'].replace(65, 1965, inplace=True)
    df['year'].replace(66, 1966, inplace=True)
    df['year'].replace(67, 1967, inplace=True)
    df['year'].replace(16, 2016, inplace=True)
    df['year'].replace(64, 1964, inplace=True)
    df['year'].replace(20, 2020, inplace=True)
    df['year'].replace(15, 2015, inplace=True)
    df['year'].replace(14, 2014, inplace=True)
    df['year'].replace(10, 2010, inplace=True)
    df['year'].replace(13, 2013, inplace=True)
    df['year'].replace(62, 1962, inplace=True)
    df['year'].replace(0, np.nan, inplace=True)
    df.dropna(subset=['year'], inplace=True) # year col

    df['year_band'] = df['year']
    df['year_band'].replace(dict.fromkeys([2021.0, 2020.0], '>2020'), inplace=True)
    df['year_band'].replace(dict.fromkeys([2015.0, 2016.0, 2017.0, 2018.0, 2019.0], '2015-2019'), inplace=True)
    df['year_band'].replace(dict.fromkeys([2011.0, 2010.0, 2012.0, 2013.0, 2014.0], '2010-2014'), inplace=True)
    df['year_band'].replace(dict.fromkeys([2004., 2007., 2006., 2008., 2005., 2009., 2003., 2000., 2002., 1972., 1970., 1969., 1968., 1965., 1966.,
    1967., 1964., 1962., 2001., 1998., 1999., 1996., 1974., 1997.], '<2010'), inplace=True) #catigorize year into year bands

    df['mileage'].astype(int) # mileage col

    # Saves pre-processed data to a new csv file
    df.to_csv('Pre-processed Data.csv', index=True)


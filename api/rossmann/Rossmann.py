import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

class Rossmann(object):
    def __init__(self):
        # self.home_path = 
        self.competition_distance_scaler   = pickle.load(open('../parameters/competition_distance_scaler.pkl', 'rb'))
        self.promo_time_week_scaler        = pickle.load(open('../parameters/promo_time_week_scaler.pkl', 'rb'))
        self.competition_time_month_scaler = pickle.load(open('../parameters/competition_time_month_scaler.pkl', 'rb'))
        self.week_of_year_scaler           = pickle.load(open('../parameters/week_of_year_scaler.pkl', 'rb'))
        self.promo2_since_week_scaler      = pickle.load(open('../parameters/promo2_since_week_scaler.pkl', 'rb'))
        self.store_type_encoder            = pickle.load(open('../parameters/store_type_encoder.pkl', 'rb'))
        self.assortment_encoder            = pickle.load(open('../parameters/assortment_encoder.pkl', 'rb'))

    def rename_columns(self, dataframe):
        df = dataframe.copy()
        title = lambda x: inflection.titleize(x)
        snakecase = lambda x: inflection.underscore(x)
        spaces = lambda x: x.replace(" ", "")
        cols_old = list(df.columns)
        cols_old = list(map(title, cols_old))
        cols_old = list(map(spaces, cols_old))
        cols_new = list(map(snakecase, cols_old))
        df.columns = cols_new
        return df

    def data_cleaning(self, df1):

        ## 1.1 Rename columns

        # Using the function we created in section 0.1
        df1 = self.rename_columns(df1)

        ## 1.3 Data types

        #Changing the type to date
        df1['date'] = pd.to_datetime(df1['date'])

        ## 1.5 Fillout NA 

        # Dealing the columns with NaNs

        # competition_distance - imputting 200000 distance for nan values
        df1['competition_distance'] = df1['competition_distance'].apply(lambda x: 200000 if np.isnan(x) else x)

        # competition_open_since_month - imputting the month of register for competition open
        df1['competition_open_since_month'] = df1[['date', 'competition_open_since_month']].apply(lambda x: x['date'].month if np.isnan(x['competition_open_since_month']) else x['competition_open_since_month'], axis=1)

        # competition_open_since_year - imptting the year of register for competition open  
        df1['competition_open_since_year'] = df1[['date', 'competition_open_since_year']].apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else x['competition_open_since_year'], axis=1)


        # promo2_since_week - imputting the week of register for promo since
        df1['promo2_since_week'] = df1[['date', 'promo2_since_week']].apply(lambda x: x['date'].week if np.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis=1)

        # promo2_since_year - imputting the year of register for promo since
        df1['promo2_since_year'] = df1[['date', 'promo2_since_year']].apply(lambda x: x['date'].year if np.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis=1)


        # promo_interval - imputting zero for promo interval
        df1['promo_interval'] = df1['promo_interval'].fillna(0)

        # Creating month map, then creating a column with the name of the month to check if there is a promo during the time or not
        month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}

        df1['month_map'] = df1['date'].dt.month.map(month_map)

        df1['is_promo'] = df1[['promo_interval', 'month_map']].apply(lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)


        ## 1.6 Change Types

        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype('int64')
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype('int64')

        df1['promo2_since_week'] = df1['promo2_since_week'].astype('int64')
        df1['promo2_since_year'] = df1['promo2_since_year'].astype('int64')
    
        return df1

    def feature_engineering(self, df2):
        
        #year
        df2['year'] = df2['date'].dt.year

        #month
        df2['month'] = df2['date'].dt.month

        #day
        df2['day'] = df2['date'].dt.day

        #week of year
        df2['week_of_year'] = df2['date'].dt.isocalendar().week.astype(int)

        #year week
        df2['year_week'] = df2['date'].dt.strftime('%Y-%W')

        #competition since
        df2['competition_since'] = df2.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'], month=x['competition_open_since_month'], day=1), axis=1)
        df2['competition_time_month'] = ((df2['date'] - df2['competition_since']) / 30).apply(lambda x: x.days).astype(int)

        #promo since
        df2['promo_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str)
        df2['promo_since'] = df2['promo_since'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7) )

        # promo time week = date - promo since/7 (pega dia)
        df2['promo_time_week'] = ((df2['date'] - df2['promo_since'])/7).apply(lambda x: x.days).astype(int)

        #assortment
        df2['assortment'] = df2['assortment'].apply(lambda x: 'basic' if x=='a' else 'extra' if x=='b' else 'extended')

        #state holiday
        df2['state_holiday'] = df2['state_holiday'].apply(lambda x: 'public_holiday' if x=='a' else 'easter_holiday' if x=='b' else 'christmas' if x=='c' else 'regular_day')

        # 3.0 Feature filtering

        ## 3.1 Filtragem das linhas

        # Selecting only the days we have sales and the days the stores are open
        #df2 = df2[(df2['sales'] > 0) & (df2['open'] != 0)]

        ## 3.2 Seleção das colunas

        # Dropping columns customers (we can't know customers will be in the stores in following days), open (we already filter for only open stores), promo interval and month map only used to create is_promo, so we don't need them
        cols_drop = ['open', 'promo_interval', 'month_map']
        df2 = df2.drop(cols_drop, axis=1)
    
        return df2

    def data_preparation(self, df5):
        
        ## 5.2 Rescaling

        #RobustScaler
        #competition_distance
        df5['competition_distance'] = self.competition_distance_scaler.fit_transform(df5[['competition_distance']].values)

        #promo_time_week
        df5['promo_time_week'] = self.promo_time_week_scaler.fit_transform(df5[['promo_time_week']].values)
        

        #competition_time_month
        df5['competition_time_month'] = self.competition_time_month_scaler.fit_transform(df5[['competition_time_month']].values)
        
        #MinMaxScaler
        #week_of_year
        df5['week_of_year'] = self.week_of_year_scaler.fit_transform(df5[['week_of_year']].values)
        

        #promo2_since_week
        df5['promo2_since_week'] = self.promo2_since_week_scaler.fit_transform(df5[['promo2_since_week']].values)
        
        
        ## 5.3 Transformation
        ### 5.3.1 Encoding
        
        #state_holiday - OneHotEnconder
        df5 = pd.get_dummies(df5, prefix=['state_holiday'], columns=['state_holiday'])

        #store_type
        df5['store_type'] = self.store_type_encoder.fit_transform(df5['store_type'])
        

        #assortment
        df5['assortment'] = self.assortment_encoder.fit_transform(df5[['assortment']].values)

        ### 5.3.3 Cyclical features encoding

        #month
        df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x * (2 * np.pi/12)))
        df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x * (2 * np.pi/12)))

        #day
        df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x * (2 * np.pi/30)))
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x * (2 * np.pi/30)))


        #week of year
        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x: np.sin(x * (2 * np.pi/52)))
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos(x * (2 * np.pi/52)))

        #day of week
        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x * (2 * np.pi/7)))
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x * (2 * np.pi/7)))
        
        cols_selected = ['store', 'promo','school_holiday', 'store_type', 'assortment', 'competition_distance', 'promo2', 'promo2_since_week', 'promo2_since_year', 'is_promo', 'year', 'promo_time_week', 'day_sin', 
                         'day_cos', 'week_of_year_sin', 'week_of_year_cos', 'day_of_week_sin', 'day_of_week_cos']

        return df5[cols_selected]
    
    def get_prediction(self, model, original_data, test_data):
        # prediction
        pred = model.predict(test_data)
        
        # join pred into the original data
        #original_data['prediction'] = np.expm1(pred)
        test_data['prediction'] = np.expm1(pred)
        
        return test_data.to_json(orient='records', date_format='iso')
    
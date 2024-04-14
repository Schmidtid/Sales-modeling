"""Help function"""
import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostRegressor

pd.options.mode.copy_on_write = True

class model:
    """Model to forecast sales

    Args:
        data (pd.DataFrame): sales data for forecast from excel or .csv file
    
    """
    def __init__(self,data) -> None:
        cat = CatBoostRegressor()
        self.cat = cat.load_model('model_91')
        self.ind = data['Продажи, рубли'].isna().values
        self.data = data
        self.value = self.predict(data[~self.ind])

    def predict(self, data: pd.DataFrame,
                param: dict = None):
        """Evalute forecast
        
        Args:
            data (pd.DataFrame): sales data for forecast
            param (dict | None): columns to change
            """
        if param:
            for i, value in param.items():
                data[i] += value*100
        return self.cat.predict(data)

    def plot_graph(self,
                   tv=None, radio=None,
                   concurent=None, digital=None,
                   trp= None):
        """Plot interactive plot
        
        Args:
            tv (float): changes for investment in TV advertising
            radio (float): changes investment in radio advertising
            concurence (float): changes Estimated investments of competitors
            digital (float): changes for investment in digital advertising"""
        names = {'Затраты на ТВ':tv,
                'Затраты Радио':radio,
                'Конкуренты, итого затрат':concurent,
                'Затраты на диджитал':digital,
                '(тотал) ТВ, trp':trp,}
        fig, ax = plt.subplots(1, figsize=(16,8), dpi=100,facecolor='w')
        ax.plot(self.data['Начало нед'], self.data['Продажи, рубли'],c='b', label='Known data')
        pred = self.predict(self.data[self.ind], names)
        ax.plot(self.data['Начало нед'].values[self.ind], pred, c='g', label='Predict data')
        ax.legend(['Known data','Predict data'])
        plt.show()

    def plot_data(self):
        """Plot statistic data"""
        plot = self.cat.get_feature_importance(prettified=True)
        plot.index = plot['Feature Id']
        plot = plot.drop(columns='Feature Id')
        plot = plot.T
        (plot[['Конкуренты, итого затрат',
               'Затраты на диджитал','Затраты на ТВ',
               'Затраты Радио']]/plot.T['Importances'].sum()*100
        ).plot(kind='barh',xlabel='Importance in forecasting',figsize=(12,8))
        plt.show()

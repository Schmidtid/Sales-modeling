"""Help function"""
import plotly.graph_objects as go
import pandas as pd
from catboost import CatBoostRegressor


class Model:
    """Model to forecast sales

    Args:
        data (pd.DataFrame): sales data for forecast from excel or .csv file
    
    """
    def __init__(self,data) -> None:
        cat = CatBoostRegressor()
        self.cat = cat.load_model('models/model_91')
        self.ind = data['Продажи, рубли'].isna().values
        self.data = data
        self.value = self.predict(data[self.ind])

    def predict(self, data: pd.DataFrame,
                param: dict = None):
        """Evalute forecast
        
        Args:
            data (pd.DataFrame): sales data for forecast
            param (dict | None): columns to change
            """
        if param:
            for i, value in param.items():
                data[i] += value*1000
        return self.cat.predict(data)
    
    def plt_pred(self):
        """Plot initial predictions"""
        fig = go.Figure(layout=go.Layout(height=600, width=1000))

        fig.add_trace(go.Scatter(
            x=self.data['Начало нед'],
            y=self.data['Продажи, рубли'], 
            mode='lines',
            name='Known data',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=self.data['Начало нед'].values[self.ind], 
            y=self.value,
            mode='lines', 
            name='Predicted data',
            line=dict(color='green')
        ))
        fig.update_layout(
            title='Sales Forecast',
            xaxis_title='Start of Week',
            yaxis_title='Sales, Rubles',
            legend_title='Legend',
        )
        return fig

    def plot_graph(self,
                   tv:float = 0, radio:float = 0,
                   concurrent:float = 0, digital:float = 0,
                   trp:float = 0):
        """Plot forecasts taking into account changes
        
        Args:
            tv (float): changes for investment in TV advertising
            radio (float): changes investment in radio advertising
            concurence (float): changes Estimated investments of competitors
            digital (float): changes for investment in digital advertising"""
        names = {'Затраты на ТВ':tv,
                'Затраты Радио':radio,
                'Конкуренты, итого затрат':concurrent,
                'Затраты на диджитал':digital,
                '(тотал) ТВ, trp':trp,}
        fig = go.Figure(layout=go.Layout(height=600, width=1200))

        pred = self.predict(self.data[self.ind], names)
        fig.add_trace(go.Scatter(
            x=self.data['Начало нед'].values[self.ind], 
            y=pred,
            mode='lines', 
            name='Predicted data',
            line=dict(color='green')
        ))
        fig.update_layout(
            title='Sales Forecast',
            xaxis_title='Start of Week',
            yaxis_title='Sales, Rubles',
        )
        return fig

    def plot_data(self):
        """Plot statistic data"""
        plot = self.cat.get_feature_importance(prettified=True)
        plot = plot[plot['Importances']>1]
        plot.loc[len(plot)+1,['Feature Id','Importances']] = ['Other',100-plot['Importances'].sum()]
        fig = go.Figure(data=[go.Pie(labels=plot['Feature Id'], values=plot['Importances'])],
                        layout=go.Layout(height=600, width=800))
        return fig

import io
import pandas as pd
import streamlit as st
from utils import Model

def parse_contents(contents, filename: str):
    try:
        if filename.endswith('csv'):
            df = pd.read_csv(io.StringIO(contents))
        elif 'xls' in filename.split('.')[-1]:
            load = False
            skiprows = 0
            while not load:
                df = pd.read_excel(io.BytesIO(contents),
                                   skiprows=skiprows,engine='openpyxl')
                df = pd.read_excel(io.BytesIO(contents),
                                   skiprows=skiprows,engine='openpyxl')
                if df.columns.str.contains('Продажи').any() == True:
                    load = True
                skiprows += 1
        else:
            return None
    except Exception as e:
        print(e)
        return None
    return df

def main():
    st.title("Sales Forecast Dashboard")

    uploaded_file = st.file_uploader("Upload a .csv or .xlsx file", type=['csv', 'xlsx'],accept_multiple_files=False)

    if uploaded_file is not None:
        filename = uploaded_file.name
        content = uploaded_file.getvalue()
        data = parse_contents(content, filename)
        
        if data is not None:
            st.success("File uploaded successfully.")
            model_instance = Model(data)
            st.plotly_chart(model_instance.plt_pred(),use_container_width=False)
            if st.toggle('Show dependencies'):
                st.plotly_chart(model_instance.plot_data(),use_container_width=False)
            tv = st.slider('Investment in TV Advertising', min_value=-1000000.0, max_value=1000000.0, step=1.0,value=0.0)
            digital = st.slider('Investment in Digital Advertising', min_value=-1000000.0, max_value=1000000.0, step=1.0,value=0.0)
            trp = st.slider('Estimated Coverage (TRP)', min_value=-1000000.0, max_value=1000000.0, step=1.0,value=0.0)
            radio = st.slider('Investment in Radio Advertising', min_value=-1000000.0, max_value=1000000.0, step=1.0,value=0.0)
            concurrent = st.slider('Estimated Investments of Competitors', min_value=-10000000.0, max_value=10000000.0, step=1.0,value=0.0)
            
            fig = model_instance.plot_graph(tv, radio, concurrent, digital, trp)
            st.plotly_chart(fig,use_container_width=False)
        else:
            st.error("Failed to parse the uploaded file.")

if __name__ == '__main__':
    main()

import streamlit as st
import log_processor
import os
import altair as alt

st.header('IMC Trading logs plot')

file_paths = ["TutorialRound/logs","Round1/logs","Round2/logs","Round3/logs","Round4/logs"]
option_path = st.selectbox(
     'Round:',
     file_paths)
option_log = st.selectbox(
    'Log:',
    os.listdir(f"../{option_path}")
)
file_path = f"../{option_path}/{option_log}"

# Load the data
log_processor = log_processor.LogProcessor(file_path)
log_processor.read_log()

prod = st.selectbox(
    'Plot:',
    log_processor.activites_log['product'].unique()
)
df = log_processor.activites_log[log_processor.activites_log['product']==prod]
trades_df = log_processor.trade_history[log_processor.trade_history['symbol']==prod]
trades_df = trades_df[(trades_df['buyer'] == 'SUBMISSION') | (trades_df['seller'] == 'SUBMISSION')]
lambd = lambda buy,sell: "buy" if buy == "SUBMISSION" else "sell"
trades_df['side'] = trades_df.apply(lambda row: "buy" if row['buyer'] == "SUBMISSION" else "sell", axis=1)

# Plots
df_melt = df[['timestamp','bid_price_1', 'ask_price_1', 'mid_price']].melt(id_vars=['timestamp'], var_name='Series', value_name='y')
color_scale = alt.Scale(
    domain=['bid_price_1', 'ask_price_1', 'mid_price'],
    range=['red', 'green', 'black']
)

chart = alt.Chart(df_melt).mark_line().encode(
    x=alt.X('timestamp:Q', axis=alt.Axis(title="Timestamp")),
    y=alt.Y('y:Q',scale=alt.Scale(domain=[df['bid_price_1'].min()-10,df['ask_price_1'].max()+10]), axis=alt.Axis(title="Bid Ask Mid price")),
    color=alt.Color('Series:N',scale = color_scale)
).properties( title=f'Price',)

trade_points = alt.Chart(trades_df).mark_circle(size=100).encode(
    x='timestamp:Q',
    y='price:Q',
    tooltip=['time:T', 'price:Q', 'side:N', 'quantity:Q']
)

chart = chart + trade_points

df['loss'] = df['profit_and_loss'].clip(upper=0)
df['profit'] = df['profit_and_loss'].clip(lower=0)

line = alt.Chart(df).mark_line(color='black').encode(
    x = alt.X('timestamp:Q', axis=alt.Axis(title="Timestamp")),
    y = alt.Y('profit_and_loss:Q', axis=alt.Axis(title="Profit and loss"))
)

area_positive = alt.Chart(df).mark_area(color='green',opacity=0.3).encode(
    x='timestamp:Q',
    y='profit:Q'
)
area_negative = alt.Chart(df).mark_area(color='red',opacity=0.3).encode(
    x='timestamp:Q',
    y='loss:Q'
)

area_chart = area_negative + area_positive + line
area_chart.properties(
    title=f'Profit and Loss',
)
chart = alt.vconcat(chart, area_chart).configure_axisX(
    labelAngle=0
).interactive()

# Display in Streamlit
st.altair_chart(chart, use_container_width=True)

# Display dataframe
st.write("Activities log")
st.write(df)
st.write("Sandbox logs")
st.write(log_processor.sandbox_logs)
st.write("Bot trade history logs")
st.write(trades_df)
st.write("All Trade history logs")
st.write(log_processor.trade_history)



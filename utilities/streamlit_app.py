import streamlit as st
import log_processor
import os
import altair as alt
import pandas as pd
import time
import numpy as np

st.set_page_config(layout="wide")

st.header('IMC Trading logs plot')

#-------------------------------------------------Setup -------------------------------------------------------
file_paths = ["TutorialRound/logs","Round1/logs","Round2/logs","Round3/logs","Round4/logs","backtests"]
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
    log_processor.activities_log['product'].unique()
)
df = log_processor.activities_log[log_processor.activities_log['product']==prod]
total_pnl = sum([log_processor.activities_log[log_processor.activities_log['product']==product]['profit_and_loss'].iloc[-1] for product in log_processor.activities_log['product'].unique()])

trades_df = log_processor.trade_history[log_processor.trade_history['symbol']==prod]
trades_df = trades_df[(trades_df['buyer'] == 'SUBMISSION') | (trades_df['seller'] == 'SUBMISSION')]
trades_df['side'] = trades_df.apply(lambda row: "buy" if row['buyer'] == "SUBMISSION" else "sell", axis=1)

positions_df = log_processor.trade_history[log_processor.trade_history['symbol']==prod]
positions_df['side'] = positions_df.apply(lambda row: "buy" if row['buyer'] == "SUBMISSION" else ("sell" if row['seller'] == "SUBMISSION" else "hold"), axis=1)
positions_df['signed_quantity'] = positions_df.apply(lambda row: row['quantity'] if row['side'] == 'buy' else (-row['quantity'] if row['side'] == 'sell' else 0), axis=1)
positions_df['cumulative_pos'] = positions_df['signed_quantity'].cumsum()

# Setting up session states
if "vline_x" not in st.session_state:
    st.session_state["vline_x"] = df['timestamp'].iloc[0] 
if "animate" not in st.session_state:
    st.session_state["animate"] = False

#------------------------------------------------- Orderbook Display -------------------------------------------------------
vline_x = st.session_state["vline_x"]
# Display in Streamlit
col1, col2 = st.columns([2, 1])

with col2:    
    idx = st.session_state['vline_x']//100
    timestamp_selector = st.selectbox(
     'Timestamp:',
     [i for i in range(int(df['timestamp'].iloc[0]),int(df['timestamp'].iloc[-1]+1),100)],
     index = int(idx))
    
    if not st.session_state['animate']:
        st.session_state["vline_x"] = timestamp_selector
    
    filtered_df = df[df["timestamp"] == st.session_state['vline_x']]
    order_book = pd.DataFrame({
        "Bid Volume": filtered_df[["bid_volume_1", "bid_volume_2", "bid_volume_3"]].values.flatten(),
        "Bid Price": filtered_df[["bid_price_1", "bid_price_2", "bid_price_3"]].values.flatten(),
        "Ask Price": filtered_df[["ask_price_1", "ask_price_2", "ask_price_3"]].values.flatten(),
        "Ask Volume": filtered_df[["ask_volume_1", "ask_volume_2", "ask_volume_3"]].values.flatten(),
    })

    max_ask = order_book['Ask Price'].max()
    min_ask = order_book['Ask Price'].min()
    max_bid = order_book['Bid Price'].max()
    min_bid = order_book['Bid Price'].min()
    def highlight_rows(row):
        if min_ask <= row["Price"] <= max_ask:  # Between lowest and highest ask
            return ["background-color: lightgreen"] * len(row)
        elif min_bid <= row["Price"] <= max_bid:  # Between highest and lowest bid
            return ["background-color: lightcoral"] * len(row)
        return ["background-color: lightyellow"] * len(row)  # All other rows
    
    price_vals = [i for i in range(int(max_ask),int(min_bid)-1,-1)]
    price_ladder = pd.DataFrame({
        "Price": price_vals,
    })
    bid_ladder = order_book.groupby("Bid Price")["Bid Volume"].sum().reset_index()
    price_ladder = price_ladder.merge(bid_ladder, how="left", left_on="Price", right_on="Bid Price").drop(columns=["Bid Price"])
    price_ladder["Bid Volume"].fillna(0, inplace=True)
    bid_ladder = order_book.groupby("Ask Price")["Ask Volume"].sum().reset_index()
    price_ladder = price_ladder.merge(bid_ladder, how="left", left_on="Price", right_on="Ask Price").drop(columns=["Ask Price"])
    price_ladder["Ask Volume"].fillna(0, inplace=True)
    price_ladder = price_ladder[['Bid Volume','Price','Ask Volume']]
    price_ladder = price_ladder.style.apply(highlight_rows, axis=1)
    
    st.table(price_ladder)
    
    if st.button("Animate"):
        st.session_state["animate"] = True
        st.session_state["vline_x"] = timestamp_selector
    if st.button("Stop Animation"):
        st.session_state['animate'] = False
    
    if st.button("Next Timestamp"):
        st.session_state["vline_x"] += 100
    
    if st.button("Prev Timestamp"):
        st.session_state["vline_x"] -= 100

#-------------------------------------------------Plots -------------------------------------------------------

df_melt = df[['timestamp','bid_price_1', 'ask_price_1', 'mid_price']].melt(id_vars=['timestamp'], var_name='Series', value_name='y')
color_scale = alt.Scale(
    domain=['bid_price_1', 'ask_price_1', 'mid_price'],
    range=['red', 'green', 'black']
)

chart = alt.Chart(df_melt).mark_line().encode(
    x=alt.X('timestamp:Q', axis=alt.Axis(title="Timestamp")),
    y=alt.Y('y:Q',scale=alt.Scale(domain=[df['bid_price_1'].min()-10,df['ask_price_1'].max()+10]), axis=alt.Axis(title="Bid Ask Mid price")),
    color=alt.Color('Series:N',scale = color_scale)
).properties( title=f'Price ({round(100*len(trades_df)/len(log_processor.trade_history[log_processor.trade_history['symbol']==prod]))}% timestamps traded)',
             height = 300)

trade_points = alt.Chart(trades_df).mark_circle(size=100).encode(
    x='timestamp:Q',
    y='price:Q',
    tooltip=['timestamp:Q', 'price:Q', 'side:N', 'quantity:Q']
)

# Vertical Line
vline_x = st.session_state["vline_x"]
vline = alt.Chart(pd.DataFrame({"x": [vline_x]})).mark_rule(color="black", strokeWidth=2).encode(x="x:Q")
chart = chart + trade_points + vline

vols = df[['timestamp','bid_volume_1', 'ask_volume_1']]
vols['ask_volume_1'] *= -1
vols['total_bid'] = df[['bid_volume_1','bid_volume_2','bid_volume_3']].sum(axis=1)
vols['total_ask'] = -df[['ask_volume_1','ask_volume_2','ask_volume_3']].sum(axis=1)

df_melt = vols.melt(id_vars=['timestamp'],var_name='Series',value_name='y')
color_scale = alt.Scale(
    domain = ['bid_volume_1','ask_volume_1','total_bid','total_ask'],
    range = ['red','green','darkred', 'darkgreen']
)

volume_chart = alt.Chart(df_melt).mark_line().encode(
    x=alt.X('timestamp:Q', axis=alt.Axis(title="Timestamp")),
    y = alt.Y('y:Q',scale=alt.Scale(domain=[vols['ask_volume_1'].min()-20,vols['bid_volume_1'].max()+20]), axis=alt.Axis(title='Bid vol Ask vol')),
    color = alt.Color('Series:N', scale = color_scale),
).properties(title='Volume',
             height = 200)

vline_x = st.session_state["vline_x"]
vline = alt.Chart(pd.DataFrame({"x": [vline_x]})).mark_rule(color="black", strokeWidth=2).encode(x="x:Q")

volume_chart = volume_chart + vline

df['spreads'] = df['ask_price_1'] - df['bid_price_1']
df['max_spreads'] = df.apply(lambda row : max([row[f'ask_price_{i}'] for i in range(1,4)]) - min([row[f'bid_price_{i}'] for i in range(1,4)]),axis=1)
# df['highest_ask_vol'] = df.apply(lambda row : 1+np.argmax([row.fillna(0)[f'ask_volume_{i}'] for i in range(1,4)]),axis=1)
# df['highest_bid_vol'] = df.apply(lambda row : 1+np.argmax([row.fillna(0)[f'bid_volume_{i}'] for i in range(1,4)]),axis=1)
# df['highest_vol_spreads'] = df.apply(lambda row : row[f"ask_price_{row['highest_ask_vol']}"]-row[f"bid_price_{row['highest_bid_vol']}"], axis=1)

spreads = df[['timestamp','spreads','max_spreads']]
df_melt = spreads.melt(id_vars=['timestamp'],var_name='Series', value_name='y')
color_scale = alt.Scale(
    domain = ['spreads','max_spreads','highest_vol_spreads'],
    range = ['lightblue','darkblue','blue']
)

spread_chart = alt.Chart(df_melt).mark_line().encode(
    x=alt.X('timestamp:Q', axis=alt.Axis(title="Timestamp")),
    y = alt.Y('y:Q',scale=alt.Scale(domain=[0,df['spreads'].max()+1]), axis=alt.Axis(title='Spread')),
    color= alt.Color('Series:N', scale = color_scale),
).properties(
    title='Spreads',
    height = 200
)

spread_chart = spread_chart + vline


df['loss'] = df['profit_and_loss'].clip(upper=0)
df['profit'] = df['profit_and_loss'].clip(lower=0)

line = alt.Chart(df).mark_line(color='black').encode(
    x = alt.X('timestamp:Q', axis=alt.Axis(title="Timestamp")),
    y = alt.Y('profit_and_loss:Q', scale=alt.Scale(domain=[df['loss'].min()-100,df['profit'].max()+100]), axis=alt.Axis(title="Profit and loss"))
).properties(
    height=300,
    title=f'Profit and Loss, Final:{df['profit_and_loss'].iloc[-1]}, Total PnL for all:{total_pnl} \nTimestamp Value:{df['profit_and_loss'].iloc[timestamp_selector//100]}',
)


area_positive = alt.Chart(df).mark_area(color='green',opacity=0.3).encode(
    x='timestamp:Q',
    y='profit:Q'
)
area_negative = alt.Chart(df).mark_area(color='red',opacity=0.3).encode(
    x='timestamp:Q',
    y='loss:Q'
)

area_chart = area_negative + area_positive + line + vline

positions_df['neg_pos'] = positions_df['cumulative_pos'].clip(upper=0)
positions_df['pos_pos'] = positions_df['cumulative_pos'].clip(lower=0)
positions_df_melt = positions_df[['timestamp','cumulative_pos']].melt(id_vars=['timestamp'],var_name='Series', value_name='y')

positions_chart = alt.Chart(positions_df_melt).mark_line(color='black').encode(
    x = alt.X('timestamp:Q', axis=alt.Axis(title="Timestamp")),
    y = alt.Y('y:Q', scale=alt.Scale(domain=[-60,60]), axis=alt.Axis(title='Cumulative position'))
).properties(
    height=300,
    title = f"Cumulative Position"
)

positions_positive = alt.Chart(positions_df).mark_area(color='green',opacity=0.3).encode(
    x='timestamp:Q',
    y='pos_pos:Q'
)

positions_negative = alt.Chart(positions_df).mark_area(color='red',opacity=0.3).encode(
    x='timestamp:Q',
    y='neg_pos:Q'
)

positions_chart = positions_chart + positions_positive + positions_negative + vline

with col1:
    st.altair_chart(chart, use_container_width=True)
    st.altair_chart(volume_chart, use_container_width=True)
    st.altair_chart(spread_chart, use_container_width=True)
    st.altair_chart(positions_chart, use_container_width=True)
    st.altair_chart(area_chart, use_container_width=True)
    
# Display dataframe
st.write("Activities log")
st.write(df)
st.write("Sandbox logs")
st.write(log_processor.sandbox_logs)
st.write("Bot trade history logs")
st.write(trades_df)
st.write(f"{prod} trade history logs")
st.write(log_processor.trade_history[log_processor.trade_history['symbol']==prod])
st.write("All Trade history logs")
st.write(log_processor.trade_history)


if st.session_state["animate"]:
    if st.session_state['vline_x'] < df['timestamp'].iloc[-1]:
        st.session_state["vline_x"] += 100
        time.sleep(0.5)
        st.rerun()
    else:
        st.session_state.animate = False
import streamlit as st
from PIL import Image
import plotly.express as px
import pickle
import pandas as pd

# Unpickling the trained model
xgbc_model = pickle.load(open("./PSL-Win-XGBC-model.pkl", "rb"))

# Set page configuration for a more professional look
st.set_page_config(page_title="PSL-8 Win Predictor",
   page_icon="🏏",
   layout="wide",
   initial_sidebar_state="expanded",)

# Custom CSS for styling
st.markdown("""
<style>
    .big-font {
        font-size:50px !important;  # Increased font size for main title
        color:Green;
        text-align:center;
    }
    .result-text {
        text-align:center;
        font-size:20px;  # Added font size for result text
    }
</style>
""", unsafe_allow_html=True)

# Main title with enhanced styling
st.markdown('<p class="big-font">Pakistan Super League-8 Win Predictor</p>', unsafe_allow_html=True)

# Displaying the image with full column width
img = Image.open("./PSL-8.jpg")
st.image(img, use_column_width=True)

# Sidebar improvements
st.sidebar.header("Current Match State")  # Changed from title to header for better emphasis
with st.sidebar:
    form = st.form(key='match_form')

    # Dropdown menus for team selection with clearer labels
    team1 = form.selectbox('Team Batting First', 
                           ['Islamabad United', 'Karachi Kings', 'Lahore Qalandars', 
                            'Multan Sultans', 'Peshawar Zalmi', 'Quetta Gladiators'],
                           help='Select the team batting first.')
    team2 = form.selectbox('Team Batting Second', 
                           ['Karachi Kings', 'Islamabad United', 'Lahore Qalandars', 
                            'Multan Sultans', 'Peshawar Zalmi', 'Quetta Gladiators'],
                           help='Select the team batting second.')

    # Numeric input fields for match details
    target = form.number_input('Target Runs', min_value=0, step=1, help='Total runs needed to win by the second batting team.')
    cur_runs = form.number_input('Current Runs', min_value=0, step=1, help='Current runs scored by the second batting team.')
    wickets = form.number_input('Wickets Lost', min_value=0, max_value=10, step=1, help='Current number of wickets lost by the second batting team.')
    overs = form.number_input('Overs Played', min_value=0.0, max_value=20.0, step=0.1, help='Number of overs played by the second batting team.')

    submit_button = form.form_submit_button('Predict Win Percentage')

# Main area layout improvements
if submit_button:
    # Calculation for prediction
    balls_left = 120 - (int(overs) * 6 + (overs % 1) * 10)
    runs_left = target - cur_runs
    input_data = {"wickets": wickets, "balls_left": balls_left, "runs_left": runs_left}
    input_data_df = pd.DataFrame(input_data, index=[0])
    prediction = xgbc_model.predict_proba(input_data_df)

    # Displaying the pie chart with enhancements
    fig = px.pie(values=prediction[0], names=[team1, team2],
             title="<b>Match Winning Percentage</b>", 
             color_discrete_sequence=['green', 'blue'],  # More distinct colors
             hole=0.3)  # Creates a donut chart

    # Adding percentage labels
    fig.update_traces(textinfo='percent+label', textfont_size=14)

    # Enhancing the layout
    fig.update_layout(
        title_font_size=22,
        legend_title_font_size=14,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Hover data
    fig.update_traces(hoverinfo='label+percent', textinfo='none')

    st.plotly_chart(fig, use_container_width=True)


    # Result interpretation with enhanced styling
    team2_win_chance = round(prediction[0][1] * 100, 2)
    st.markdown(f'<p class="result-text">Chance of {team2} winning: {team2_win_chance}%</p>', unsafe_allow_html=True)
    st.success("Interpretation : There is a "+str(round(prediction[0][0] * 100))+ "% chance the team batting second ("+ team2 +") is going to lose (or the first team ("+ team1 +") is going to win) and a " + str(round(prediction[0][1] * 100))+"% chance that ("+ team2 +") will win.")
# Footer for additional app information
st.markdown("<hr>", unsafe_allow_html=True)
st.info("This app uses machine learning to predict winning chances in PSL-8 matches. Input the match details on the left and see the prediction.")

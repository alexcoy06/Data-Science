import streamlit as st
import pandas as pd
import plotly.io as px
from Data_Science import *

st.title('Project 4')

# Display the checkbox
show_content = st.checkbox('By checking this box, you acknowledge that this project is based on an open dataset from Kaggle and is not the work of an actual Glassdoor employee.')

# Check if the checkbox is checked
if show_content:
    
    st.header('What I achieved with this project.')
    
    st.write("After analyzing the data, I've come to understand several key outlooks. Firstly, in terms of skill development, I believe I am on the right track to achieve my career goals. Secondly, it can be seen that a high salary doesn't necessarily equate to a high job rating. From personal experience, job satisfaction is often based on how much you enjoy the work, which would be an interesting topic for further analysis. Additionally, based on the industries I have the most experience with and their average salaries, I should expect to easily make close to $100,000.")
    
    st.header('The final result of my process data frame')
    
    st.dataframe(ds)
    
    st.header('These were the graphs I created to reach my conclusion.')
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ['Type of Ownership', 
         'Industry Salary', 
         'Industry Rating', 
         'Salary vs Rating', 
         'Skills Used']
    )

    with tab1:
        st.plotly_chart(ownership_box, theme="streamlit", use_container_width=True)

    with tab2:
        st.plotly_chart(industry_box, theme="streamlit", use_container_width=True)

    with tab3:
        st.plotly_chart(industry_rating, theme="streamlit", use_container_width=True)

    with tab4:
        st.plotly_chart(salary_rating, theme="streamlit", use_container_width=True)

    with tab5:
        st.plotly_chart(skill_bar, theme="streamlit", use_container_width=True)

    footer = """
    ---

    Challenge by [TripleTen](https://tripleten.com/data-science/). Coded by [Alexander Coy](https://alexander-coy.netlify.app/).
    """

    st.markdown(footer)
else:
    st.write("### Please check the box to proceed.")


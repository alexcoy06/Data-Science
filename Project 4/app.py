#!/usr/bin/env python
# coding: utf-8

import pandas as pd 
import plotly.express as px

path = {
    'local': './datasets/Glassdoor_Salary.csv',
    'online': ''
}

def load_csv(file_path):
    try:
        df = pd.read_csv(file_path['local'])
    except FileNotFoundError:
        df = pd.read_csv(file_path['online'])
    return df

ds = load_csv(path)

ds = ds.replace(r'\n', ' ', regex=True)

ds_columns = list(ds.columns)

ds = ds.drop(
    ['Salary Estimate', 'Job Description', 'Location', 'Headquarters', 'Founded', 'Sector', 'Revenue', 'Competitors', 'employer_provided', 'company_txt', 'age'], axis=1
)

ds['hourly'].replace({1: 'hourly', 0: 'salary'}, inplace=True)
ds['same_state'].replace({1: 'local', 0: 'distant'}, inplace=True)
ds.loc[:, ['python_yn', 'R_yn', 'spark', 'aws', 'excel']] = ds[['python_yn', 'R_yn', 'spark', 'aws', 'excel']].replace({1: "Used", 0: "Unused"})

def unit_fix(df, columns):
    for column in columns:
        df[column] *= 1000
    return df

ds = unit_fix(ds, ['avg_salary', 'max_salary', 'min_salary'])

ds_missing = ds.isna().sum()
ds_dupl = ds.duplicated(keep=False)

ds_dupl_check = ds[ds_dupl].sort_values(by='Company Name')

ds.drop_duplicates(inplace=True)
ds = ds.reset_index()
ds_dupl_2 = ds.duplicated(keep=False) 

errors = ((ds == -1) | (ds == '-1')).any(axis=1)
errors_check = ds[errors]
ds = ds[~errors]

ds['job_state'] = ds['job_state'].replace(' Los Angeles', 'CA')

industries = [
    'Computer Hardware & Software', 'Construction', 'Consulting', 'Consumer Products Manufacturing', 'Financial Analytics & Research', 'Insurance Carriers', 'Other Retail Stores', 'Staffing & Outsourcing', 'Video Games'
]

industry = ds[ds['Industry'].isin(industries)]

ds_melted = ds.melt(
    id_vars=["Job Title", "Rating", "Company Name", "Size", "Industry", "hourly", "min_salary", "max_salary", "avg_salary", "job_state", "same_state"], 
    value_vars=["python_yn", "R_yn", "spark", "aws", "excel"], var_name="skill", 
    value_name="required"
)

ds_melted = ds_melted[ds_melted['required'] == 'Used']

skill_stats = ds_melted.groupby(['skill', 'required']).agg({'avg_salary': 'mean', 'Job Title': 'count'}).round().reset_index()
skill_stats = skill_stats.rename(columns={'Job Title': 'total'})

ownership_box = px.box(
    ds, 
    x='Type of ownership', 
    y='avg_salary', 
    title='Average Salary by Type of Ownership', 
    labels={'Type of ownership': 'Type of Ownership', 'avg_salary': 'Average Salary'}
)

unknown = ds[ds['Type of ownership'] == 'Unknown']

industry_box = px.box(
    industry, 
    x="Industry", 
    y="avg_salary", 
    labels={"avg_salary": "Average Salary ($)"}, 
    title="Average Salary by Industry",
)

industry_rating = px.box(
    industry, 
    x='Industry', 
    y='Rating', 
    title='Rating per Industry', 
    labels={'Rating': 'Company Rating'}
)

mean_rating = ds['Rating'].mean()

industry_rating.add_hline(
    y=mean_rating, 
    line_dash="dash", 
    line_color="red"
)

salary_rating = px.scatter(
    ds, 
    x='avg_salary', 
    y='Rating', 
    title='Average Salary vs Company Rating', 
    labels={'avg_salary': 'Average Salary', 'Rating': 'Company Rating'}
)

salary_rating.add_hline(
    y=mean_rating, 
    line_dash="dash", 
    line_color="red"
)

rating_correlation = (ds['avg_salary'].corr(ds['Rating'])).round(4)

skill_bar = px.bar(
    skill_stats, 
    x='skill', 
    y='total', 
    color='total', 
    hover_name='skill', 
    labels={'skill': 'Skill Used', 'total': 'Number of Times Used'}, 
    title='Skills Used'
)

import streamlit as st

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


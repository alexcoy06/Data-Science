#!/usr/bin/env python
# coding: utf-8

# ## 1 Project Overview and Setup: Understanding the Purpose

# The primary objective of this project is to leverage a comprehensive dataset extracted from the [Glassdoor website](https://www.kaggle.com/datasets/fahadrehman07/data-science-jobs-and-salary-glassdoor), which includes detailed information on data science jobs and salaries. This dataset provides a rich source of information, covering job titles, estimated salaries, job descriptions, company ratings, and essential company details such as location, size, and industry. By conducting a thorough analysis of this dataset, we aim to gain valuable insights into the job market, identify key trends, and understand what to expect when job hunting or researching career opportunities in the data science field.
# 
# To make these insights accessible and actionable, the analysis will be visually represented on a self-made website. This approach ensures that the findings are not only informative but also easily interpretable, catering to a wide audience ranging from job seekers to industry analysts. The goal is to provide a clear and comprehensive overview of job market trends, helping users make informed career decisions based on reliable data.
# 
# <u>Note: this file will be converted to a python file using `jupyter nbconvert --to script Data_Science.ipynb` in the terminal.<u>
# 

# ## 2 Initialization

# ### 2.1 Add imports

# To begin our analysis, we first need to set up our Jupyter Notebook environment and import the necessary libraries.

# In[1]:


import pandas as pd 
import plotly.express as px


# - Pandas simplifies data organization, converting messy CSV files into tidy, easy-to-handle formats, simplifying analysis.
# - And `plotly.express` is a high-level, easy-to-use interface for creating interactive and complex visualizations in Python using Plotly.

# ### 2.2 Set up DataFrames

# To ensure the Jupyter notebook works in all environments, I will configure the necessary paths, using a `dictionary`.

# In[2]:


path = {
    'local': './datasets/Glassdoor_Salary.csv',
    'online': ''
}


# With all paths set, I will now create a function that selects the correct path regardless of the current environment, using `exception handling`.

# In[3]:


def load_csv(file_path):
    try:
        df = pd.read_csv(file_path['local'])
    except FileNotFoundError:
            df = pd.read_csv(file_path['online'])
    return df


# Now that `load_csv` is created, I can run the `path` dictionary through it to generate the main dataframe for this project. To store and access this dataframe, I will initialize it and assign it to a variable of my choosing.

# In[4]:


ds = load_csv(path)


# The variable `ds` is now initialized, abbreviated for Data Science.

# ## 3 Preparing the Data

# To effectively utilize the data frame, it's crucial to inspect it first. Addressing any issues found is necessary to ensure the accuracy and usability of the data.

# ### 3.1 Initial Inspect

# First, we need to examine `ds` to determine the type of information it contains.

# In[5]:


ds


# The initial step I'd like to take is to remove the `\n` characters, as they likely signify new lines but serve no purpose in `ds`.

# In[6]:


ds = ds.replace(r'\n', ' ', regex=True)
ds


# Personally, I find it significantly improves the appearance.

# ### 3.2 Simplify `ds`

# As observed, `ds` has 28 rows, not all of which are displayed above. Next, I would like to review all the rows and decide which ones are actually needed.

# In[7]:


ds_columns = list(ds.columns)
ds_columns


# I believe the following columns are not needed for this project:
# 
# - Salary Estimate
# - Job Description
# - Location
# - Headquarters
# - Founded
# - Sector
# - Revenue
# - Competitors
# - employer_provided
# - company_txt
# - age

# In[8]:


ds = ds.drop(
    ['Salary Estimate', 'Job Description', 'Location','Headquarters','Founded', 'Sector', 'Revenue', 'Competitors', 'employer_provided', 'company_txt', 'age'], axis=1
)
ds.head()


# Now, `ds` contains only the columns I am most interested in. All columns are now displayed without any being hidden.
# 
# Note that we removed `Salary Estimate` because it would be redundant to keep it, considering we have `min_salary` and `max_salary`.

# ### 3.3 Check the Data Types

# Now, let's execute the `info()` method to ensure that each column contains a value that enables the column to be utilized effectively.

# In[9]:


ds.info()


# The columns `hourly`, `same_state`, `python_yn`, `R_yn`, `spark`, `aws`, and `excel` contain numerical data types that are not immediately clear. These values are binary, where `1` signifies `true` and `0` signifies `false`.
# 
# Each of these columns needs to be converted to a more descriptive format.

# In[10]:


ds['hourly'].replace({1: 'hourly', 0: 'salary'}, inplace=True)

ds['same_state'].replace({1: 'local', 0: 'distant'}, inplace=True)

ds.loc[:, ['python_yn', 'R_yn', 'spark', 'aws', 'excel']] = ds[['python_yn', 'R_yn', 'spark', 'aws', 'excel']].replace({1: "Used", 0: "Unused"})

ds.info()


# Now the chart is a bit easier to understand, and the data types better represent each column with more relatable values.
# 
# Additionally, `min_salary`, `max_salary`, and `avg_salary` are shown in units of tens instead of thousands.

# In[11]:


def unit_fix(df, columns):
    for column in columns:
        df[column] *= 1000
    return df

ds = unit_fix(ds, ['avg_salary', 'max_salary', 'min_salary'])

ds.head()


# ### 3.4 Duplicate and Missing values 

# Now that we have all the necessary columns with corrected values, we can begin checking for missing or duplicated values in `ds`. Once this is completed, we will have ensured the accuracy and usability of the data.

# In[12]:


ds_missing = ds.isna().sum()
ds_dupl = ds.duplicated(keep=False)

print(f'There are {ds_dupl.sum()} duplicate values, and {ds_missing.sum()} missing values.')


# It appears that some rows are duplicated, but there are no missing values. Therefore, I want to sample and check some of the duplicated rows.

# In[13]:


ds_dupl_check = ds[ds_dupl].sort_values(by='Company Name')
ds_dupl_check.head(6)


# It's evident that there are duplicated values that need to be removed.

# In[14]:


ds.drop_duplicates(inplace=True)
ds = ds.reset_index()
ds_dupl_2 = ds.duplicated(keep=False) 

print(f'There are {ds_dupl_2.sum()} duplicate values.')


# Now that we have removed all duplicated values, we can proceed to finally use the data.

# ### 3.5 Hindsight correction

# As the saying goes, "hindsight is 20/20." It's often noticed during data analysis, after processing and organizing, that there are negative one numbers and "-1" strings that distort the data and are logically inaccurate. I'll go back to address this issue earlier in my notebook to ensure more accurate calculations later on. 

# In[15]:


errors = ((ds == -1) | (ds == '-1')).any(axis=1)
errors_check = ds[errors]
errors_check


# As can be seen, for some reason, certain columns such as `Rating` contain a value of -1, which is logically impossible since ratings should range from 0 to 5.

# In[16]:


ds = ds[~errors]


# The above code edits `ds` to exclude all rows with erroneous values. With this step completed, we are almost done.
# 
# There is one more error concerning `job_state`. Instead of showing the initials of each state, there is an instance where the value is ` Los Angeles`.

# In[17]:


ds['job_state'].unique()


# As previously stated, a value within the 'job_state' column is not a valid state abbreviation.

# In[18]:


ds['job_state'] = ds['job_state'].replace(' Los Angeles', 'CA')
ds['job_state'].unique()


# With this task finished, we can now effectively analyze the data from GlassDoor.

# ## 4 Criteria for Analysis 

# ### 4.1 Inspecting `ds` for Analysis

# Now, my goal is to group the data into distinct data frames, focusing on various points of interest. Let's check `ds` once more to determine these points.

# In[19]:


ds


# After reviewing 'ds,' I recommend a detailed analysis of the following: `Type of ownership`, `Industry`, skills used, `Ratings`, and the salary ranges.
# 
# The remaining columns, while valuable, are less relevant to my needs. For example:
# - I work remotely, so the job's location is irrelevant.
# - Job titles can vary for the same position.

# ### 4.2 Column Inspects

# To prevent any unforeseen issues, it is advisable to closely examine each column of interest, excluding two types:
# - Columns with binary values
# - Columns with numerical values

# In[20]:


ds['Type of ownership'].unique()


# All values for the 'Type of ownership' appear to be straightforward.

# In[21]:


ds['Industry'].unique()


# The 'Industry' column contains numerous values, many of which are not of interest. Therefore, it would be most beneficial to focus on industries in which I have the most experience.

# In[22]:


# Created a list of the industries im interested in
industries = [
    'Computer Hardware & Software', 'Construction', 'Consulting', 'Consumer Products Manufacturing', 'Financial Analytics & Research', 'Insurance Carriers', 'Other Retail Stores', 'Staffing & Outsourcing', 'Video Games'
]

# set a dataframe to only show the rows that include the `industries`
industry = ds[ds['Industry'].isin(industries)]


# The `industries` list includes all the industries I am familiar with, either professionally or through personal interest. Using this list, I created `industry`, which contains only the rows from `ds` that match any values listed in `industries`.

# ### 4.3 Reformating Skill Usage Statistics

# We could use `ds` as it is for the skills used, but that wouldn't be very beneficial. We would either end up with single-row data frames or a data frame with rows showing all the different combinations of skills used. Instead, we only want one dataframe showing each skill used with the associated data.

# In[23]:


ds_melted = ds.melt(
    id_vars=["Job Title", "Rating", "Company Name", "Size", "Industry", "hourly", "min_salary", "max_salary", "avg_salary", "job_state", "same_state"], 
    value_vars=["python_yn", "R_yn", "spark", "aws", "excel"], var_name="skill", 
    value_name="required"
)


# Now, `ds` is restructured into `ds_melt` by converting the specified columns into rows and creating two new columns. This expands the DataFrame to better display the skills required for each job listing.

# In[24]:


ds_melted = ds_melted[ds_melted['required'] == 'Used']


# Now we have `ds_melt`, focusing specifically on the skills being used.
# 
# Now we can perform the same process as we did in the previous section, using the `groupby()` function.

# In[25]:


# Group by the skills and their usage status
skill_stats = ds_melted.groupby(['skill', 'required']).agg({'avg_salary': 'mean', 'Job Title': 'count'}).round().reset_index()
skill_stats = skill_stats.rename(columns={'Job Title': 'total'})
skill_stats


# Unlike the other points of interest, where we focused on salary, with `skill_stats`, our primary interest is in the usage of each skill. 

# ## 5 Job analysis

# ### 5.1 `Type of ownership`

# First, I will examine the different types of ownership and their relationship with salary.

# In[26]:


ownership_box = px.box(
    ds, 
    x='Type of ownership', 
    y='avg_salary', 
    title='Average Salary by Type of Ownership', 
    labels={'Type of ownership': 'Type of Ownership', 'avg_salary': 'Average Salary'}
)
ownership_box.show()


# Based on the data, most types of ownership can offer salaries below $50,000 per year. However, they are also generally capable of paying over $100,000 per year. Notably, hospitals and unknown ownership types tend to offer the lowest salaries.

# In[27]:


unknown = ds[ds['Type of ownership'] == 'Unknown']
unknown


# Curiosity led me to investigate jobs with an 'unknown' ownership type. I was relieved to find that this job was in an industry I have no experience. Despite having a rating of four, this job did not require any data scientist skills, which likely explains the lower pay. Overall, this was an interesting discovery.

# ### 5.2 `Industry`

# The previous section has increased my interest in understanding the range of salary rates for the industries in which I have experience.

# In[28]:


industry_box = px.box(
    industry, 
    x="Industry", 
    y="avg_salary", 
    labels={"avg_salary": "Average Salary ($)"}, 
    title="Average Salary by Industry",
)
industry_box.show()


# When comparing the industries I have experience in to the different types of business ownerships, I notice some differences. Among the industries I have experience in, only Construction falls completely below $50,000. In contrast, the industries I'm skilled in tend to average fairly close to or well above $100,000.

# In[29]:


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

industry_rating.show()


# Despite the previous analysis indicating that many of the jobs I have experience in are high-paying, it's worth noting that the ratings don't necessarily reflect them as the best jobs to have. While I **wouldn't** say these jobs **'aren't worth having,'** a significant portion of them fall below the average. The clear winners here appear to be 'Computer Hardware & Software' and 'Consulting'.

# ### 5.3 Salary

# To better understand my previous statement, I would like to further explore the relationship between salary and ratings for various jobs.

# In[30]:


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

salary_rating.show()


# Regarding the data in the above graph, it can be observed that there are above-average ratings at both the high and low ends of the salary spectrum. These ratings are heavily concentrated between $50,000 and $150,000. While this distribution makes it challenging to interpret the graph qualitatively, calculating the correlation can provide a clearer understanding of the relationship.

# In[31]:


rating_correlation = (ds['avg_salary'].corr(ds['Rating'])).round(4)
rating_correlation


# With a correlation coefficient of 0.1146, this indicates a weak positive relationship between the average salary and company rating. The correlation is not strong enough to suggest that higher pay corresponds to better-rated jobs.

# ### 5.4 Skills

# Finally, I would like to determine if there are any additional tech skills that might be worth adding to my skillset.

# In[32]:


skill_bar = px.bar(
    skill_stats, 
    x='skill', 
    y='total', 
    color='total', 
    hover_name='skill', 
    labels={'skill': 'Skill Used', 'total': 'Number of Times Used'}, 
    title='Skills Used'
)
skill_bar.show()


# Based on the analysis, I've identified the following key skills and their relevance to my career development:
# 
# - The most used skills are Python and Excel, both of which I am experienced in.
# - AWS and Spark are also highly used; I plan to gain a better understanding of Spark and might pursue an AWS certification in the future.
# - The programming language R appears to be less valuable to learn at this time, as it is the lowest-paid skill on average.

# ## 6 Conclusion 

# After analyzing the data, I've come to understand several key outlooks. Firstly, in terms of skill development, I believe I am on the right track to achieve my career goals. Secondly, it can be seen that a high salary doesn't necessarily equate to a high job rating. From personal experience, job satisfaction is often based on how much you enjoy the work, which would be an interesting topic for further analysis. Additionally, based on the industries I have the most experience with and their average salaries, I should expect to easily make close to $100,000.

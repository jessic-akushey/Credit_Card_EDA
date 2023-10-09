#!/usr/bin/env python
# coding: utf-8

# # India Credit Card Spendings EDA
# ![cards.jpg](attachment:cards.jpg)
# 
# 
# A prominent bank in India offers different ways of payments: __Gold, Platimum, Silver__ and __Signature__. In order to promote these methods of payment, they have gathered data on the expenditure habits of their customer within the country. An analysis is neccessary to any key business questions to facilitate the promotion exercise. 
# 
# This task covers the analysis and prediction of the spending habits of Indians in 986 cities. It uses data available on [Kaggle](https://www.kaggle.com/datasets/thedevastator/analyzing-credit-card-spending-habits-in-india). 
# 
# The task is divided into seven phases.
# 1. Data Importing and Inspection.
# 2. Feature Engineering
# 3. Business Solution Data Analysis
# 

# The data is imported from __Kaggle's__ database repository

# # Data Importing and Inspection
# 
# In this section, various libraries useful for the tasks metioned are imported. The file containing the data that will be worked on is also imported.
# Also the data will be 

# ### Importing the libraries needed

# In[1]:


# importing useful libraries
import pandas as pd #for importing data and dataframe manipulation
import numpy as np  # for data array and matrix manipulation

import matplotlib.pyplot as plt # for plotting data
import plotly.express as px # for data visualization


# ### Reading the csv dataset
# 
# The file containing the data is a comma seperated val file. The Pandas library has a function that can be used to import files of different formats including .csv files.

# In[2]:


# The data is read using pandas' read_csv function and using the index column as the index of the imported data
df = pd.read_csv('Credit card transactions India.csv', index_col = 'index')


# ### Inspecting the data

# In[3]:


# printing the first five elements of the table
df.head()


# As seen the table contain the City of transaction, Date of transaction, Card Type used, Expense paid for, Gender of the person who made the payment and the Amount paid

# In[4]:


df.info()


# The dataset has 6 columns with 26,502 rows and no explicit null values. The data types of the columns are also right except the "__Date__" column which will be corrected later

# In[5]:


# printing a description of the various columns of the data including both numerical and string columns
df.describe(include="all")


# A description of the dataset shows that there are __986__ distinct cities out of __26052__, __2__ gender types, __4__ different card types used and __6__ expense types

# # Data Exploration

# In this section, the different data attributes are explored further, example looking at the distinct values 

# ##### Distinct Gender values

# In[6]:


# finding the different genders using the value_counts() function
df['Gender'].value_counts()


# In[7]:


# finding the different card types using the value_counts() function
df['Card Type'].value_counts()


# In[8]:


# finding the different expense types using the value_counts() function
df['Exp Type'].value_counts()


# In[9]:


# finding the different cities and their frequency using the value_counts() function
df['City'].value_counts().head(10)


# #### Findings

# | Column | Distinct Values | Comments |
# | --- | --- | --- |
# | Gender | Females and Males | Females have a higher count than Males. |
# | Card Type | Silver, Gold, Platinum and Signature | Silver has the most card type record. It will be worth looking into why that is so. |
# | Exp Type | Food, Fuel, Bills, Entertainment, Grocery and Travel | People generally spend on food most times and on Travel the least. |
# | City | 986 distinct cities | The margin between the top 4 cities is a lot higher than the rest of the cities and some cities also record single transactions. |
# | Amount | None | The description of the amount column already shows its feature. |

# None of the columns contains null values, and the value counts also show no implicit error values. The entries however, could contain duplicate rows which might lead to a higher frequency count of the various distinct values

# ###### Checking to see duplicacy

# In[10]:


# checking if there are any duplicated rows in the dataset
df.columns.duplicated().sum()


# There are no duplicated rows in the data

# The data exploration is now complete.

# ## Data Preprocessing
# 
# This section involves taking a critical look at the various attributes/features.

# In[11]:


# printing the first 5 rows
df.head()


# ### The City column
# 
# Feature Engineering the __City__ column

# In[12]:


# Viewing the new dataset
df.head()


# Since the data is about spending in India it is irrelevant to have to country attached to the city in the __City__ column.

# To correct this, City column is split at the ', ' and the city name is taken using its index and assigned to df['City']

# In[13]:


# splitting the city column and assigning the city to a new column
df['City'] = df['City'].astype(str).str.split(", ", expand = True)[0]


# In[14]:


# printing dataset to inspect the city column
df.head()


# In[ ]:





# ## Business Solution Data Analysis

# ### Question 1: Which 10 city was the difference in the total expenditure between males and females the most and the least?

# #### Reason:
# As a bank it is relevant to know the various locations where customers use the cards the most. Knowing this helps to better imrove targeting of advertisement. Further knowing whether males and females have the same spending will inform the bank when to perform a targeted advertisement or a general adversitement.

# #### Procedure:
# 
# 1. To be able to do this the total expenses per gender for the city has to be generated from the table, and this can be done using a groupby function
# 2. From the new table the Males and Females can be seperated into different datasets
# 3. The two datasets will be merged into a single dataset that has columns as City, Male Amount, and Female Amount.
# 4. The difference between the male and female amount can then be calculated.

# In[15]:


# Finding cities with the highest expenditure
city_group = df.groupby('City', as_index=False)['Amount'].aggregate('sum')            .sort_values(by='Amount', ascending=False)
city_group


# In[16]:


# creating a groupby table of City and Gender with the total amount as the sum
gender_group = df.groupby(['City', 'Gender'], as_index = False).sum()

# Seperating the Male and Female total amounts and renaming the column for better identification
Male_Amount = gender_group[gender_group['Gender']=='M'].drop('Gender', axis=1)
Male_Amount.rename(columns={'Amount':'Male_Amount'}, inplace=True)

Female_Amount = gender_group[gender_group['Gender']=='F'].drop('Gender', axis=1)
Female_Amount.rename(columns={'Amount':'Female_Amount'}, inplace=True)

# merging the two datasets
Total_amount_Gender = Male_Amount.merge(Female_Amount, on = 'City')

# printing the new dataset
Total_amount_Gender.head()


# In[17]:


# calculating the difference in Expenditure Amount
Amount_Difference = []  # list to store the difference values

for i in Total_amount_Gender.City.unique():
    
    # calculating the difference in amount and storing it in a variable diff (type cast amounts to integers)
    diff = int(Total_amount_Gender[Total_amount_Gender.City == i]['Male_Amount'])-         int(Total_amount_Gender[Total_amount_Gender.City == i]['Female_Amount'])
    
    # appending the differenct to a list
    Amount_Difference.append(diff)

# creating a new column whose values are the differences calculated
Total_amount_Gender['Difference'] = Amount_Difference


# Per the difference formular above, having a positive difference shows that males spend more than females and having a negative amount shows that femaless spend more than males.
# 
# The dataset is seperated into positive and negative value differences.

# In[18]:


# seperating the dataset into positive and negative difference values 
Female_Dominated_Areas = Total_amount_Gender[Total_amount_Gender['Difference']<0]
Male_Dominated_Areas = Total_amount_Gender[Total_amount_Gender['Difference']>0]


# In[19]:


# Printing the top 10 rows where females spend more than males
Female_Dominated_Areas.sort_values(by='Difference')[['City', 'Difference']].head(10)


# In[20]:


# Printing the bottom 5 rows where females spend more than females
Female_Dominated_Areas.sort_values(by='Difference', ascending=False)[['City', 'Difference']].head(5)


# In[21]:


# Printing the top 10 rows where males spend more than females
Male_Dominated_Areas.sort_values(by='Difference', ascending=False)[['City', 'Difference']].head(10)


# In[22]:


# Printing the bottom 5 rows where males spend more than females
Male_Dominated_Areas.sort_values(by='Difference')[['City', 'Difference']].head(5)


# #### Conclusion:
#     
# As a bank it will be best to advertise in Greater Mumbai, Delhi, Bengaluru, Ahmedabad, Kanpur, Jaipur, etc to females since they spend the most.
# 
# Also, it will be best to advertise in Kolkata, Chennai, Fatehpur Sikri, Margao, Nautanwa, etc to males since they spend the most.

# ### Question 2: Which expense type generates the most revenue and what are the card types used?

# #### Reason:
# Since the bank provides 4 different card types, it could be neccessary to know the card type that generates more revenue and have more of that in stock.
# 
# 
# #### Procedure:
# 1. The dataset is grouped, to be able to know the expense type, the card type and the total amount spent with that card type using a group by function
# 2. Visualizing the data to give a clearer view

# In[23]:


# finding the sum of amount for each combination of Expense Type and Card Type using the groupby function 
Expense = df.groupby(["Exp Type", "Card Type"], as_index=False).sum()

# printing the dataset
Expense


# In[24]:


# creating a bar chart of the amounts spent per expense type per card type
px.bar(Expense, x="Exp Type", y="Amount", color="Card Type", text_auto=True)


# #### Conclusion:
# 
# From the car chart is can be seen that Card Type __Silver__ is used the most for every expense type.
# Also __Bills__ has the highest expense amount. Hence card type __Silver__ should be stocked the most and be available to individuals who pay __Bills__ the most.

# ### Question 3: What are the amount monthly trends for the card types with the various related-expenses

# #### Reason:
# 
# Knowing the how people spend per month per card could inform the bank policy maker as to when they should roll out or intensify services in increase card spendings.
# 
# 
# #### Procedure:
# 1. Extract the Month names from the __Date__ column.
# 2. Finding the total expenditure for the different months per card type.
# 3. Creating a plot to visualize the trend of the data.

# This task requires data on a monthly basis. However, the dataset contains only the full date of transaction. Since the __Date__ column can be converted into data type datetime, the months, days, years and date number can be easily extracted

# ### Creating a class to extract Year, Month, Day Number and Date Name from the "Date" column

# In[25]:


# creating a class called DateColumnsExtractor
class DateColumnsExtractor:
    def __init__(self, date_format='%d-%m-%Y'):
        self.date_format = date_format
    
    def extract_year(self, df, date_column):
        """
        Extracts the year from a date column in a Pandas DataFrame
        
        Returns:
        pandas.DataFrame: The original DataFrame with a new column for the year
        """
        
        # converting the Date column into a datetime datatype and extracting the year and assigning it to a new column
        df['Year'] = pd.to_datetime(df[date_column]).dt.year
        
        # returning the new dataframe
        return df
    
    def extract_month_name(self, df, date_column):
        """
        Extracts the month name from a date column in a Pandas DataFrame
              
        Returns:
        pandas.DataFrame: The original DataFrame with a new column for the month name
        """
        
        # converting the Date column into a datetime datatype and extracting the Month name and assigning 
        # it to a new column
        df['Month'] = pd.to_datetime(df[date_column]).dt.month_name()
        
        # returning the new dataframe
        return df
    
    def extract_day_number(self, df, date_column):
        """
        Extracts the day number from a date column in a Pandas DataFrame
                
        Returns:
        pandas.DataFrame: The original DataFrame with a new column for the day number
        """
        
        # converting the Date column into a datetime datatype and extracting the Day number and assigning 
        # it to a new column
        df['Day Number'] = pd.to_datetime(df[date_column]).dt.day
        
        # returning the new dataframe 
        return df
    
    def extract_day_name(self, df, date_column):
        """
        Extracts the day name from a date column in a Pandas DataFrame
        
        Returns:
        pandas.DataFrame: The original DataFrame with a new column for the day name
        """
        
        # converting the Date column into a datetime datatype and extracting the Day name and assigning 
        # it to a new column
        df['Day Name'] = pd.to_datetime(df[date_column]).dt.day_name()
        
        # returning the new dataframe
        return df


# In[26]:


# calling the datecolumnsextractor to a variable date_extractor 
date_extractor = DateColumnsExtractor()

# extracting the month name from the date column
df = date_extractor.extract_month_name(df, 'Date')


# In[27]:


# checking to see if the Month name column has beeen created
df.head()


# In[28]:


# finding theh total amount spent per month per card type
df_month_group=df.groupby(["Month", "Card Type"]).sum()

# printing the grouped data
df_month_group.head(10)


# ##### Plotting the data

# In[29]:


# creating a dictionary to store the data to be plotted
cat={}

# setting the size of the plot
plt.figure(figsize=(15, 7))

# create a line chart for each category
for category, category_df in df_month_group.groupby('Card Type', as_index=False):
    
    # storing the key and its value in the dictioanry
    cat[category]=category_df
    
    # creating a line plot of the data
    plt.plot(category_df.index.get_level_values('Month'), category_df['Amount'], label=category)

# add labels and title
plt.xlabel('Card Type')
plt.ylabel('Amount Spent')
plt.title('Total Amount Spent with Specific Card Type')

# showing the legends
plt.legend()

# display the chart
plt.show()


# #### Conclusion:
# 
# From the trend graph, it can be seen that the months of August, July, June and September has the lowest total expdeniture amounts. More incentives can be offered withing these months to increase spendings. 

# ### Question 4: What is the spending habit of males and females for the different days of the week and months of the year?

# #### Reason:
# Knowing the amount spent  could further improve adverstisement or facilitate the provision of incentives for those days and months to increase spendings
# 
# 
# #### Procedure:
# 1. The total expenditure per day name and months should be grouped according to gender
# 2. Plot a chart showing the total amount spent for easy visualization

# In[30]:


# extracting the month name from the date column

df = date_extractor.extract_day_name(df, 'Date')


# In[31]:


# grouping the data per day and gender
spending_per_day = df.groupby(["Day Name", "Gender"], as_index=False)        .sum()
# printing the grouped data
spending_per_day

px.bar(spending_per_day, x="Day Name", y="Amount", color="Gender", text_auto=True)


# In[32]:


# grouping the month and gender
spendings_per_month = df.groupby(["Month", "Gender"], as_index=False)        .sum()
# printing the grouped data
spendings_per_month

# creating a chart of the data
px.bar(spendings_per_month, x="Month", y="Amount", color="Gender", text_auto=True)


# #### Conclusion:
# 
# There isn't a lot of variation in the spending between males and females, however generally females spend more than males in both weekly and monthly categorizations.

# ### Question 5: Which expense type do customers spend the most on at the end of each month?

# #### Reason: 
# The end of the month is usually the period where people recieve their salaries. Knowing how they spend and providing incenties to increase spending will increase the revenue generated.
# 
# #### Procedure:
# 1. Extract the day number from the data column
# 2. Subset the data based on the day numbers for the end of the month, that is 29th, 30th and 31st.
# 3. Group the data by end of those days
# 4. Create a chart showing the expenses

# In[33]:


# extracting the month name from the date column

df = date_extractor.extract_day_number(df, 'Date')


# In[34]:


def DaysNum(df, days):
    DayData = pd.DataFrame()
    for i in days:
        data = df[df['Day Number']==i]
        DayData = pd.concat([DayData, data])
    return DayData


# In[35]:


# list of desired days
Days = [29, 30, 31]

# using the function to subset the dataset for the 29th, 30th and 31st
End_of_Month = DaysNum(df, Days)


# In[36]:


# grouping the subset data using the egroupby function
End_of_month_grouped = End_of_Month.groupby(['Day Number', 'Month', 'Exp Type'], as_index=False).sum()


# In[37]:


# for loop to create a chart
for i in Days:
    # regrouping the data for only day at a time
    End_of_month_grouped_ = End_of_month_grouped[End_of_month_grouped['Day Number']==i].drop('Day Number', axis=1)
    
    # plotting the chart
    fig = px.bar(End_of_month_grouped_, x="Month", y="Amount", color="Exp Type", text_auto=True, 
                 title=f"Expenses of Various Expense Types for day {i} of each month")
    
    # show plot
    fig.show()


# #### Conclusion:
# 
# Generally more money is spent on travel within the months of December and January. Advertisement during these months can be intensified for __Travel Expenses__. January, December, March and October have the highest expenditure at the end of the months

# ## ADDITIONAL ANALYSIS TO USE OTHER FUNCTIONS DISCUSSED

# ### Using __IF__ staements

# ### Checking if a particular City uses any of the cards produced by the bank
# 

# In[38]:


if "Salem" in df['City'].unique():
    print("Card is present in Salem")
else:
    print('Card is not present')


# ### Using FOR and IF statement together

# ### Checking if a particular Cities in a tuple uses any of the cards produced by the bank
# 

# In[39]:


# tuple of cities to check
cities = ("Amravati","Tezpur", "Dimapur", "Gangtok", "Noida", "Ballia")

for city in cities:
    if city in df['City'].unique():
        # printing the results using string formating
        print("Card is present in {}".format(city))
    else:
        # printing the results using string formating
        print('Card is not present in {}'.format(city))


# ## Using string and list operations and multiple nested logical operations

# ### Finding cities that start with a particular letter and contain a number of letter
# 

# In[40]:


# list to store the cities
cities = []

# for loop to findd cities whose name begin with J or K and have a length greater than 10
for city in df['City'].unique():
    if ((city[0] == "J") or (city[0] == "K")) and (len(city) > 10):
        cities.append(city)

# Printing the cities
print(cities)


# # FINAL CONCLUSSION AND DISCUSSION

# Based on the analysis performed, if the goal of the bank is to increase the amount of spent by customer, they should:
# 1. Advertise more to the female population
# 2. Advertise more in the following cities Greater Mumbai, Delhi, Bengaluru, Ahmedabad and Dehli since most spendings come those cities.
# 3. In the month of December and January, more incetivess should be provided on travel to further boost spendings
# 4. Card Type Silver is used the most for every expense type hence it should be stocked the most
# 
# If the goal of the bank is to reach more customers, they should:
# 1. Advertise more to the male population
# 2. Provide more spending incentive for cities like Fazilka,	Mahbubnagar, Bahraich, Tirur and Changanassery
# 
# __Limitations of analysis__
# 1. Data ranges from 2013 to 2015. More data would give a more comprehensive insight in spending habit.
# 2. A break down of the expense type is not added in the analysis 
# 3. Age ranges of customers were not analyzed. Hence customers could not be targeted for advertisement based on their ages.
# 
# __Implication for business__
# Applying the recommendations could potentially increase the number of customers using the different cards and the amount of revenue that will be generated

# In[ ]:





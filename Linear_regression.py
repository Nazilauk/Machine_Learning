



#Importing Required Libraries
#We will use pandas to read the dataset, seaborn for visualization, and scikit-learn for linear regression analysis.
#--------------------------------------------------------------------------------------
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Reading the Dataset
#The dataset we will be using is "insurance.csv" and we will store it in a variable called "insurance_df".
insurance_df = pd.read_csv("insurance.csv")
#--------------------------------------------------------------------------------------------------------------------------
#After reading in our dataset, we will create a scatterplot to visualize the relationship between age and insurance charges. 
# We will use seaborn's sns.scatterplot() method to create the plot. 
#---------------------------------------------------------------------------------------------------------------------------


g = sns.scatterplot(x="age", y="charges", hue="smoker", data=insurance_df, palette=['green','orange'])

plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------
#Building a Linear Regression Model
#Now we will use linear regression from scikit-learn to fit a model to our data.
# We will store the independent variable ("age") in a variable called "X" and dependent variable ("charges") in a variable called "Y". 
# Then we will create an instance of the LinearRegression class and fit the model using the fit() method.
#-------------------------------------------------------------------------------------------------------------------------------------

X = insurance_df[['age']]
Y = insurance_df['charges']

lr = LinearRegression()
lr.fit(X, Y)

#Making Predictions and Creating a Best Fit Line
#After fitting the model, we can now make predictions on data. We will create a new column in our original dataframe called "predictions" using the built-in predict() method.
# From here, we can create a scatterplot with a best fit line using seaborn's sns.regplot() method.
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

insurance_df['predictions'] = lr.predict(X)

g = sns.scatterplot(x="age", y="charges", data=insurance_df)

#Displaying the Plot
sns.regplot(x="age", y="charges", data=insurance_df).set(title='Age vs. Insurance Charge', xlabel= 'Age', ylabel= 'Insurance Charge')
plt.show()

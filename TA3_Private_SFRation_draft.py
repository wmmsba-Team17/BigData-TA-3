# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:18:32 2020

@author: lesli
"""

from __future__ import print_function

from pyspark.ml.regression import LinearRegression

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark import SparkConf, SparkContext

#method 1
# Create a SparkSession (Note, the config section is only for Windows!)
spark = SparkSession.builder.config("spark.sql.warehouse.dir", "file:///C:/temp").appName("LinearRegression").getOrCreate()

'''# Load up our data and convert it to the format MLLib expects.
inputLines = spark.sparkContext.textFile("college.csv")'''


#method 2 
# Using RDD to filter school type 
conf = SparkConf().setMaster("local").setAppName("GradRate")
sc = SparkContext(conf = conf)

lines = sc.textFile("file:///Users/lesli/BigData/TA3/College.csv")  #We deleted the header row 

def parseLine(line):
    fields = line.split(',')
    #college = fields[0]
    SchoolType = fields[1]
    SF_Ratio = float(fields[15])
    Grad_Rate = float(fields[18])
    return (SchoolType, SF_Ratio, Grad_Rate)

parsedLines = lines.map(parseLine)

Private = parsedLines.filter(lambda x: "Yes" in x[0])   #filter to keep private school only 

# Select SF_Ration and Grad_Rate as DF columns.
data = Private.map(lambda x: x.split(",")).map(lambda x: (float(x[0]), Vectors.dense(float(x[1]))))


#Private school only by using filter, and then selct S.F ration column and Graduataion Rate column as the data.
#data = inputLines.map(lambda x: x.split(",")).filter(lambda x: "Yes" in x[1]).map(lambda x: (float(x[15]),float(x[18]))).cache()

# Convert this RDD to a DataFrame
colNames = ["Grad.Rate","S.F Ratio"]  #Y is Grad.rate
df = data.toDF(colNames)

# Note, there are lots of cases where you can avoid going from an RDD to a DataFrame.
# Perhaps you're importing data from a real database. Or you are using structured streaming
# to get your data.

# Let's split our data into training data and testing data
trainTest = df.randomSplit([0.8, 0.2])
trainingDF = trainTest[0]
testDF = trainTest[1]

# Now create our linear regression model
lir = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Train the model using our training data
model = lir.fit(trainingDF)

# Now see if we can predict values in our test data.
# Generate predictions using our linear regression model for all features in our
# test dataframe:
fullPredictions = model.transform(testDF).cache()

# Extract the predictions and the "known" correct labels.
predictions = fullPredictions.select("prediction").rdd.map(lambda x: x[0])
labels = fullPredictions.select("label").rdd.map(lambda x: x[0])

# Zip them together
predictionAndLabel = predictions.zip(labels).collect()

# Print out the predicted and actual values for each point
for prediction in predictionAndLabel:
  print(prediction)

# Stop the session
spark.stop()

# !spark-submit TA3_Private_SFRation_draft.py


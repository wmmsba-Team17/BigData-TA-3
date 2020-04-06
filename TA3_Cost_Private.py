# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 15:16:17 2020

@author: lesli
"""

from __future__ import print_function

#from pyspark import SparkConf, SparkContext
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession, SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
#from pyspark.sql import functions

# Create a SparkSession (Note, the config section is only for Windows!)
spark = SparkSession.builder.config("spark.sql.warehouse.dir", "file:///C:/temp").appName("LinearRegression").getOrCreate()

inputLines = spark.sparkContext.textFile("file:///Users/lesli/BigData/TA3/College.csv")

def parseLine(line):
    fields = line.split(',')
    SchoolType = fields[1]
    Room_Board = float(fields[10])
    Books = float(fields[11])
    Expend = float(fields[17])
    Grad_Rate = float(fields[18])
    return (SchoolType, Room_Board, Books, Expend, Grad_Rate)

parsedLines = inputLines.map(parseLine)


Private = parsedLines.filter(lambda x: "Yes" in x[0]) 

data = Private.map(lambda x: (Vectors.dense(float(x[1])),Vectors.dense(float(x[2])),Vectors.dense(float(x[3])),float(x[4]))).cache()


# Convert this RDD to a DataFrame
colNames = ['Room_Board','Books','Expend','Grad_Rate'] #Y is Grad.rate
df = data.toDF(colNames)
#vector assembler method 
vectorAssembler = VectorAssembler(inputCols = ['Room_Board','Books','Expend'],
                                  outputCol = 'Total_Cost')


College_df = vectorAssembler.transform(df)
College_df = College_df.select(['Total_Cost', 'Grad_Rate'])


# Let's split our data into training data and testing data
trainTest = College_df.randomSplit([0.8, 0.2])
trainingDF = trainTest[0]
testDF = trainTest[1]


# Now create our linear regression model
lir = LinearRegression(featuresCol='Total_Cost',labelCol='Grad_Rate', maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Train the model using our training data
model = lir.fit(trainingDF)


# Generate predictions using the model for all features in the test dataframe:
fullPredictions = model.transform(College_df).cache()

#print("Coefficients: " + str(model.coefficients)) #comment this
#print("Intercept: " + str(model.intercept)) # comment this


# Extract the predictions and the "known" correct labels.
predictions = fullPredictions.select("prediction").rdd.map(lambda x: x[0])
labels = fullPredictions.select("Grad_Rate").rdd.map(lambda x: x[0])

# Zip them together
predictionAndLabel = predictions.zip(labels).collect()

# Print out the predicted and actual values for each point
for prediction in predictionAndLabel:
  print(prediction)  

  


# Stop the session
spark.stop()

# !spark-submit TA3_Cost_Private.py > TA3_Cost_Private.txt
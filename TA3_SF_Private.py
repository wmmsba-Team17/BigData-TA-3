# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 15:16:17 2020

@author: lesli
"""

from __future__ import print_function

#from pyspark import SparkConf, SparkContext
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession, SQLContext
#from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors

# Create a SparkSession (Note, the config section is only for Windows!)
spark = SparkSession.builder.config("spark.sql.warehouse.dir", "file:///C:/temp").appName("LinearRegression").getOrCreate()

inputLines = spark.sparkContext.textFile("file:///Users/lesli/BigData/TA3/College.csv")

def parseLine(line):
    fields = line.split(',')
    SchoolType = fields[1]
    SF_Ratio = float(fields[15])
    Grad_Rate = float(fields[18])
    return (SchoolType, SF_Ratio, Grad_Rate)

parsedLines = inputLines.map(parseLine)

Private = parsedLines.filter(lambda x: "Yes" in x[0]) 

data = Private.map(lambda x: (Vectors.dense(float(x[1])),float(x[2]))).cache()

#data = inputLines.map(lambda x: x.split(",")).map(lambda x: (Vectors.dense(float(x[15])),float(x[18]))).cache()
#[16] is Grad_Rate (Y_variable), [13] is predictor -SF.Ratio 
                                                    

# Convert this RDD to a DataFrame
colNames = ['SF_Ratio','Grad_Rate'] #Y is Grad.rate
df = data.toDF(colNames)

# Let's split our data into training data and testing data
trainTest = df.randomSplit([0.8, 0.2])
trainingDF = trainTest[0]
testDF = trainTest[1]
wholeDF = df

# Now create our linear regression model
lir = LinearRegression(featuresCol='SF_Ratio',labelCol='Grad_Rate', maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Train the model using our training data
model = lir.fit(trainingDF)


# Generate predictions using the model for all features in the test dataframe:
#fullPredictions = model.transform(testDF).cache()
fullPredictions = model.transform(wholeDF).cache()  #use the same dataframe to make prediction 

#print("Coefficients: " + str(model.coefficients)) #comment this
#print("Intercept: " + str(model.intercept)) # comment this


# Extract the predictions and the "known" correct labels.
predictions = fullPredictions.select("prediction").rdd.map(lambda x: x[0])  #what i predicted
labels = fullPredictions.select("Grad_Rate").rdd.map(lambda x: x[0])    #actual 

# Zip them together
predictionAndLabel = predictions.zip(labels).collect()

# Print out the predicted and actual values for each point
for prediction in predictionAndLabel:
  print(prediction)  #print out column grad_rate only  #but a lot few outputs than original dataset

  

# Stop the session
spark.stop()

# !spark-submit TA3_SF_Private.py >TA3_SF_Private.txt
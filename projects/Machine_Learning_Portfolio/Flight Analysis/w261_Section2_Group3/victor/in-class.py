# Databricks notebook source
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['label'] = pd.Series(iris.target) #target classes 0, 1, 2
print(df_iris.head())
#sc = SparkContext().getOrCreate()
# Create SparkSession from builder
#sqlContext = SparkSession.builder.config(“k1”, “v1").getOrCreate() # SQLContext(sc)
data = sqlContext.createDataFrame(df_iris)
print(data.printSchema())
features = iris.feature_names
va = VectorAssembler(inputCols = features, outputCol='features')
va_df = va.transform(data)
va_df = va_df.select(['features', 'label'])
va_df.show(3)

# COMMAND ----------


(train, test) = va_df.randomSplit([0.8, 0.2])
dtc = DecisionTreeClassifier(featuresCol="features", labelCol="label")
dtc = dtc.fit(train)
#predict test
pred = dtc.transform(test)  # pred dataframe
pred.show(3)

# COMMAND ----------

# use the label and prediction from pred dataframe
evaluator=MulticlassClassificationEvaluator(predictionCol="prediction")

acc = evaluator.evaluate(pred)
print("Prediction Accuracy: ", acc)
y_pred=pred.select("prediction").collect()
y_orig=pred.select("label").collect()
cm = confusion_matrix(y_orig, y_pred) #sklearn
print("Confusion Matrix:")
print(cm)

# COMMAND ----------

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from sklearn.metrics import confusion_matrix
import pandas as pd
import pyspark
from pyspark.sql import SparkSession

from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape) #input data

iris = load_iris()
df_digits = pd.DataFrame(digits.data, columns=iris.feature_names)
df_iris['label'] = pd.Series(digits.target) #target classes 0, 1, 2
print(df_iris.head())
#sc = SparkContext().getOrCreate()
# Create SparkSession from builder
#sqlContext = SparkSession.builder.config(“k1”, “v1").getOrCreate() # SQLContext(sc)
data = sqlContext.createDataFrame(df_digits)
print(data.printSchema())
features = digits.feature_names
va = VectorAssembler(inputCols = features, outputCol='features')
va_df = va.transform(data)
va_df = va_df.select(['features', 'label'])
va_df.show(3)

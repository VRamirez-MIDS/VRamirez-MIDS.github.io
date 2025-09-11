# Databricks notebook source
# MAGIC %md 
# MAGIC # Flight Delay Modeling: The Puffins

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Abstract
# MAGIC Flight delays create problems in scheduling for airlines and airports, leading to passenger inconvenience, and huge economic losses. Predicting these delays ahead of time can alleviate some of the issues caused by these delays. Inspired by puffins, birds that are known to be [very punctual](https://www.rspb.org.uk/about-the-rspb/about-us/media-centre/press-releases/rspb-ni-rathlin-punctual-puffins/) in their migratory patterns, our team plans to predict flight departure delays using airport and weather data. Our primary customer for this project are passengers, who we will be able to proactively notify about flight delays 2 hours before the scheduled depature of the flight. We will be predicting whether a flight can be categorized into one of the following four categories. We chose these bins, as this will allow customers to plan their schedules to the airport accordingly.
# MAGIC  - Will depart on-time (less than 15 minutes after the scheduled departure) 
# MAGIC  - Will depart between 15-44 minutes late
# MAGIC  - Will depart between 45-89 minutes late
# MAGIC  - Will depart between greater than 90 minutes late
# MAGIC 
# MAGIC We plan on building models using Logistic Regression and XGBoost to start, though we will be evaluating and assessing other models as we progress through EDA and preliminary pipeline construction. We will report the success of our models using F1-Score, Precision and Recall.
# MAGIC 
# MAGIC \
# MAGIC <img src="https://drive.google.com/uc?id=1VNYZJf7ypNNjC24USv9lIZgItLNC3h_z" alt="Google Drive Image" width=50%/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Team Members
# MAGIC 
# MAGIC Name: Rathin Bector \
# MAGIC Email: rathin.bector@berkeley.edu \
# MAGIC <img src="https://drive.google.com/uc?id=13yvUjEUS6aybbRax_wOIffeIIaD4jMNt" alt="Google Drive Image" width=20%/>
# MAGIC 
# MAGIC Name: Victor Ramirez \
# MAGIC Email: victor.ramirez@berkeley.edu \
# MAGIC <img src="https://drive.google.com/uc?id=1NbNsQiYvLQNZdXi0HPEkd3biCRV_eXZY" alt="Google Drive Image" width=20%/>
# MAGIC 
# MAGIC Name: Francisco Meyo \
# MAGIC Email: francisco@berkeley.edu \
# MAGIC <img src="https://drive.google.com/uc?id=1F7qT1sVXMjM_lh1Qr6MnvFF1ZuwTAtaT" alt="Google Drive Image" width=20%/>
# MAGIC 
# MAGIC Name: Landon Morin \
# MAGIC Email: morinlandon@berkeley.edu \
# MAGIC <img src="https://drive.google.com/uc?id=1KLrkBsnwrWNhakBRSRc_Jr2UpEmHLfFN" alt="Google Drive Image" width=20%/>

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Phase Leader Plan
# MAGIC 
# MAGIC | Phase | Description                                                                                                      | Project Manager |
# MAGIC |-------|------------------------------------------------------------------------------------------------------------------|-----------------|
# MAGIC | I     | Project Plan, describe datasets, joins, tasks, and metrics                                                       | Rathin Bector   |
# MAGIC | II    | EDA, baseline pipeline on all data, Scalability, Efficiency, Distributed/parallel Training, and Scoring Pipeline | Victor Ramirez  |
# MAGIC | III   | Feature engineering and hyperparameter tuning, In-class Presentation                                             | Francisco Meyo  |
# MAGIC | IV    | Select the optimal algorithm, fine-tune and submit a final report (research style)                               | Landon Morin    |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Credit Assignment Plan
# MAGIC | Task Name                                                                                         | Phase   | Assignee                                                    | Due Date   | Status |
# MAGIC |---------------------------------------------------------------------------------------------------|---------|-------------------------------------------------------------|------------|--------|
# MAGIC | Create Phase 1 Notebook                                                                           | Phase 1 | Rathin Bector                                               | 10/24/2022 | DONE   |
# MAGIC | Make Phase Leader Plan                                                                            | Phase 1 | Rathin Bector                                               | 10/26/2022 | DONE   |
# MAGIC | Add Pictures and Emails to Notebook                                                               | Phase 1 | Rathin Bector                                               | 10/26/2022 | DONE   |
# MAGIC | Project Plan Abstract                                                                             | Phase 1 | Rathin Bector                                               | 10/27/2022 | DONE   |
# MAGIC | Credit Assignment Plan                                                                            | Phase 1 | Landon Morin, Rathin Bector, Francisco Meyo, Victor Ramirez | 10/28/2022 | DONE   |
# MAGIC | Description of Data                                                                               | Phase 1 | Victor Ramirez, Francisco Meyo, Landon Morin                | 10/29/2022 | DONE   |
# MAGIC | Basic EDA writeup                                                                                 | Phase 1 | Landon Morin, Francisco Meyo, Victor Ramirez                | 10/29/2022 | DONE   |
# MAGIC | Data Joins Plan                                                                                   | Phase 1 | Rathin Bector, Landon Morin                                 | 10/29/2022 | DONE   |
# MAGIC | ML Algorithms and Metrics                                                                         | Phase 1 | Landon Morin, Rathin Bector                                 | 10/30/2022 | DONE   |
# MAGIC | Machine Learning Pipelines                                                                        | Phase 1 | Victor Ramirez                                              | 10/30/2022 | DONE   |
# MAGIC | Conclusions and Next Steps                                                                        | Phase 1 | Francisco Meyo                                              | 10/30/2022 | DONE   |
# MAGIC | Submit Notebook and PDF                                                                           | Phase 1 | Rathin Bector                                               | 10/30/2022 | DONE   |
# MAGIC | Create Post of Discussion Page                                                                    | Phase 1 | Rathin Bector                                               | 10/30/2022 | DONE   |
# MAGIC | EDA on weather table                                                                              | Phase 2 | Francisco Meyo                                              | 11/01/2022 | DOING  |
# MAGIC | EDA on airline table                                                                              | Phase 2 | Landon Morin                                                | 11/01/2022 | DOING  |
# MAGIC | EDA on stations table                                                                             | Phase 2 | Victor Ramirez                                              | 11/01/2022 | DOING  |
# MAGIC | Conduct Data Joins                                                                                | Phase 2 | Rathin Bector                                               | 11/01/2022 | DOING  |
# MAGIC | Train, Validation, Test Split                                                                     | Phase 2 | Landon Morin                                                | 11/02/2022 | TO DO  |
# MAGIC | Feature Cleanup and Transformations                                                               | Phase 2 | Rathin Bector, Landon Morin, Francisco Meyo, Victor Ramirez | 11/05/2022 | TO DO  |
# MAGIC | PCA for Dimensionality Reduction                                                                  | Phase 2 | Rathin Bector                                               | 11/07/2022 | TO DO  |
# MAGIC | Logistic Regression Baseline Model                                                                | Phase 2 | Francisco Meyo                                              | 11/08/2022 | TO DO  |
# MAGIC | Cross-Validation Scoring Pipeline                                                                 | Phase 2 | Landon Morin                                                | 11/10/2022 | TO DO  |
# MAGIC | 2021 Scoring Pipeline                                                                             | Phase 2 | Victor Ramirez                                              | 11/10/2022 | TO DO  |
# MAGIC | Run Additional Experiments                                                                        | Phase 2 | Rathin Bector                                               | 11/13/2022 | TO DO  |
# MAGIC | Update Phase Project Notebook                                                                     | Phase 2 | Victor Ramirez                                              | 11/13/2022 | TO DO  |
# MAGIC | Create 2min Video Update                                                                          | Phase 2 | Victor Ramirez                                              | 11/13/2022 | TO DO  |
# MAGIC | Submit Notebook and PDF                                                                           | Phase 2 | Victor Ramirez                                              | 11/13/2022 | TO DO  |
# MAGIC | Create Post of Discussion Page                                                                    | Phase 2 | Victor Ramirez                                              | 11/13/2022 | TO DO  |
# MAGIC | Research SMOTE                                                                                    | Phase 4 | Landon Morin                                                | 11/15/2022 | DONE   |
# MAGIC | Feature Engineering                                                                               | Phase 3 | Francisco Meyo, Landon Morin, Victor Ramirez                | 11/16/2022 | TO DO  |
# MAGIC | Create a baseline model                                                                           | Phase 3 | Rathin Bector                                               | 11/17/2022 | TO DO  |
# MAGIC | Conduct test on using new features and report metrics                                             | Phase 3 | Rathin Bector                                               | 11/17/2022 | TO DO  |
# MAGIC | Update leaderboard and write a gap analysis of your best pipeline against the Project Leaderboard | Phase 3 | Landon Morin                                                | 11/18/2022 | TO DO  |
# MAGIC | Fine-tune baseline pipeline using a grid search                                                   | Phase 3 | Victor Ramirez                                              | 11/18/2022 | TO DO  |
# MAGIC | Video update                                                                                      | Phase 3 | Francisco Meyo, Landon Morin, Rathin Bector, Victor Ramirez | 11/20/2022 | TO DO  |
# MAGIC | Slides for presentation                                                                           | Phase 3 | Francisco Meyo, Landon Morin, Rathin Bector, Victor Ramirez | 11/20/2022 | TO DO  |
# MAGIC | Consider other models and build pipelines for these models                                        | Phase 4 | Victor Ramirez, Francisco Meyo, Rathin Bector, Landon Morin | 11/22/2022 | TO DO  |
# MAGIC | Hyperparameter tuning for all models using cross-validation                                       | Phase 4 | Rathin Bector                                               | 11/26/2022 | TO DO  |
# MAGIC | Final feature engineering and refinement                                                          | Phase 4 | Landon Morin                                                | 11/26/2022 | TO DO  |
# MAGIC | Consider and formalize written analysis of exciting, novel directions that we pursued             | Phase 4 | Victor Ramirez                                              | 11/26/2022 | TO DO  |
# MAGIC | Clean up code                                                                                     | Phase 4 | Francisco Meyo                                              | 12/04/2022 | TO DO  |
# MAGIC | Gap analysis of best pipeline against project leaderboard                                         | Phase 4 | Landon Morin                                                | 12/04/2022 | TO DO  |
# MAGIC | Final Writeup                                                                                     | Phase 4 | Landon Morin, Victor Ramirez, Francisco Meyo, Rathin Bector | 12/04/2022 | TO DO  |
# MAGIC | Submission and discussion board post                                                              | Phase 4 | Landon Morin                                                | 12/04/2022 | TO DO  |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Project Plan
# MAGIC 
# MAGIC We are using ClickUp for project planning for this project. The Clickup folder for this project can be found [here](https://sharing.clickup.com/42080451/g/h/184663-160/d7cc34e69aa3512). Below, you can find the Gantt charts we have for each phase of this project.
# MAGIC 
# MAGIC ### Phase 1
# MAGIC <img src="https://drive.google.com/uc?id=1igtHkkr-dC7tVNMZ3uk_Q2kvqATs4KDB" alt="Google Drive Image" width=90%/>
# MAGIC 
# MAGIC ### Phase 2
# MAGIC <img src="https://drive.google.com/uc?id=1iMnTRgQCdYP30K9udHHF7KhbZ6DyDMrg" alt="Google Drive Image" width=90%/>
# MAGIC 
# MAGIC ### Phase 3
# MAGIC <img src="https://drive.google.com/uc?id=1lcFOfhD9nK0X_jTPk6HvpnJH-PHYkNhP" alt="Google Drive Image" width=90%/>
# MAGIC 
# MAGIC ### Phase 4
# MAGIC <img src="https://drive.google.com/uc?id=1_0ZIc9Jt-9N01KCk27O3q1cq0srAAwtL" alt="Google Drive Image" width=90%/>

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Dataset Summary
# MAGIC 
# MAGIC 
# MAGIC **Airlines:**    
# MAGIC 
# MAGIC The Airline dataset contains on-time performance data from the TranStats data collection available from the U.S. Department of Transportation (DOT). The airline flight dataset provides many features about the flight including flight, date, carrier, delay, cancel and diversion information. The data consist of numerical (integers and doubles) and categorical (strings) value types. The flight dataset does contain high null / missing values. Many of the dataset columns are related to flight diverts and delays. As only a low number of flights are delayed or diverted the dataset contains many null values.  
# MAGIC 
# MAGIC We will use the following features to build our model:  
# MAGIC 1.	FL_DATE: Flight date (yyymmdd)  
# MAGIC 2.	DEP_DELAY_GROUP: Difference in minutes between scheduled and actual departure time. Early departures show negative numbers.  
# MAGIC 3.	OP_CARRIER: Flight Carrier  
# MAGIC 4.	ORIGIN: Flight departure origin  
# MAGIC 5.	TAIL_NUM: Flight aircraft tail number  
# MAGIC 6.	DISTANCE: Total flight distance  
# MAGIC 
# MAGIC **Weather:**   
# MAGIC 
# MAGIC The weather dataset contains information from 2015 to 2021. The dataset contains summaries from major airport weather stations that include a daily account of temperature extremes, degree days, precipitation amounts and wind.  
# MAGIC 
# MAGIC We will use the following features to build our model:  
# MAGIC 1.	FL_DATE: Flight date (yyymmdd)  
# MAGIC 2.	DEP_DELAY_GROUP: Difference in minutes between scheduled and actual departure time. Early departures show negative numbers.  
# MAGIC 3.	OP_CARRIER: Flight Carrier  
# MAGIC 4.	ORIGIN: Flight departure origin  
# MAGIC 5.	TAIL_NUM: Flight aircraft tail number  
# MAGIC 6.	DISTANCE: Total flight distance  
# MAGIC 
# MAGIC 
# MAGIC **Station:** 
# MAGIC 
# MAGIC The stations dataset contains information from the US naval stations in the US and territories. The data consist of numerical (doubles) and categorical (string) value types. The dataset is complete with no null / missing values.   
# MAGIC We will use the following features to build our model:  
# MAGIC 1.	station_id: A character string that is a unique identifier for the weather station.  
# MAGIC 2.	neighbor_call: The ICAO identifier for the station.  
# MAGIC 3.	neighbor_lat: Latitude (degrees) rounded to three decimal places of the neighbor location.  
# MAGIC 4.	neighbor_lon: Longitude (degrees) rounded to three decimal places of the neighbor location.  
# MAGIC 5.	distance_to_neighbor: Total distance to the next station. 
# MAGIC 
# MAGIC **Summary description of dataset:**  
# MAGIC 
# MAGIC * Airlines Dataset
# MAGIC   + Size: 2,806,942 Rows (3 month subset)  / 74,177,433 Rows (full dataset)  
# MAGIC   + Source: https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ    
# MAGIC * Weather Dataset  
# MAGIC   + Size: 30,528,602 Rows (3 month subset) /  898,983,399 Rows (full dataset) 
# MAGIC   + Source: https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00679    
# MAGIC * Stations Dataset
# MAGIC   + Size: 5,004,169 Rows (full dataset)      
# MAGIC   + Source: W261 Course provided  
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## TODO:
# MAGIC 
# MAGIC Correlation analysis

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Exploratory Data Analysis
# MAGIC 
# MAGIC ## Statons EDA
# MAGIC 
# MAGIC In reviewing the station data, we discovered the numerical latitude, longitude columns for all the weather stations. We will be using an additional dataset to join the stations dataset with the airlines and weather dataset. 
# MAGIC 
# MAGIC Using the latitude and longitude columns we generated a table with all the required coordinates. Leveraging the data bricks Map (Marker) visualization tool, we plotted all the station coordinates on a satellite layer.
# MAGIC 
# MAGIC **Weather Stations Map Plot**
# MAGIC 
# MAGIC <img src="https://drive.google.com/uc?id=14Nl8mtCTJJOpNMGXZfMdpm7iCnEMdMlU" alt="Google Drive Image" width=90%/>
# MAGIC 
# MAGIC **Stations State Distribution**
# MAGIC 
# MAGIC <img src="https://drive.google.com/uc?id=1TBOvm0CmpLpwPpVZOH3uQEp4KAjVel8k" alt="Google Drive Image" width=90%/>
# MAGIC 
# MAGIC 
# MAGIC ## Airlines EDA
# MAGIC The airlines dataset is characterized by high null values in about half of the columns. For the columns that remain after dropping columns with high nulls, we are left with columns with 2% null values or fewer. In the following table, we observe the percent of null values in the remaining columns. Note that the remainder of our kept features contain no null values.
# MAGIC 
# MAGIC Remaining Columns With Null Values | Remaining Columns With No Nulls
# MAGIC -|-
# MAGIC <img src="https://drive.google.com/uc?id=1KoNxWoZB5Ptl6WtwLog-3F8jB3sQFkXg" alt="Google Drive Image" width=90%/> | <img src="https://drive.google.com/uc?id=1nD0OoIV4bJU9ijWHao7xo386EY1oHqwZ" alt="Google Drive Image" width=90%/>
# MAGIC 
# MAGIC Of these columns, we map the descriptive statistics of each numerical feature below. From this brief analysis we can begin to see skew in features such as flight delay, year (geopolitical and health factors in 2020), flight arrival, flight distance, flight airtime, and diverted flights. Further analysis in phase 2 will provide us with the necessary information to take actions to address imbalances in both numerical and categorical data. Furthermore, differing numerical scales will require normalization of the training and test sets such that gradient descent and loss optimization is possible.
# MAGIC 
# MAGIC <img src="https://drive.google.com/uc?id=1YWXDgy3Rntkgn-IxXexL9r7KnROASFOx" alt="Google Drive Image" width=30%/>
# MAGIC 
# MAGIC 
# MAGIC As previously mentioned, our labels will be taken from DEP_DELAY_GROUP, which has a heavy right skew. This indicates that we will need to use SMOTE to adjust for class imbalances.
# MAGIC 
# MAGIC <img src="https://drive.google.com/uc?id=17xi9M-oImTFDFMdSuIud9wzlhNFzqMUs" alt="Google Drive Image" width=90%/>
# MAGIC 
# MAGIC From preliminary analyses, we find that there are potential patterns between our selected features and labels that will provide predictive power despite class imbalances.. 
# MAGIC 
# MAGIC Relationship of Sampled Airlines With Grouped Departure Delays | Relationship Of Total Distance With Total Flight Delay
# MAGIC -|-
# MAGIC <img src="https://drive.google.com/uc?id=1cGyGUFGyo7CbeDNQJYUmXVd6pPNYMfGd" alt="Google Drive Image" width=90%/> | <img src="https://drive.google.com/uc?id=1EHG5MpWRbQOcSA5ihu4sugSaaS0hGXVj" alt="Google Drive Image" width=90%/>
# MAGIC 
# MAGIC ## Weather EDA
# MAGIC 
# MAGIC Selected features of the weather dataset contain several string fields that will require further analysis for categorization. On the other hand, numeric fields are included in several scales that will make normalization necessary:  
# MAGIC 
# MAGIC 
# MAGIC Summary Statistics P.1 | Summary Statistic P.2
# MAGIC -|-
# MAGIC <img src="https://drive.google.com/uc?id=1BbKOPtB9d_zLrPx2k7mx3u2S0o_AUHsl" alt="Google Drive Image" width=90%/> | <img src="https://drive.google.com/uc?id=1yhMlL92fRLnYr_Ue9TVD4K5cZke_a5Hv" alt="Google Drive Image" width=90%/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Transformations 
# MAGIC 
# MAGIC 1. Do you need any dimensionality reduction? (e.g., LASSO regularization, forward/backward selection, PCA, etc..)
# MAGIC 2. Specify the feature transformations for the pipeline and justify these features given the target (ie, hashing trick, tf-idf, stopword removal, lemmatization, tokenization, etc..)
# MAGIC 3. Other feature engineering efforts, i.e. interaction terms, Breiman's method, etc…)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Joins
# MAGIC 
# MAGIC In order to have features derived from the weather dataset to predict flight, we must join the weather data to the airlines data. This join can be performed in the following steps:
# MAGIC 1. The [Master Coordinate table](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FLL), available on the Beaureau of Transportation Statistics website, contains the latitude an longitude for each airport in the world. We can find the closest weather station in the stations table to each airport using Haversine Distance formula Analysis. A guide on how to do this can be found [here](https://medium.com/analytics-vidhya/finding-nearest-pair-of-latitude-and-longitude-match-using-python-ce50d62af546) 
# MAGIC 2. Left join the airlines table with the table derived from step 1, on `ORIGIN_AIRPORT_SEQ_ID` and `AIRPORT_SEQ_ID` respectively. This will give us the closest weather station to the origin ariport for each flight. 
# MAGIC 3. Left join the the table from step 2 with the weather table, on the `station_id` and `STATION` as well as the `FL_DATE` and `DATE` columns respectively. This gives us the weather at the origin airport the day of each flight.
# MAGIC 
# MAGIC ## TODO:
# MAGIC 
# MAGIC 1. Join tables (full join of all the data) and generate the dataset that will be used for training and evaluation
# MAGIC 2. Joins take 2-3 hours with 10 nodes;
# MAGIC 
# MAGIC     a. Join stations data with flights data
# MAGIC     
# MAGIC     b. Join weather data with flights + Stations data
# MAGIC     
# MAGIC     c. Store on cold blob storage on overwrite mode [Use your Azure free credit  (of $100) for storage only]   
# MAGIC     
# MAGIC 3. EDA on joined dataset that will be used for training and evaluation
# MAGIC 
# MAGIC ## Data Split
# MAGIC Train / Test

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Machine Learning Algorithms and Metrics
# MAGIC 
# MAGIC **TODO:**
# MAGIC 
# MAGIC 1. Review the following material regarding developing machine learning pipelines in Spark:
# MAGIC    
# MAGIC    a. https://pages.databricks.com/rs/094-YMS-629/images/02-Delta%20Lake%20Workshop%20-%20Including%20ML.htmlLinks to an external site.
# MAGIC    
# MAGIC    b. https://spark.apache.org/docs/latest/ml-tuning.htmlLinks to an external site. 
# MAGIC    
# MAGIC 2. Create machine learning baseline pipelines and do experiments on ALL the data (the entire dataset, not just  three/six/12 months)
# MAGIC 
# MAGIC    a. Use 2021 data as a blind test set that is never consulted during training.
# MAGIC    
# MAGIC    b. Report  evaluation metrics in terms of cross-fold validation over the training set (2015-2020)
# MAGIC    
# MAGIC    c. Report  evaluation metrics in terms of the 2021 dataset
# MAGIC    
# MAGIC    d. Create a baseline model using logistic/linear regression, ensemble models
# MAGIC    
# MAGIC    e. Discuss experimental results on cross-fold validation, and on the held-out test dataset (2021).
# MAGIC    
# MAGIC    f. Hint: cross-validation in Time Series is very different from regular cross-validation. Please review the following to get more background on CV for time-based data 
# MAGIC    
# MAGIC    + Cross-validation in Time Series data (very different from regular cross-validation)
# MAGIC       * The method that can be used for cross-validating the time-series model is cross-validation on a rolling basis. Start with a small subset of data for training purposes, forecast for the later data points, and then check the accuracy of the forecasted data points. The same forecasted data points are included as part of the next training dataset and subsequent data points are forecasted.
# MAGIC       
# MAGIC       * For more information see:
# MAGIC         + https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/
# MAGIC         
# MAGIC         + https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4#:~:text=Cross%20Validation%20
# MAGIC 
# MAGIC 
# MAGIC We will be operationalizing the problem of predicting flight delays for consumers using multiclass classification. To accomplish this, we will create a hybrid label from DEP_DELAY_GROUP, which consists of bins of 15 minute delay intervals. Since DEP_DELAY_GROUP is skewed toward lower departure delays, we will consider anything less than 15 minutes delayed as on time, then will bin delays into 30 minute classification intervals until 90 minutes, with all times after 90 minutes merged into one bucket +90. 
# MAGIC 
# MAGIC Our metric for success will be the F1 score, since consumers will both care that we are making accurate delay predictions (precision), but also that we are minimizing false negatives(recall). The F1 score is a harmonic mean of both precision and recall, and therefore considers both metrics in its calculation. 
# MAGIC 
# MAGIC $$Equation one: F1 = 2 * \frac{Precision * Recall}{Precision + Recall}$$
# MAGIC 
# MAGIC To start, we will use a baseline algorithm that predicts that a flight will not be delayed. Since most flights are not delayed, we predict that this will be reasonably accurate, but will have a zero recall, precision, and F1 score. Against this baseline, we will be using three machine learning algorithms. 
# MAGIC 
# MAGIC Model one will be logistic regression with a momentum optimizer to predict the probability of a flight being classified as k-minutes delayed, where k is a bucket that is comprised of 15 minute intervals up to 90 minutes and infinite minutes after 90 minutes. We can represent this problem as the following:
# MAGIC 
# MAGIC $$Model One - Logistic Regression: \hat{y} = argmax f_k(x)$$ Where k represents the following:
# MAGIC 
# MAGIC k | Bucket
# MAGIC - | -
# MAGIC 1 | delay <= 15 min
# MAGIC 2 | 15 min > delay >= 45 min
# MAGIC 3 | 45 min > delay >= 75 min
# MAGIC 4 | 75 min > delay >= 90 min
# MAGIC 5 | 90 min > delay
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC We will use a multiclass Binary Cross Entropy loss function, which can be represented by the following function:
# MAGIC 
# MAGIC $$Model One - BCE Loss: -\sum_{c=1}^My_{o,c}\log(p_{o,c})$$ Where M represents the number of classes, y represents the binary classification of class c, and p represents the probability of the classification of class c. 
# MAGIC 
# MAGIC Model 2 will be XGBoost, also with a Binary Cross Entropy loss function. XGBoost is known to scale well, and to provide more accurate results than a simple random forest. XGBoost improves upon each iteration by considering and weighting underperforming predictions from the previous iteration. This algorithm is better performing than random forests when operating on imbalanced data. This feature will be necessary for a large and imbalanced dataset that consists of mostly on-time flights. The XGBoost loss function can be represented by the following function:
# MAGIC $$Model Two - BCE Loss: -\sum_{c=1}^My_{o,c}\log(p_{o,c})$$

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Machine Learning Pipelines Overview
# MAGIC 
# MAGIC We will following the industry standard machine learning pipeline that is an end-to-end process. A machine learning pipeline is a well-defined set of steps to develop, train, test, and optimize a machine learning algorithm. 
# MAGIC 
# MAGIC Each stage of the machine learning pipeline makes up a specific step in the entire pipeline. The workflow is broken up into modular stages of work. The stages are independent and can be optimized. 
# MAGIC 
# MAGIC The pipeline begins with the ingestion and flow of raw data into the pipeline. Once the data is cleaned, sanitized the dataflow continues to the next stage. 
# MAGIC 
# MAGIC The following stage will handle the feature engineering portion of the pipeline. This stage will handle the process of selecting, manipulating, and transforming the data into features that can be used in machine learning model. 
# MAGIC 
# MAGIC The next stage is the model development, training, testing, and tuning. This stage will use the data split between train and test sets. Once the model is trained and tested the model validation happens based on standard testing metrics and results. 
# MAGIC 
# MAGIC In the final stage the machine learning model is used on new data.
# MAGIC 
# MAGIC <img src="https://drive.google.com/uc?id=1tho8U5UBa20AMBZidTTtJ6hiMkmoRENC" alt="Google Drive Image" width=90%/>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Machine Learning Models
# MAGIC 
# MAGIC <img src="https://drive.google.com/uc?id=1xcM2gvYZa17nslSxD0i2Qh038DlfoGLj" alt="Google Drive Image" width=90%/>
# MAGIC 
# MAGIC 
# MAGIC    
# MAGIC   
# MAGIC   

# COMMAND ----------

# ML Pipeline

# importing packages
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# COMMAND ----------

# import data

# COMMAND ----------

# train /test data split

# COMMAND ----------

# create pipelines for Logistic regression, XGBoost
# pipeline will include
# 1. Data import / ingest
# 2. Data preprocessing 
# 3. Data joining
# 4. Train model - loop both pipelines
# 5. Test model
# 6. Deploy model 
# 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Model One: Logistic Regression Pipeline Details 
# MAGIC 
# MAGIC **TODO:**
# MAGIC 
# MAGIC 1. Data Ingestion  
# MAGIC    i.  Gather data from various sources     
# MAGIC    ii. Combine and join all data sources     
# MAGIC 2. Data Preperation  
# MAGIC    i.  Feature engineering     
# MAGIC    ii. Split train / test data  
# MAGIC 4. Model Development  
# MAGIC    i.   Train model with training data     
# MAGIC    ii.  Model tunning /optimazation     
# MAGIC    iii. Paramenter / feature selection     
# MAGIC    iV.  Model selection validation     
# MAGIC 5. Model Deployment  
# MAGIC    i.   New data predictions     
# MAGIC 6. Monitor Model  
# MAGIC    i.   Model feedback  
# MAGIC 
# MAGIC ## Model Two: XBoost Pipeline Details 
# MAGIC 
# MAGIC **TODO:**
# MAGIC 
# MAGIC 1. Data Ingestion  
# MAGIC    i.  Gather data from various sources     
# MAGIC    ii. Combine and join all data sources     
# MAGIC 2. Data Preperation  
# MAGIC    i.  Feature engineering     
# MAGIC    ii. Split train / test data  
# MAGIC 4. Model Development  
# MAGIC    i.   Train model with training data     
# MAGIC    ii.  Model tunning /optimazation     
# MAGIC    iii. Paramenter / feature selection     
# MAGIC    iV.  Model selection validation     
# MAGIC 5. Model Deployment  
# MAGIC    i.   New data predictions     
# MAGIC 6. Monitor Model  
# MAGIC    i.   Model feedback  
# MAGIC    

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion and Next Steps
# MAGIC 
# MAGIC The work conducted during Phase I allowed us to: 
# MAGIC 
# MAGIC 1. Obtain clean datasets (eg performing simple transformations or disregarding features with a high content of missing values)  
# MAGIC 2. Selec relevant features of each dataset after performing basic EDA  
# MAGIC 3. Join datasets to enhance the features available for our model  
# MAGIC 4. Define preliminary models and performance metrics  
# MAGIC 
# MAGIC 
# MAGIC During Phase II we will be conducting the following next steps:  
# MAGIC 1. Perform a detailed EDA that will allow us to define any normalization needed as well as parameter fine tunning   
# MAGIC 2. Determine and calculate our baseline model (most likely Logistic regression)  
# MAGIC 3. Perform experiments   
# MAGIC 4. Adjust our plan as needed
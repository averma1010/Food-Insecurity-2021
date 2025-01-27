An Analysis of Food Security in USA - 2021
==========================================

Hi, this is the final project I did for my Intro to Data Science class at GW. We are trying to Classify Food Security in the United States. This is a typical classification problem with an imbalanced response class.

We are using the CPS Data provided by the US Census Bureau. Since I was constrained by time, I couldn't do EDA on the dataset to an satisfactory level and had to select factors that made an intuitive sense. This lead to some problems in the classification as you'll see.

Hopefully, I'll continue to work on this project and keep updating it on here.

* * * * *

* * * * *

#### The models/techniques that we have used here are:

 Logistic Regression

KNN Algorithm

Decision Tree

* * * * *


* * * * *

Dataset
=======

We have obtained the data from the US Census website using CPS data. The CSV file contains approximately a hundred thousand observations.

The link to our dataset is: <https://www2.census.gov/programs-surveys/cps/datasets/2021/supp/dec21pub.csv>

* * * * *

Logistic Regression
===================

A sort of generalized linear model (GLM) called logistic regression is used in statistical analysis to forecast a binary event (i.e., dependent variable) based on predetermined variables. It estimates the likelihood that an event will occur by fitting data to a logit function.

In this model, the binary outcome is whether the unit of observation (individuals within housing units) is Food Secure or not. The variables in the model will try to predict the binary outcome. Since the data set used in the model is unbalanced, we'll use try adding weights to the minority class and try under sampling the majority class to see whether that leads to an improvement in results.


* * * * *

The model is predicting on the testing data, the cutoff is at 0.5



| Prediction     | Food Insecure | Food Secure     |
| :---        |    :----:   |          ---: |
|  Food Insecure       |  145        |  144   |
|  Food Secure    | 2663     | 25684    |



![](https://lh6.googleusercontent.com/bNDhvdLOMry3dMb6bGATQoa2mSXN57HhUYbibyE0uwS4yW_DHEHh0aIRDzn4Z-FjzCY2Kh60unS02LWEFAl19BDfjyD1lRt-1t-dy8N6DCfm6vQAPEE_x2H8vDHx1EbOeJBeZHqg5hP0fXTmynO5GPgyfrBLHp2Ky8f_0VzZUuOUCyneb1Q3qNto-IwFkw)

* * * * *

We decided that recall is important to us as we want to get our false negative lower as possible. False positives are inclusion errors, while false negatives are exclusion errors. It's better if we are able to more accurately identify who all are food insecure and need of aid rather than identifying who all are not food insecure and hence don't need aid.

* * * * *

Lets look at the probability distribution of the our two response classes in our training dataset. We'll select a new cutoff from this


![](https://lh5.googleusercontent.com/Lplobsfn5FmWpQdxiXTw7s1VeHQwPi8DAWU4t0CkrDLj4YkG7VpRERITf9oCiL59adqBZJcyceyl2fbAR09J7dh-ZF_puhxp2aJqwvadGQElLXFfmCnAuDXXz9UGss3pljdWQjYFBTXm4O3yyAkIAM8TgEpmrru9h2BwE8JBi49Ctg4lx6Fa8b-kEXXLtQ)

We can see that the probability distribution is heavily right skewed. In our next attempt we will select the cutoff at 0.09.

* * * * *



![](https://lh6.googleusercontent.com/po8fP6oU9NMUjYAncdvy4lWPuGw2s1C4ZsWxTrDs2FPIEMHaEFFqTWTubr4hR4PQ6_Uwp3Y0YKYx_GSyj9n8UdJO4dOMdBm96IxEcVo-Ya2kWA1XS0NWH58XI11UJVa3AGzUtiPX3UKnAP0Hx3FIgfqxQnEfFem954tfRefkBWO-hc0-wlcb4VqzUzwzag)



## Area under the curve: 0.8698

The area under curve is 0.87. Even though this seems good, but since our classes are imbalanced we can't put much significance to this metric in comparing different models.

* * * * *

Now we change the cutoff to 0.09


| Prediction     | Food Insecure | Food Secure     |
| :---        |    :----:   |          ---: |
|  Food Insecure       |  2646         |  7579   |
|  Food Secure    | 162    | 18249    |



![](https://lh6.googleusercontent.com/DjpRwcaTHZdvn5dAjzi44MAxYwvZ3XGYa64TJn0dnrfClgOJ_iqvjVMtrGySgrI1wg1SBuv0MF_uLJHmpEO9BHklkP3uZXL-mqmZfgFJwlknZiTuEV8L5ZY-0L8ukFj3oMpxjJMfaZFjz7zqOf5r5qSe71GBLiWnpUiBeI-ZqcsbQmnNyHN3BSr0PeCRKg)

As you can see our recall has increased to 94.2%, while our precision has come down to 25.9%. 

We think this is better than our previous model since the previous model had just a recall of 5%.

* * * * *

The ration of Food Secure to Food Insecure in our data is 9.03. We are gonna add a weight equivalent to that to our minority class.

* * * * *

-   We are going to be adding weights to the data so our minority class, which is food insecurity, affects the model more.


| Prediction     | Food Insecure | Food Secure     |
| :---        |    :----:   |          ---: |
|  Food Insecure       |  2645        |  7540   |
|  Food Secure    | 163    | 18288    |



![](https://lh5.googleusercontent.com/rjb553OAb3B1TxttbQTlLwPv4QogxgSxgZp-i_ewPkhU1oCNZe0IyiJzQKRJLDzGClEGLg4SHVsgLAfA9RuLUPjnWltbEliqeQcRFiYx3lavm3t0tDdK4AHSWNfnFSDOzh18ohb0C5p1D8Et6h0ZXJdeeaviptRDVFFSgpi2GbV6IggfHOAu8yHxRbenhw)

-   As you can see even though our classification rate of the model went down and the precision of model went down drastically, we were able to raise our recall rate significantly.

-   But this result is almost the same as the one we got from the model where we changed the cutoff to 0.09.

* * * * *



![](https://lh5.googleusercontent.com/4qxz3lImXkLPBMZw96g5lREaUnwk-b4TSHbNBgV2-j6xDHucYbP7DhRuS8sxmUD023QNeDNLd1pW8D2_B4MMNxNGnmwXUAwBmN-nVRxTRnj7pdl9kkPkjSOoN8eIIR6cQ0xkWvfaJnPUwT2IWWPFWcGiKXrxoB79FzKTKtRSbbjOSvY7eZ-bZ2LITfNCpg)



## Area under the curve: 0.8699

-   The response probability distribution of this model looks very different from the previous models. We can clearly see two response classes are clearly separated for most part.

-   The little overlap that is visible may be due to the fact that our response originally had 4 classes but we had converted it into binary.

* * * * *



![](https://lh4.googleusercontent.com/BD6t2jjQMreRqwG5YiwvhamsYL-8OvTlLza-vMZPWPJ2AX4yVdbDfOMAZFxCKh0zhyfhavz7pZiMYCeGyhSlxn7OVTH3A2ZlTzM799jeVIaEeU2ITD7zNwQPLj2JxLANDZaY5r7HvEYxn_7v6SkuNF11tFhNHUcLKkYI4Zdhko5CqFDcIT6lE33nzSqxAQ)

* * * * *

-   Trying to balance data by under sampling. We are using the ROSE library to do this

| Prediction     | Food Insecure | Food Secure     |
| :---        |    :----:   |          ---: |
|  Food Insecure       |  2644          |  7633   |
|  Food Secure    | 164    | 18195    |


![](https://lh5.googleusercontent.com/SCGoM3yIKJ5sWupz8ymCv20Y9BeGIboTrCcU2CE3P1vyVPLImg9dVh6Gcs--W7uk98lu5V-ciLz97GIdx2qxlCSRMzAujmK39kFP055KgaW07eKVGX4Qf6KlQW7pxKdAEBBpW5WUgHeYPNWIBdp2G2yldmehaTfgOoKAbeFkNs1Bx1M9m6wn8vMdqMw-gA) 

* As you can see the results are almost the same as the previous models.



![](https://lh6.googleusercontent.com/SDQWI9jiB-FCUYpa34HQ7QRv4dAhioZDuYyMjlrHD0wvUgRIRSYO68Got6UdhkNJ4Uop8TcvB7vYlBPtEHGof_P_z0W4Fwqh-ezHDvNVqt-y4-zm1YrGQM2Y_rAOeGLrrndfNA3ryrTEH-25nruVOAZE12KzZ_3g62LTdFBF0271tCeeghyYriJXrzFnGQ)

-   The probability distribution also looks the same as the previous model.

* * * * *

K-Nearest Neighbors
===================

K-Nearest Neighbor is a Supervised ML Algorithm used to solve both Classification and Regression models. K-NN algorithm assumes the similarity between the new case /data and available cases and put the new case into the category that is most similar to the available categories.It is also called a lazy learner algorithm because it does not learn from the training set immediately instead it stores the dataset and at the time of classification, it performs an action on the dataset. KNN models shows high accuracy for classification problems.

##### Preparing Data for KNN

The preparation of Data for KNN starts with selecting variables. As KNN only accepts numeric variables in or it needs to be transformed to numeric. Since only ordinal variables can be transformed to numeric, we eliminate nominal variables and hence, the list of variables used for KNN model are - Family Size, Household Income, SNAP, Number of jobs, Hours of Job, Number of Children. The predicting variable FS_Status remains binary but numeric.

## 'data.frame':    71442 obs. of  7 variables:

##  $ Family_Size       : num  1 2 2 2 2 2 2 2 2 1 ...

##  $ Household_Income  : num  16 14 14 12 12 13 13 9 9 11 ...

##  $ SNAP              : num  3 3 3 5 5 3 3 5 5 3 ...

##  $ Number_of_Jobs    : num  1 1 1 1 1 1 1 1 1 1 ...

##  $ Hours_on_Jobs     : num  67 43 2 43 43 62 43 2 2 2 ...

##  $ Number_of_children: num  1 2 1 1 1 1 1 1 1 1 ...

##  $ FS_Status         : num  0 0 0 0 0 0 0 0 0 0 ...

##### Balanced Data for KNN

The data is unbalanced as we mentioned above. The number of food secure units of observations are very few when compared to food secured. This issue is resolved in Logstic Regression by adding weights as a hypervariable. But for KNN, sampling methods needs to be used to resolve this isse. The two options are Under-sampling and Up-sampling. Even though, new methods are coming out like creating artificial data points, the accuracy of the model through such model is still debatable.

This does not hide the fact that under-sampling can lead to data lose as we are discarding a lot of observation units and up-sampling will cause over-fitting in the model. Trying both techniques and finding the best model is a considerable solution but it depends on the resources and time.

For the KNN model discussed below, we choose for Under-sampling. The results based on both sampling technique in data are discussed here. Now the data is balanced and we continue with data pre-processing.

## 

##     0     1 

## 35621 35821

## 'data.frame':    71442 obs. of  7 variables:

##  $ Family_Size       : num  1 3 2 2 2 5 6 4 6 7 ...

##  $ Household_Income  : num  7 9 6 14 16 16 16 12 15 14 ...

##  $ SNAP              : num  5 4 5 3 3 3 3 3 3 5 ...

##  $ Number_of_Jobs    : num  1 1 1 1 1 1 1 1 1 1 ...

##  $ Hours_on_Jobs     : num  1 2 2 2 43 18 2 53 2 43 ...

##  $ Number_of_children: num  1 1 1 1 1 1 1 1 1 6 ...

##  $ FS_Status         : num  0 0 0 0 0 0 0 0 0 0 ...

For preparing the data for KNN, we selected numerical variables. Scaling the data is the next step. The algorithm should not be biased towards variables with higher magnitude. To overcome this problem, bring down all the variables to the same scale. Feature Scaling is inevitable for KNN. If not scaled, the feature with large value range will dominate while calculating distances. KNN which uses Euclidean distance needs scaling when the features has a broad range of values.

We also need to create test and train data sets, we will do this slightly differently by using the sample function. The 2 says create 2 data sets essentially, replacement means we can reset the random sampling across each vector and the probability gives sample the weight of the splits, 2/3 for train, 1/3 for test. So here 67% of the sample is taken as training data and 33% as testing data.

We then just need to use the new variable to create the test/train outputs, selecting the first four rows as they are the numeric data in the iris data set and we want to predict Species. And then we need to create our Y variables or labels need to input into the KNN function.

##### Building KNN Model

# create an empty dataframe to store the results from confusion matrices

ResultDf = data.frame( k=numeric(0), Total.Accuracy= numeric(0), row.names = NULL )

|

k

 |

Total.Accuracy

 |
|

3

 |

0.8149

 |
|

5

 |

0.8279

 |
|

7

 |

0.8293

 |
|

9

 |

0.8327

 |
|

11

 |

0.8342

 |
|

13

 |

0.8353

 |
|

15

 |

0.8359

 |
|

15

 |

0.8359

 |

![](https://lh3.googleusercontent.com/aT8aZIaZMkmGbP3SM1I2hnvMM3nXF2RcDCjNlCVUrcIcj6zZSWjRFE9PpKVsw3yUTQqwcg3Pobyx9ryDyPclUJqZCu0mcrosm3PirpJNHAle-k1q9jWfAjIbVXNINmI0BofFng96HZo-PsnETPoJgCfQVKNRRiLPF4BFY8Rl9Zf0M7iEhaVzBKNoR7Fm3g)

From the above table and graph, it is quite clear that k=15 KNN model gives the most accuracy. Since the predicted variable is binary, we take into consideration k values which are odd.

From the above results, the least accuracy is when k = 3, which is 81.49% and the maximum is for when k = 15 and it goes high up to 83.59%. Now we use confusion matrix of KNN model when k=15, to derive more information.

## 

##  

##    Cell Contents

## |-------------------------|

## |                       N |

## |           N / Row Total |

## |           N / Col Total |

## |         N / Table Total |

## |-------------------------|

## 

##  

## Total Observations in Table:  23420 

## 

##  

##                | knn_pred 

## knn.testLabels |         0 |         1 | Row Total | 

## ---------------|-----------|-----------|-----------|

##              0 |      8658 |      3043 |     11701 | 

##                |     0.740 |     0.260 |     0.500 | 

##                |     0.915 |     0.218 |           | 

##                |     0.370 |     0.130 |           | 

## ---------------|-----------|-----------|-----------|

##              1 |       800 |     10919 |     11719 | 

##                |     0.068 |     0.932 |     0.500 | 

##                |     0.085 |     0.782 |           | 

##                |     0.034 |     0.466 |           | 

## ---------------|-----------|-----------|-----------|

##   Column Total |      9458 |     13962 |     23420 | 

##                |     0.404 |     0.596 |           | 

## ---------------|-----------|-----------|-----------|

## 

##

Total Number of Observations = 23430. True Negative = 9145 :- 39% of the observations were correctly predicted as NOT Food Insecure True Positive = 9912 :- 42% are correctly identified as Food Insecure. There were 1807 cases of False Negatives (FN). The FN's poses a potential threat for the reason that model predicts 8% of the total observations as Food Secure, but was actually food insecure and the main focus to increase the accuracy of the model is to reduce FN's. There were 2556 cases of False Positives (FP) meaning 11% were actually food secure in nature but got predicted as food insecure. The total accuracy of the model is 83.59 % ((TN+TP)/Total) which shows that the model is good. But still there is scope of improvement.

xkabledply(data.frame(cm$byClass), title=paste("k = ",15))

|\
 |

cm.byClass

 |
|

Sensitivity

 |

0.7399

 |
|

Specificity

 |

0.9317

 |
|

Pos Pred Value

 |

0.9154

 |
|

Neg Pred Value

 |

0.7821

 |
|

Precision

 |

0.9154

 |
|

Recall

 |

0.7399

 |
|

F1

 |

0.8184

 |
|

Prevalence

 |

0.4996

 |
|

Detection Rate

 |

0.3697

 |
|

Detection Prevalence

 |

0.4038

 |
|

Balanced Accuracy

 |

0.8358

 |

##### KNN Model: Conclusion

Sensitivity is designate a food insecure individual as Food Insecure.A highly sensitive test means that there are few false negative results, and thus fewer cases of Food Insecurity are missed. For this model the Sensitivity is 73.9936758%

Specificity is designate a food secure individual as negative. A high specificity makes sure that, the projects for eradicating Food Insecurity focuses on the right population. Specificity for this model is 93.173479%

Trees
=====

#Using Decision Tree

tree1 <- rpart(FS_Status ~ Ethnicity + Family_Size + Household_Income + SNAP + Citizenship_status + Number_of_Jobs + Education_Level, data=data.balanced.ou, method="class")

printcp(tree1) # display the results

## 

## Classification tree:

## rpart(formula = FS_Status ~ Ethnicity + Family_Size + Household_Income + 

##     SNAP + Citizenship_status + Number_of_Jobs + Education_Level, 

##     data = data.balanced.ou, method = "class")

## 

## Variables actually used in tree construction:

## [1] SNAP

## 

## Root node error: 4266/8577 = 0.49738

## 

## n= 8577 

## 

##        CP nsplit rel error  xerror      xstd

## 1 0.68026      0   1.00000 1.03047 0.0108513

## 2 0.01000      1   0.31974 0.31974 0.0079392

plotcp(tree1) # visualize cross-validation results

# plot tree 

plotcp(tree1)

![](https://lh3.googleusercontent.com/LVkcBUxJ6OO-gmcjTBvapi2rZIyo_pEwFJ-EPRUPRC7XhijZtND1JUYDiHts-jsJTR6Phh9xrw3Y3cQ3B0J2CstFgdfvqZmpA3b7kL9MYjxdcEF9FmRdF9Gf0QFzzczi4g3A17R7Vfg9_pptQ-ZIatUHaP0pp4hKYIsXAIRN4naie1H2frHYd5WTZRUiSw)

#text(tree1, use.n=TRUE, all=TRUE, cex=.8)

rpart.plot(tree1)

![](https://lh5.googleusercontent.com/OGf-KMNAcUrSF2Y8RduOjmHkWoQBU_F85-1mzAmdBFStMLBCaOE1Qd86jYzBY-rQUvhaYGK7GtaNemtWfOdfDlSxRhOzax37GLIXhwqUUMTCk221F2xSQST5MquspTJsVnObT2qz1iVddZLCkncfeXFjqSghDLDyzMZV46W_gCpopvZuffyDidS2sdyUlw)

predict_unseen <-predict(tree1, test, type = 'class')

table_mat <- table(predict_unseen, test$FS_Status)

table_mat[2:1,2:1]

##                

## predict_unseen  Food Insecure Food Secure

##   Food Insecure          2655        7650

##   Food Secure             153       18178

accuracy_tune <- function(tree1) {

predict_unseen <- predict(tree1, test, type = 'class')

    table_mat <- table(test$FS_Status, predict_unseen)

    accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)

    accuracy_Test

}

control <- rpart.control(minsplit = 4,

    minbucket = round(5 / 3),

maxdepth = 3,

cp = 0.0001)

tune_fit <- rpart(FS_Status ~ Ethnicity + Family_Size + Household_Income + SNAP + Citizenship_status + Number_of_Jobs + Education_Level , data = data.balanced.ou, method = 'class', control = control)

accuracy_tune(tune_fit)

## [1] 0.7241584

tree2 <- rpart(FS_Status ~ Ethnicity + Family_Size + Household_Income + SNAP + Citizenship_status + Number_of_Jobs + Education_Level, data=data.balanced.ou, method="class", control = control)

printcp(tree2) # display the results

## 

## Classification tree:

## rpart(formula = FS_Status ~ Ethnicity + Family_Size + Household_Income + 

##     SNAP + Citizenship_status + Number_of_Jobs + Education_Level, 

##     data = data.balanced.ou, method = "class", control = control)

## 

## Variables actually used in tree construction:

## [1] Family_Size      Household_Income SNAP            

## 

## Root node error: 4266/8577 = 0.49738

## 

## n= 8577 

## 

##           CP nsplit rel error  xerror      xstd

## 1 0.68026254      0   1.00000 1.00000 0.0108545

## 2 0.00093765      1   0.31974 0.31974 0.0079392

## 3 0.00023441      3   0.31786 0.31997 0.0079416

## 4 0.00010000      5   0.31739 0.31997 0.0079416

plotcp(tree2) # visualize cross-validation results

# plot tree 

plotcp(tree2)

![](https://lh5.googleusercontent.com/EuPWeIdEKRkj11aMUDK9DgICZo_jSP52XkLjwH9aJlk5GA94nPPxIhdhfj-vyMH5tgOcPSjf8MZRLeRPsJQLahfzATCcDnWk6qvToqm4lBkCMUzlS8jrYivYY3ZswUbo6Gm9mB2yLSHPNaM87VyuVKV5_B_sppX1xe6UsF7YzDsIelUD3EcvaqvSVw3stQ)

#text(tree1, use.n=TRUE, all=TRUE, cex=.8)

rpart.plot(tree2)

![](https://lh5.googleusercontent.com/H1yP3P6jfsbx2t_eM0cvaoLbppJu89qtVFNWtdBZhXfk8bblwuJuKmv0wXm-PEh-LmKmM3xs9KRrV1E31F7qH-xG7-jvl3Q9xdJitHT9WtswThBORvp63l1KDrCyEzIkh3BWQU-Y8pro0rI5CpaM8ztblsd1WJRm_LH3-NR5YmXoC9oyop2t5DRV2RUU-Q)

predict_unseen <-predict(tree2, test, type = 'class')

table_mat <- table(predict_unseen, test$FS_Status)

table_mat[2:1,2:1]

##                

## predict_unseen  Food Insecure Food Secure

##   Food Insecure          2661        7752

##   Food Secure             147       18076

Conclusions and further recommendations
=======================================

SNAP is the most important factor in food security in our models. In future, We could remove SNAP or take SNAP longitudinally to make some analysis; as the correlation between SNAP and food insecurity is almost tautological - Only a food insecure person would have SNAP and Food secure person won't have SNAP.

We saw that KNN performs better than other models for these factors.



# References

•	https://www2.census.gov/programs-surveys/cps/datasets/2021/supp/dec21pub.csv
•	https://www.census.gov/data/datasets/time-series/demo/cps/cps-supp_cps-repwgt/cps-food-security.html
•	https://www.ers.usda.gov/data-products/food-security-in-the-united-states/
•	https://r4ds.had.co.nz/exploratory-data-analysis.html
•	https://www.webpages.uidaho.edu/~stevel/519/How%20to%20get%20correlation%20between%20two%20categorical%20variable%20and%20a%20categorical%20variable%20and%20continuous%20variable.html
•	http://statseducation.com/Introduction-to-R/modules/graphics/cont/
•	https://www.tutorialspoint.com/r/r_mean_median_mode.htm


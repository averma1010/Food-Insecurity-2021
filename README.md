---
title: "An Analysis of Food Security in USA - 2021"
author: "Team - 02: Akshay Verma, Aveline Mariya Shaji and Uugangerel Bold"
date: "`r Sys.Date()`"
output:
  html_document:
    css: bootstrap.css
    code_folding: hide
    number_sections: no
    toc: yes
    toc_depth: 3
    toc_float: yes
  pdf_document:
    toc: yes
    toc_depth: '3'
  word_document:
    toc: yes
    toc_depth: '3'
editor_options: 
  markdown: 
    wrap: 72
---

***

# Precap

This paper is an extension to the Exploratory Data Analysis of the **Analysis of Food Security in USA - 2021**. From the previous analysis, we brought down to 7 variables such as Family_Size, Household_Income, SNAP, Ethnicity, Citizenship_status, Hours_on_Jobs, Education_Level which has an effect on the FoodSecurity_score variable. The extended part will deal different classification techniques/models to predict the food insecurity.

***

#### The models/techniques that we have used here are:

 -- Logistic Regression
 
 -- KNN Algorithm
 
 -- Decision Tree

***

#### Our SMART questions are below: 

--	Specific: To extend the EDA on the Census data to find out what combinations of socio-economic factors lead to food insecurity. 

--	Measurable: Measure the risk of food insecurity at different combinations of selected factors.

--	Achievable: Make a prediction model using logistic regression for the food insecurity. Extend the modelling using KNN and Random Forest.

--	Relevant: Food being the basic requirement of any human, this study can shed light on what the authorities and we ourselves can do in order to eradicate food insecurity.

--	Time-Oriented: Data set for the month of December 2021 is considered for the study and the study is expected to come-up with interesting results by early December 2022.

***

```{r, echo=FALSE, results='hide', message=FALSE, warning=FALSE}
library(dplyr)
library(ezids)
library(ggplot2)
library(epiR)
library(pROC)
library(smotefamily)
library(ROSE)
library(ggthemes)
library(caret)
library(car)
library(ROSE)

```

***

# Dataset

We have obtained the data from the US Census website using CPS data. The CSV file contains approximately a hundred thousand observations. 

The link to our dataset is: 
https://www2.census.gov/programs-surveys/cps/datasets/2021/supp/dec21pub.csv

***

```{r, echo=FALSE, cache=TRUE }
Food_Sec <- data.frame(read.csv("dec21pub.csv"))
```

```{r, echo=FALSE, results='hide', message=FALSE}


FS_Subset <- subset(Food_Sec, HRINTSTA == 001 & HRSUPINT == 001 & HRFS12MD != -9)
FS_Subset <- subset(FS_Subset, select = c(	"GESTFIPS",	"HRNUMHOU",	"HEFAMINC",	"HESP1",	"PTDTRACE",	"PRCITSHP",	"PEMJNUM",	"PEHRUSL1",	"PEEDUCA", "PRNMCHLD" , "HRFS12MD"))

  

FS_Subset <- FS_Subset %>% rename("States" = "GESTFIPS", "Family_Size" = "HRNUMHOU",	"Household_Income" = "HEFAMINC",	"SNAP" = "HESP1",	"Ethnicity" =	"PTDTRACE", "Citizenship_status" = "PRCITSHP",	"Number_of_Jobs" = "PEMJNUM",	"Hours_on_Jobs" = "PEHRUSL1" , "Education_Level" = "PEEDUCA" , "Number_of_children" = "PRNMCHLD",  "FoodSecurity_score" = "HRFS12MD")
```



```{r, echo=FALSE, results='hide', message=FALSE}
## Converting the all the columns to factors as they are all ordinal(except the Id, but since it's categorical i'm converting it into a factor too)


FS_Subset[] <- lapply( FS_Subset, factor)

str(FS_Subset)

```



```{r, echo=FALSE, results='hide', message=FALSE}
levels(FS_Subset$'FoodSecurity_score') <- c( "High Food Security", "Marginal Food Security", "Low Food Security", "Very Low Food Security")


FS_Subset$FS_Status <- FS_Subset$FoodSecurity_score

levels(FS_Subset$FS_Status) <- c( "Food Secure", "Food Secure", "Food Insecure", "Food Insecure")

levels(FS_Subset$'Ethnicity') <- c('White only', 'Black only', 'American Indian, Alaskan native only', 'Asian Only', 'Hawaiian', 'White-black', 'White-AI', 'White-Asian', 'White-HP', 'Black-AI', 'Black-Asian', 'Black-HP', 'AI-Asian', 'AI-HP', 'Asian-HP', 'W-B-AI', 'W-B-A', 'W-B-HP', 'W-AI-A', 'W-AI-HP', 'W-A-HP', 'B-AI-A', 'W-B-AL-A', 'W-AI-A-HP', 'Other 3 race combo', 'Other 4 and 5 race combo')

levels(FS_Subset$'Citizenship_status') <- c('NATIVE, BORN IN THE UNITED STATES', 'NATIVE, BORN IN PUERTO RICO OR OTHER U.S. ISLAND AREAS', 'NATIVE, BORN ABROAD OF AMERICAN PARENT OR PARENTS', 'FOREIGN BORN, U.S. CITIZEN BY NATURALIZATION', 'FOREIGN BORN, NOT A CITIZEN OF THE UNITED STATES')
summary(FS_Subset$'Citizenship_status', title = "PRCITSHP")


```

```{r, echo=FALSE, results='hide', message=FALSE}
table(FS_Subset$Ethnicity)

## Lets drop all levels that have less than 10 observations.

for(i in levels(FS_Subset$Ethnicity)) {
  if(count(subset(FS_Subset, Ethnicity == i)) < 10){
    print(i)
    FS_Subset <- FS_Subset[!(FS_Subset$Ethnicity %in% c(i)), ]
  }}

FS_Subset$Ethnicity <- droplevels(FS_Subset$Ethnicity)
     
table(FS_Subset$Ethnicity)
```



# Logistic Regression

A sort of generalized linear model (GLM) called logistic regression is used in statistical analysis to forecast a binary event (i.e., dependent variable) based on predetermined variables. It estimates the likelihood that an event will occur by fitting data to a logit function.

In this model, the binary outcome is whether the unit of observation (individuals within housing units) is Food Secure or not. The variables in the model will try to predict the binary outcome. Since the data set used in the model is unbalanced, we'll use try adding weights to the minority class and try under sampling the majority class to see whether that leads to an improvement in results.

```{r, results ='hide'}
set.seed(1)

sample <- sample(c(TRUE, FALSE), nrow(FS_Subset), replace = TRUE, prob = c(0.6, 0.4))

train <- FS_Subset[sample, ]
test  <- FS_Subset[!sample, ]

str(test)
str(train)
```



```{r, echo=FALSE, results='hide', message=FALSE}
logistic <- glm(FS_Status ~ Ethnicity +  Family_Size + Household_Income:SNAP + Citizenship_status + Number_of_Jobs + Education_Level   , data = train, family = "binomial")
```

***

The model is predicting on the testing data, the cutoff is at 0.5

```{r, results='markup'}

logistic_model1.prob <- predict(logistic, test, type = "response")
logistic_model1.pred = rep("Food Secure", dim(test)[1])
logistic_model1.pred[logistic_model1.prob > .5] = "Food Insecure"


tb <- table(logistic_model1.pred, test$FS_Status)
tb[1:2,2:1]


precision <- round(precision(tb[1:2,2:1])*100,2)
recall <- round(recall(tb[1:2,2:1])*100,1)


lg_model_graph <- data.frame(precision , recall )
p <- barplot(as.matrix(lg_model_graph),beside=TRUE, col=rgb(0.2,0.3,0.6,0.6), space = c(0.1, 0.1))
text(x = p, y = lg_model_graph -  2, labels = lg_model_graph)


```

*** 

We decided that recall is important to us as we want to get our false negative lower as possible. False positives are inclusion errors, while false negatives are exclusion errors. It's better if we are able to more accurately identify who all are food insecure and need of aid rather than identifying who all are not food insecure and hence don't need aid.

***

Lets look at the probability distribution of the our two response classes in our training dataset. We'll select a new cutoff from this

```{r, results='markup'}
train$logistic_model1_train.prob <- predict(logistic, train, type = "response")

# distribution of the prediction score grouped by known outcome
ggplot( train, aes( logistic_model1_train.prob, color = as.factor(FS_Status) ) ) + 
geom_density( size = 1 ) +
ggtitle( "Training Set's Predicted Score" ) + 
scale_color_economist( name = "data", labels = c( "Food Secure", "Food Insecure" ) ) + 
theme_economist()

```
We can see that the probability distribution is heavily right skewed. In our next attempt we will select the cutoff at 0.09.

***

```{r, results='markup'}
test$logistic_model1.prob <- logistic_model1.prob
roc_model1 <- roc(FS_Status ~ logistic_model1.prob, data = test)
plot(roc_model1)
auc(roc_model1)


```

The area under curve is 0.87. Even though this seems good, but since our classes are imbalanced we can't put much significance to this metric in comparing different models.

***

Now we change the cutoff to 0.09


```{r, results='markup'}
logistic_model_ncf.prob <- predict(logistic, test, type = "response")
logistic_model_ncf.pred = rep("Food Secure", dim(test)[1])
logistic_model_ncf.pred[logistic_model1.prob > .09] = "Food Insecure"


tb <- table(logistic_model_ncf.pred, test$FS_Status)
tb[1:2,2:1]


precision <- round(precision(tb[1:2,2:1])*100,2)
recall <- round(recall(tb[1:2,2:1])*100,1)


lg_model_graph <- data.frame(precision , recall )
p <- barplot(as.matrix(lg_model_graph),beside=TRUE, col=rgb(0.2,0.3,0.6,0.6), space = c(0.1, 0.1))
text(x = p, y = lg_model_graph -  2, labels = lg_model_graph)
```

As you can see our recall has increased to 94.2%, while our precision has come down to 25.9%. We think this is better than our previous model since the previous model had just a recall of 5%.

***

The ration of Food Secure to Food Insecure in our data is 9.03. We are gonna add a weight equivalent to that to our minority class.


***

* **We are going to be adding weights to the data so our minority class, which is food insecurity, affects the model more**.


```{r, echo=FALSE, results='hide', message=FALSE, cache = TRUE}
weight_minority_class = sum(FS_Subset$FS_Status == "Food Secure")/sum(FS_Subset$FS_Status == "Food Insecure")
for(i in seq_len(NROW(FS_Subset))){
  
if(FS_Subset$FS_Status[i] == "Food Insecure"){
  FS_Subset$Weight[i] = weight_minority_class

}
  else
    FS_Subset$Weight[i] = 1
}

FS_Subset$Weight <- as.numeric(FS_Subset$Weight)
summary(FS_Subset$Weight)

## Splitting again because train and test doesn't contain the weight column

set.seed(1)

sample <- sample(c(TRUE, FALSE), nrow(FS_Subset), replace = TRUE, prob = c(0.6, 0.4))

train <- FS_Subset[sample, ]
test  <- FS_Subset[!sample, ]

```



```{r, echo=FALSE, results='hide', message=FALSE}

logistic_weighted <- glm(FS_Status ~ Ethnicity +  Family_Size + Household_Income:SNAP + Citizenship_status + Number_of_Jobs + Education_Level   , data = train,  weights = Weight, family = "binomial")


```


```{r, results='markup'}

logistic_model2.prob <- predict(logistic_weighted, test, type = "response")
logistic_model2.pred = rep("Food Secure", dim(test)[1])
logistic_model2.pred[logistic_model2.prob > .5] = "Food Insecure"



tb <- table(logistic_model2.pred, test$FS_Status)
tb[1:2,2:1]


precision <- round(precision(tb[1:2,2:1])*100,2)
recall <- round(recall(tb[1:2,2:1])*100,1)


lg_model_graph <- data.frame(precision , recall )
p <- barplot(as.matrix(lg_model_graph),beside=TRUE, col=rgb(0.2,0.3,0.6,0.6), space = c(0.1, 0.1))
text(x = p, y = lg_model_graph -  2, labels = lg_model_graph)


```


* As you can see even though our classification rate of the model went down and the precision of model went down drastically, we were able to raise our recall rate significantly. 

* But this result is almost the same as the one we got from the model where we changed the cutoff to 0.09. 

***

```{r, results='markup'}
test$logistic_model2.prob <- logistic_model2.prob
roc_model2 <- roc(FS_Status ~ logistic_model2.prob, data = test)
plot(roc_model2)
auc(roc_model2)



```

* The response probability distribution of this model looks very different from the previous models. We can clearly see two response classes are clearly separated for most part. 

* The little overlap that is visible may be due to the fact that our response originally had 4 classes but we had converted it into binary.

***

```{r, results='markup'}

train$logistic_model2_train.prob <- predict(logistic_weighted, train, type = "response")

# distribution of the prediction score grouped by known outcome
ggplot( train, aes( logistic_model2_train.prob, color = as.factor(FS_Status) ) ) + 
geom_density( size = 1 ) +
ggtitle( "Training Set's Predicted Score" ) + 
scale_color_economist( name = "data", labels = c( "Food Secure", "Food Insecure" ) ) + 
theme_economist()


```


*** 

* Trying to balance data by under sampling. We are using the ROSE library to do this

```{r,echo=FALSE, results='hide', message=FALSE}

data.balanced.ou <- ovun.sample(FS_Status~., data=train, p=0.5,  seed=1, method="under")$data




```

```{r, echo=FALSE, results='hide', message=FALSE}
logistic_both <- glm(FS_Status ~ Ethnicity + Family_Size + Household_Income:SNAP + Citizenship_status + Number_of_Jobs + Education_Level   , data = data.balanced.ou,   family = "binomial")

```

```{r, results='markup'}

logistic_model3.prob <- predict(logistic_both, test, type = "response")
logistic_model3.pred = rep("Food Secure", dim(test)[1])
logistic_model3.pred[logistic_model3.prob > .5] = "Food Insecure"




tb <- table(logistic_model3.pred, test$FS_Status)
tb[1:2,2:1]


precision <- round(precision(tb[1:2,2:1])*100,2)
recall <- round(recall(tb[1:2,2:1])*100,1)


lg_model_graph <- data.frame(precision , recall )
p <- barplot(as.matrix(lg_model_graph),beside=TRUE, col=rgb(0.2,0.3,0.6,0.6), space = c(0.1, 0.1))
text(x = p, y = lg_model_graph -  2, labels = lg_model_graph)

```
* As you can see the results are almost the same as the previous models. 

```{r, results='markup'}
train$logistic_model3_train.prob <- predict(logistic_both, train, type = "response")
ggplot( train, aes( logistic_model3_train.prob, color = as.factor(FS_Status) ) ) + 
geom_density( size = 1 ) +
ggtitle( "Training Set's Predicted Score" ) + 
scale_color_economist( name = "data", labels = c( "Food Secure", "Food Insecure" ) ) + 
theme_economist()
```

* The probability distribution also looks the same as the previous model.


***


# K-Nearest Neighbors

K-Nearest Neighbor is a Supervised ML Algorithm used to solve both Classification and Regression models. K-NN algorithm assumes the similarity between the new case /data and available cases and put the new case into the category that is most similar to the available categories.It is also called a lazy learner algorithm because it does not learn from the training set immediately instead it stores the dataset and at the time of classification, it performs an action on the dataset. KNN models shows high accuracy for classification problems.

##### Preparing Data for KNN

The preparation of Data for KNN starts with selecting variables. As KNN only accepts numeric variables in or it needs to be transformed to numeric. Since only ordinal variables can be transformed to numeric, we eliminate nominal variables and hence, the list of variables used for KNN model are - Family Size, Household Income, SNAP, Number of jobs, Hours of Job, Number of Children. The predicting variable FS_Status remains binary but numeric.
 
```{r, results='markup', echo=FALSE}
knn_data <- FS_Subset
knn_data$Family_Size <- as.numeric(FS_Subset$Family_Size)
knn_data$Household_Income <- as.numeric(FS_Subset$Household_Income)
knn_data$SNAP <- as.numeric(FS_Subset$SNAP)
knn_data$Number_of_Jobs <- as.numeric(FS_Subset$Number_of_Jobs)
knn_data$Hours_on_Jobs <- as.numeric(FS_Subset$Hours_on_Jobs)
knn_data$Number_of_children <- as.numeric(FS_Subset$Number_of_children)
knn_data$FS_Status <- as.numeric(FS_Subset$FS_Status)
num_data <- select_if(knn_data, is.numeric)
num_data <- as.data.frame(num_data[-c(8)])
num_data$FS_Status <- replace(num_data$FS_Status, num_data$FS_Status == 1, 3)
num_data$FS_Status <- replace(num_data$FS_Status, num_data$FS_Status == 2, 1)
num_data$FS_Status <- replace(num_data$FS_Status, num_data$FS_Status == 3, 2)
num_data$FS_Status <- replace(num_data$FS_Status, num_data$FS_Status == 2, 0)
str(num_data)
```

##### Balanced Data for KNN

The data is unbalanced as we mentioned above. The number of food secure units of observations are very few when compared to food secured. This issue is resolved in Logstic Regression by adding weights as a hypervariable. But for KNN, sampling methods needs to be used to resolve this isse. The two options are Under-sampling and Up-sampling. Even though, new methods are coming out like creating artificial data points, the accuracy of the model through such model is still debatable. 

This does not hide the fact that under-sampling can lead to data lose as we are discarding a lot of observation units and up-sampling will cause over-fitting in the model. Trying both techniques and finding the best model is a considerable solution but it depends on the resources and time. 

For the KNN model discussed below, we choose for Under-sampling. The results based on both sampling technique in data are discussed here. Now the data is balanced and we continue with data pre-processing. 

```{r, results='markup', echo=FALSE}
data_balanced_under <- as.data.frame(ovun.sample(FS_Status ~ ., data = num_data, method = "both", N = nrow(num_data), seed = 1)$data)
table(data_balanced_under$FS_Status)
```
```{r,results='markup', echo=FALSE}
num_data <- data_balanced_under
str(num_data)
```

For preparing the data for KNN, we selected numerical variables. Scaling the data is the next step. The algorithm should not be biased towards variables with higher magnitude. To overcome this problem, bring down all the variables to the same scale. Feature Scaling is inevitable for KNN. If not scaled, the feature with large value range will dominate while calculating distances. KNN which uses Euclidean distance needs scaling when the features has a broad range of values.

```{r, echo=FALSE, results='hide', message=FALSE}
scaledata <- as.data.frame(scale(num_data, center = TRUE, scale = TRUE))
str(scaledata)
```

We also need to create test and train data sets, we will do this slightly differently by using the sample function. The 2 says create 2 data sets essentially, replacement means we can reset the random sampling across each vector and the probability gives sample the weight of the splits, 2/3 for train, 1/3 for test. So here 67% of the sample is taken as training data and 33% as testing data.

```{r, echo=FALSE, results='hide', message=FALSE}
set.seed(1000)
knn_sample <- sample(2, nrow(scaledata), replace=TRUE, prob=c(0.67, 0.33))
```

We then just need to use the new variable to create the test/train outputs, selecting the first four rows as they are the numeric data in the iris data set and we want to predict Species. And then we need to create our Y variables or labels need to input into the KNN function.

```{r, results='markup', echo=FALSE}
# X variables
knn_training <- scaledata[knn_sample==1, 1:6]
knn_test <- scaledata[knn_sample==2, 1:6]
# Y variable
knn.trainLabels <- num_data[knn_sample==1, 7]
knn.testLabels <- num_data[knn_sample==2, 7]
```

```{r, echo=FALSE, results='hide', message=FALSE, warning=FALSE}
# Loading package
library(e1071)
library(caTools)
library(class)
loadPkg("gmodels")
loadPkg("gmodels")
loadPkg("FNN")
loadPkg("caret")
library(class)
```

##### Building KNN Model


```{r, results='markup'}
# create an empty dataframe to store the results from confusion matrices
ResultDf = data.frame( k=numeric(0), Total.Accuracy= numeric(0), row.names = NULL )
```

```{r, results='hide', echo=FALSE}
kseq <- seq(3,15,2)
for (kval in kseq) 
  {
  print( paste("k = ", kval) )
  knn_pred <- knn(train = knn_training, test = knn_test, cl=knn.trainLabels, k=kval)
  knn_crosst <- CrossTable(knn.testLabels, knn_pred, prop.chisq = FALSE)
  
  knn_crosst
  # 
  cm = confusionMatrix(knn_pred, reference = as.factor(knn.testLabels )) # from caret library
  # print.confusionMatrix(cm)
  # 
  cmaccu = cm$overall['Accuracy']
  #print( paste("Total Accuracy = ", cmaccu ) )
  # print("Other metrics : ")
  # print(cm$byClass)
  # 
  cmt = data.frame(k=kval, Total.Accuracy = cmaccu, row.names = NULL ) # initialize a row of the metrics 
  # cmt = cbind( cmt, data.frame( t(cm$byClass) ) ) # the dataframe of the transpose, with k valued added in front
  ResultDf = rbind(ResultDf, cmt)
  print( xkabledply(   as.matrix(cm), title = paste("ConfusionMatrix for k = ",kval ) ) )
  print( xkabledply(data.frame(cm$byClass), title=paste("k = ",kval)) )
  }
```

```{r, results='markup', echo=FALSE}
ResultDf = rbind(ResultDf, cmt)

xkabledply(ResultDf, "Total Accuracy Summary")
```

```{r, results='markup', echo=FALSE}
ggplot(ResultDf,aes(x = k, y = Total.Accuracy)) +
  geom_line(color = "maroon", size = 1.5) +
  geom_point(size = 3) + 
  labs(title = "Accuracy vs k")
```

From the above table and graph, it is quite clear that k=15 KNN model gives the most accuracy. Since the predicted variable is binary, we take into consideration k values which are odd. 

From the above results, the least accuracy is when k = 3, which is 81.49% and the maximum is for when k = 15 and it goes high up to 83.59%. Now we use confusion matrix of KNN model when k=15, to derive more information. 

```{r, results='markup', echo=FALSE}
library(class)
knn_pred <- knn(train = knn_training, test = knn_test, cl=knn.trainLabels, k=15)
knn_crosst <- gmodels::CrossTable(x = knn.testLabels, y = knn_pred, prop.chisq = FALSE)
```

*Total Number of Observations = 23430.
*True Negative = 9145 :- 39% of the observations were correctly predicted as NOT Food Insecure
*True Positive = 9912 :- 42% are correctly identified as Food Insecure.
*There were 1807 cases of False Negatives (FN). The FN’s poses a potential threat for the reason that model predicts 8% of the total observations as Food Secure, but was actually food insecure and the main focus to increase the accuracy of the model is to reduce FN’s.
*There were 2556 cases of False Positives (FP) meaning 11% were actually food secure in nature but got predicted as food insecure. 
*The total accuracy of the model is 83.59 % ((TN+TP)/Total) which shows that the model is good. But still there is scope of improvement.


```{r, results='markup'}
xkabledply(data.frame(cm$byClass), title=paste("k = ",15))
```

##### KNN Model: Conclusion 
Sensitivity is designate a food insecure individual as Food Insecure.A highly sensitive test means that there are few false negative results, and thus fewer cases of Food Insecurity are missed. For this model the Sensitivity is `r cm$byClass[1]*100`%

Specificity is designate a food secure individual as negative. A high specificity makes sure that, the projects for eradicating Food Insecurity focuses on the right population. Specificity for this model is `r cm$byClass[2]*100`%

# Trees

```{r, echo=FALSE, results='hide', message=FALSE, warning=FALSE}
# Loading package
library("caret")
library("dplyr")
library("data.tree")
library("caTools")
library("rpart.plot")
library("RColorBrewer")
library("rattle")
library("ISLR")
library("tree") 
library("rpart")
```
#Using Decision Tree


```{r, echo = T, fig.dim=c(6,4)}
tree1 <- rpart(FS_Status ~ Ethnicity + Family_Size + Household_Income + SNAP + Citizenship_status + Number_of_Jobs + Education_Level, data=data.balanced.ou, method="class")
printcp(tree1) # display the results 
plotcp(tree1) # visualize cross-validation results 

# plot tree 
plotcp(tree1)
#text(tree1, use.n=TRUE, all=TRUE, cex=.8)
rpart.plot(tree1)

predict_unseen <-predict(tree1, test, type = 'class')
table_mat <- table(predict_unseen, test$FS_Status)
table_mat[2:1,2:1]
```


```{r}
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
```

```{r}
tree2 <- rpart(FS_Status ~ Ethnicity + Family_Size + Household_Income + SNAP + Citizenship_status + Number_of_Jobs + Education_Level, data=data.balanced.ou, method="class", control = control)
printcp(tree2) # display the results 
plotcp(tree2) # visualize cross-validation results 

# plot tree 
plotcp(tree2)
#text(tree1, use.n=TRUE, all=TRUE, cex=.8)
rpart.plot(tree2)
```


```{r, results='markup'}
predict_unseen <-predict(tree2, test, type = 'class')
table_mat <- table(predict_unseen, test$FS_Status)
table_mat[2:1,2:1]
```

# Conclusions and further recommendations

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

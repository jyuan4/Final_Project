setwd('/Users/tomo/Desktop/7152/Final_Project')
getwd()

data1 <- read.csv("data/train.csv")
data2 <- read.csv("data/test_with_solutions.csv")
data <- rbind(data1, data2[,1:3])
#delete date
data <- data[,-c(2)]
write.csv(data, 'data.csv')
names(data)
dim(data)
head(data)
#check missing data
length(which(data$Insult==0))+length(which(data$Insult==1))
#no NA

#split data sets
smp_size <- floor(0.70 * nrow(data))
## set the seed to make your partition reproductible
set.seed(123) #fix training and testing sampling
train_ind <- sample(seq_len(nrow(data)), size = smp_size)
training <- data[train_ind, ]
testing <- data[-train_ind, ]

com.train <- training$Comment
com.test <- testing$Comment

tmp <- com.train[4]





## tm_map

# references
# http://cran.r-project.org/web/packages/tm/vignettes/tm.pdf
# http://cran.r-project.org/web/packages/tm/tm.pdf
# http://cran.r-project.org/web/packages/rjson/rjson.pdf
# http://www.inside-r.org/packages/rtexttools
# https://github.com/chenmiao/Big_Data_Analytics_Web_Text/wiki/Text-Preprocessing-with-R

char <- com.train
require(tm)
require(SnowballC)
pp = function(char){
  dd <- Corpus(VectorSource(char))
  dd <- tm_map(dd, stripWhitespace)
  dd <- tm_map(dd, tolower)
  dd <- tm_map(dd, removePunctuation)
  dd <- tm_map(dd, removeWords, stopwords("english"))
  dd <- tm_map(dd, stemDocument)
  dd <- tm_map(dd, removeNumbers)
  dtm <- tm_map(dd, stemDocument,language = 'english')
  return(dd)
}



#library(RTextTools)

# RTextTools is a machine learning package for automatic text classification that makes it simple for 
# novice users to get started with machine learning, while allowing experienced users to easily experiment 
# with different settings and algorithm combinations. The package includes nine algorithms for ensemble 
# classification (svm, slda, boosting, bagging, random forests, glmnet, decision trees, neural networks, 
#                 maximum entropy), comprehensive analytics, and thorough documentation.



## example of tm packages in R (TF-IDF algorithm)
library(tm)
library(plyr)
library(class)

docs <- c(D1 = "The sky is blue.", D2 = "The sun is bright.", D3 = "The sun in the sky is bright.")

docs <- gsub("\\\\\\\\n"," ",docs)
docs <- gsub("\\\\n"," ",docs)
docs <- gsub("\\n"," ",docs)
docs <- gsub("\\\\\\\\xa0"," ",docs)
docs <- gsub("\\\\xa0"," ",docs)
docs <- gsub("\\xa0"," ",docs)
docs <- gsub("\\\\\\\\u"," ",docs)
docs <- gsub("\\\\u"," ",docs)
docs <- gsub("\\u"," ",docs)
docs <- gsub("\\\\\\\\xc2"," ",docs)
docs <- gsub("\\\\xc2"," ",docs)
docs <- gsub("\\xc2"," ",docs)
docs <- gsub("\\\\\\\\"," ",docs)
docs <- gsub("\\\\"," ",docs)
docs <- gsub("\""," ",docs)


dd <- Corpus(VectorSource(docs)) #Make a corpus object from a text vector
#Clean the text
dd <- tm_map(dd, stripWhitespace)
dd <- tm_map(dd, tolower)
dd <- tm_map(dd, removePunctuation)
dd <- tm_map(dd, removeWords, stopwords("english"))
dd <- tm_map(dd, stemDocument, language = "english")  
dd <- tm_map(dd, removeNumbers)


dd <- tm_map(dd, stripWhitespace)
dd <- tm_map(dd, tolower)
dd <- tm_map(dd, removePunctuation)
dd <- tm_map(dd, removeWords, stopwords("english"))
dd <- tm_map(dd, stemDocument)
dd <- tm_map(dd, removeNumbers)
dd <- tm_map(dd, stemDocument,language = 'english')


#inspect(dd)
#gsub(" +"," ",gsub("^ +","",gsub("[^a-zA-Z0-9]"," ",dd[[1]])))

for (i in 1:length(dd)){
  dd[[i]] <- gsub(" +"," ",gsub("^ +","",gsub("[^a-zA-Z0-9]"," ", dd[[i]])))
  dd[[i]] <- gsub("youre"," ",dd[[i]])
  dd[[i]] <- gsub("youve"," ",dd[[i]])
  dd[[i]] <- gsub("www"," ",dd[[i]])
  dd[[i]] <- gsub("you"," ",dd[[i]])
  dd[[i]] <- gsub("xxx"," ",dd[[i]])
  dd[[i]] <- gsub("zzz"," ",dd[[i]])
  #dd[[i]] <- gsub("kkk"," ",dd[[i]])
  dd[[i]] <- gsub("wow"," ",dd[[i]])
  dd[[i]] <- gsub("its"," ",dd[[i]])
  dd[[i]] <- gsub("years"," ",dd[[i]])
  dd[[i]] <- gsub("haha"," ",dd[[i]])
  dd[[i]] <- gsub("http"," ",dd[[i]])
  dd[[i]] <- gsub("aaa"," ",dd[[i]])
}
dd <- Corpus(VectorSource(dd)) 





## stemming

matrix.tdm1 <- TermDocumentMatrix(dd, control = list(minWordLength = 3,maxWordLength=10))
matrix.tdm1 <- as.matrix(matrix.tdm1)
matrix.tdm1 <- t(matrix.tdm1) #transpose matrix
dim(matrix.tdm1)
matrix.tdm1 <- data.frame(matrix.tdm1)

insult.train <- training$Insult
train.dat <- cbind(matrix.tdm1, insult.train)
train.dat <- data.frame(train.dat)
names(train.dat)[ncol(train.dat)] <- "Insult"


library(randomForest)
set.seed(100)
#500 ntree
# Start the clock!
ptm <- proc.time()
RF <- randomForest(train.dat[,1:ncol(train.dat)-1], factor(train.dat$Insult),
                   sampsize=500, do.trace=TRUE, importance=TRUE, ntree=20, forest=TRUE) #control sampsize based on rows #
# 37.454 seconds for ntree=10
# Stop the clock
proc.time() - ptm
varImpPlot(RF)
rf.pred <- data.frame(Insult.pred=predict(RF,train.dat[,1:ncol(train.dat)-1],type="prob")[,2])

library(pROC)
set.seed(10)
roc.curve <- roc(rf.pred[,1], as.numeric(train.dat$Insult))
plot(roc.curve, main = "ROC: RF", col = "red")
auc.score<-auc(as.numeric(train.dat$Insult), rf.pred[,1])
auc.score
#Area under the curve: 0.8431

rf.pred[rf.pred>=0.5] <- 1
rf.pred[rf.pred<0.5] <- 0
rf.table <- table(pred=rf.pred[,1], train.dat$Insult)
sum(diag(rf.table))/sum(rf.table) #misclassification rate

#R memory stack size
#http://stat.ethz.ch/R-manual/R-devel/library/base/html/Memory.html
#logistic & decision tree - stack overflow problem

new.train <- train.dat[, c("idiot", "shit", "ass","ssy","losers","fuckers","whine","freak","coward",
                           "foolish","come","fake","despise","apology")]




#classification tree
library(tree)
train.dat$Insult <- factor(train.dat$Insult)
# Start the clock!
ptm <- proc.time()
tree.fit <- tree(Insult~.,data=train.dat) #takes time
# Stop the clock
proc.time() - ptm
summary(tree.fit)
plot(tree.fit)
text(tree.fit, pretty=0)
tree.fit

set.seed(2)
testing$Insult <- as.factor(testing$Insult)
tree.pred <- predict(tree.fit, testing, type="class")
tree.table <- table(tree.pred, testing$Insult)

library(pROC)
roc.curve <- roc(as.numeric(tree.pred)-1, as.numeric(testing$Insult))
plot(roc.curve, main = "ROC: Classification Tree", col = "red")
auc.score<-auc(as.numeric(testing$Insult), as.numeric(tree.pred)-1)
auc.score

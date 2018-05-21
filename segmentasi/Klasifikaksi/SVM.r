data <- read.csv("mytomat.csv",header = FALSE)
datas <- data

datas[, 1] <- NULL

summary(datas)

datas$V8 <- as.factor(as.character(datas$V8))

#sampling
set.seed(100)
sample.ind = sample(2,
                    nrow(datas),
                    replace =  T,
                    prob = c(0.1, 0.9)
)
dataTest = datas[sample.ind==1,]
dataTrain = datas[sample.ind==2,]

dataKu = dataTest[1,]


#Decision Tree
library(party)
robots.ct = ctree(class~., data=dataTrain)
robots.ct
plot(robots.ct)
dataTes

#SVM
library(e1071)
tomatku = svm(V8 ~., data=dataTrain)
summary(tomatku)
dataTest$Predicted = predict(tomatku, dataTest)
print(dataTest$Predicted)

library(caret)
print(
  confusionMatrix(data =dataTest$Predicted, 
                  reference=dataTest$V8
  )
)
  
################################
##### Breast Cancer Dataset#####
################################
cancerb<-read.csv(file="breast-cancer.csv", header=TRUE)
head(cancerb)
dim(cancerb)            ### Breast cancer data set 569 units 32 variables 
table(cancerb$diagnosis)
names(cancerb)       
#diagnosis is my response variable Malignant :( or Benign :)
#missing value?
anyNA(cancerb)                    #no missing value 
str(cancerb)                      #diagnosis is target variable M or B 
cancerb$id<-NULL                  #id variable is not needed
cancerb$diagnosis<-as.factor(cancerb$diagnosis)       # for classification 
##################
library(ggplot2)##
##################
#visualization of imbalanced data set
ggplot(cancerb, aes(x=factor(diagnosis)))+
  geom_bar(stat="count", width=0.7, fill="steelblue")+ theme_minimal() #350 units B, 200 units M data imbalanced!
##################################
##### SUPPORT VECTOR MACHINE #####
##################################
##### "caret" package needed for SVM
install.packages("caret")
#################
library(caret) ##           package to apply SVM 
#################
cancerb[["diagnosis"]]<-factor(cancerb[["diagnosis"]])   #in caret package use response as factor!
summary(cancerb)
#random sampling set train and test set 
set.seed(583)
traincreate<-createDataPartition(y=cancerb$diagnosis, p=0.7,list=FALSE)
training<-cancerb[traincreate,]         #train set
testing<-cancerb[-traincreate,]         #test set
dim(training) ; dim(testing)            #check dimentions of sets to make sure split

traincont<-trainControl(method="repeatedcv",number=10,repeats=3) #random sampling, repeated cross validation 
###### SVM use Linear classifier
svm_linear<-train(diagnosis~., data=training,method="svmLinear",trControl=traincont, 
                  preProcess=c("center","scale"), tuneLength=10)    #fit model for train set 
svm_linear
svm_linear2<-train(diagnosis~., data=training,method="svmLinear",trControl=traincont, 
                  preProcess=c("center","scale"), tuneGrid = expand.grid(C=seq(0,2,length=20)))
svm_linear2               # looking for most accuracy depending on cost                            
plot(svm_linear2)
svm_linear2$bestTune      #cost 0.21 is best tune
#test for the linear model with best tune, cost=0.21
test_pred_l<- predict(svm_linear2, newdata=testing)   # test the model with best tune model 
test_pred_l
CM1<- confusionMatrix(table(test_pred_l,testing$diagnosis))
CM1                     # accuracy 0.97

##### SVM Radial
svm_radial <-train(diagnosis~., data=training,method="svmRadial",trControl=traincont, 
                  preProcess=c("center","scale"), tuneLength=10)
svm_radial
plot(svm_radial)
svm_radial$bestTune          # cost=2 is best accuracy!  
#test for radial model with best tune, cost =2 
test_pred_r<- predict(svm_radial, newdata=testing)
test_pred_r
CM2<-confusionMatrix(table(test_pred_r,testing$diagnosis))
CM2                          # accuracy 0.96.47

##### SVM with Poly kernel 
svm_poly <-train(diagnosis~., data=training,method="svmPoly",trControl=traincont, 
                   preProcess=c("center","scale"), tuneLength=2)
svm_poly
plot(svm_poly)
svm_poly$bestTune          # cost= 0.5 is best accuracy!  
#test for the poly model with best tune cost =0.5
test_pred_p<- predict(svm_poly, newdata=testing)
test_pred_p
CM3<-confusionMatrix(table(test_pred_p,testing$diagnosis))
CM3                        # accuracy 0.97

#############################################
##### RANDOM FOREST & RANDOM FOREST SRC #####
#############################################
cancerb<-read.csv(file="breast-cancer.csv", header=TRUE)
dim(cancerb)
str(cancerb)
cancerb$diagnosis<- as.factor(cancerb$diagnosis)
##### Split data terain and test sets 
set.seed(583)
train = sample(1:nrow(cancerb), nrow(cancerb)*7/10)
traincancerb<-cancerb[train,]
testingcancerb<-cancerb[-train,]
dim(traincancerb) ; dim(testingcancerb)   #check dimesntions 
########################
library(randomForest) ##
########################
###### RF tree
set.seed(1)
rf.cancerbdata = randomForest(diagnosis~., data=traincancerb,mtry=5, importance=TRUE)  #fit model for train set 
print(rf.cancerbdata)

par(mfrow=c(1,1))
plot(rf.cancerbdata)    # OOB estimate of error 5.99%

# green M black OOB red B

yhat.rf=predict(rf.cancerbdata, newdata=testingcancerb)
mean(yhat.rf != cancerb$diagnosis[-train])

#Variable importance
importance(rf.cancerbdata)
varImpPlot((rf.cancerbdata))

# RandomForestSRC for imbalanced dataset method="BRF"
######################################   
library(randomForestSRC)            ##
install.packages("randomForestSRC") ##
######################################
set.seed(123)
c.brf<-imbalanced(diagnosis~., data=traincancerb, ntree=500,mtry=3, method=c("brf"),
                  importance="permute", pref.type="g.mean")
print(c.brf)                      # OOB misclassification rate is 3.952

pred.c.brf= predict(object=c.brf, newdata = testingcancerb)

### variable importance for BRF
vpcb<- c.brf$importance[,1]
print(vpcb)
plot(vpcb)
nms <- c("id","radiu_m","texture_m","perimeter_m","area_m","smoothness_m","compactness_m",
         "concavity_m","concave.points_m","symmetry_m","fractal_dim-m","raius_se","textre_se","prmtr_se",
         "area_se","s.hness_se","com.ness_se","concavity_se","concave.ps_se","symmetry_se","fractal_dim_se",
         "radius_w","texture_w","perimeter_w","area_w","smoothness_w","compactness_w","concavity_w",
         "concave.ps_w","symmetry_w","fractal_dimension_w")
barplot(vpcb,main="VIMP for BRF",col=4,horiz=TRUE,las=2, names.arg=nms)
barplot(vpcb,main="VIMP for BRF",col=4,horiz=FALSE,las=2, names.arg=nms)      

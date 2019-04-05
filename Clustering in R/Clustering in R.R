data<- read.csv("E:\\Shanu\\Machine-Learning-Projects\\Clustering in R\\Clustering in R.R")



## Glimse Of Data
head(data)

## Understanding Dataset


str(data)

## Global Imports


library(dplyr)
library(cluster)


# Creating New columns by adding Myprod and Comprod

data <-mutate(data, prod= MyProd1_Rx + MyProd2_Rx)
data<- mutate(data, comprod = CompProd1_Rx + CompProd2_Rx + CompProd3_Rx)
names(data)


## Subsetting required columns

data1 = data[,c("prod","comprod")]
View(data1)

## Standardizing data

data <- scale(data1)
head(data)

## Finding Optimum Cluster Size By Elbow Diagram

set.seed(6)
wcss = vector()
for (i in 1:10) wcss[i] = sum(kmeans(data, i)$withinss)
plot(1:10,
     
     wcss,
     
     type = 'b',
     
     main = paste('The Elbow Method'),
     
     xlab = 'Number of clusters',
     
     ylab = 'WCSS')


# Fitting K-Means to the dataset


set.seed(29)

kmeans = kmeans(x = data, centers = 4)

## Visualising the cluster


plot(data, col =(kmeans$cluster) , main="K-Means result with 4 clusters", pch=20, cex=1)

## Adding Cluster To Data

X<-data.frame(cbind(data1, kmeans$cluster))
head(X)

### Finding means of all clusters


a <- subset(X,kmeans.cluster==1)
mean(a$prod)
mean(a$comprod)

b <- filter (X,kmeans.cluster==2)
mean(b$prod)
mean(b$comprod)

c <- filter (X,kmeans.cluster==3)
mean(c$prod)
mean(c$comprod)

d <- filter (X,kmeans.cluster==4)
mean(d$prod)
mean(d$comprod)

##Final Output-  Mean of Pharma products and competiter products

#CLUSTER    PROD	      COMPROD	     HCP RANGE
#1	        0.3948	   47.2056	     SUPER HIGH
#2	        0.0066	   53.5391	     MEDIUM
#3	        0.0122	   129.465	     HIGH
#4	        0.00254	   22.9159	     LOW

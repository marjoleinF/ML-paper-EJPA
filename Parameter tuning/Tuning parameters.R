#########################
##
## Prepare data
##

data <- read.delim("data.csv", header = TRUE)

## Items should be scored 1-5, 0 may be missings
sapply(data[ , 1:48], table)
data[ , 1:48][sapply(data[ , 1:48], function(x) x == 0)] <- NA
data <- data[complete.cases(data[ , 1:48]), ]
sapply(data[ , 1:48], table)

## Select only university students
data <- data[data$education >= 3, ]

# ## Typos: psyhcology, psycotherapy
# table(data$major[grepl("therapy", data$major, ignore.case = TRUE)]) ## no, also includes massage therapy, physiotherapy
# table(data$major[grepl("education", data$major, ignore.case = TRUE)]) ## no, also includes other fields
# table(data$major[grepl("behavior", data$major, ignore.case = TRUE)]) ## yes, but exclude animal behavio(u)r
# table(data$major[grepl("behaviour", data$major)]) ## yes
# table(data$major[grepl("couns", data$major, ignore.case = FALSE)]) # yes
# table(data$major[grepl("neuro", data$major, ignore.case = FALSE)]) # tes
# 
# ## TODO: Perhaps  include all educational levels?
# ## TODO: If including all educational levels, exclude animal behavio(u)r 

## Recode response variable
table(data$major)
psych_ids <- rowSums(sapply(c("psych", "psyhcology", "psycotherapy", "couns", 
                              "behavior", "behaviour", "neuro"),
                            function(x) grepl(x, data$major, ignore.case = TRUE)))
table(psych_ids)
table(data$major[psych_ids == 1])
data$major <- factor(ifelse(psych_ids > 0, "psychology", "other"))


## Create train and test sets
set.seed(42)
test_ids <- sample(1:nrow(data), ceiling(nrow(data)/4))
train_ids <- which(!1:nrow(data) %in% test_ids)
train_y <- as.numeric(data$major)[train_ids] - 1
test_y <- as.numeric(data$major)[test_ids] - 1

## Generate scale scores
data$Real <- rowSums(data[ , paste0("R", 1:8)])
data$Inve <- rowSums(data[ , paste0("I", 1:8)])
data$Arti <- rowSums(data[ , paste0("A", 1:8)])
data$Soci <- rowSums(data[ , paste0("S", 1:8)])
data$Ente <- rowSums(data[ , paste0("E", 1:8)])
data$Conv <- rowSums(data[ , paste0("C", 1:8)])










####################################
##
## Cross validate parameters
##
##





#############
##
## glmnet
##
##
## Note: cv.glmnet function excludes lambda = 0 from default range,
##   so include 0 manually
##
library("glmnet")
plot.cv.glmnet <- function (x, sign.lambda = 1, cex = .7, main = "") {
  cvobj = x
  xlab = expression(Log(lambda))
  if (sign.lambda < 0) 
    xlab = paste("-", xlab, sep = "")
  plot.args = list(x = sign.lambda * log(cvobj$lambda), y = cvobj$cvm, 
                   ylim = range(cvobj$cvup, cvobj$cvlo), xlab = xlab, ylab = cvobj$name, 
                   type = "n", cex = cex, cex.lab = cex, cex.main = cex, cex.axis = cex,
                   main = main)
  do.call("plot", plot.args)
  glmnet:::error.bars(sign.lambda * log(cvobj$lambda), cvobj$cvup, cvobj$cvlo, 
                      width = 0.01, col = "darkgrey", cex = cex)
  points(sign.lambda * log(cvobj$lambda), cvobj$cvm, pch = 20, 
         col = "red", cex = cex)
  axis(side = 3, at = sign.lambda * log(cvobj$lambda), labels = paste(cvobj$nz), 
       tick = FALSE, line = 0, cex.axis = cex)
  abline(v = sign.lambda * log(cvobj$lambda.min), lty = 3)
  abline(v = sign.lambda * log(cvobj$lambda.1se), lty = 3)
  invisible()
}


## Lasso scale scores
varnames <- c("Real", "Inve", "Arti", "Soci", "Ente", "Conv")
X <- as.matrix(data[train_ids, varnames])
set.seed(42)
l1 <- cv.glmnet(X, train_y, alpha = 1, family = "binomial")
lambda_l1 <- l1$lambda
set.seed(42)
l1 <- cv.glmnet(X, train_y, alpha = 1, family = "binomial",
                   lambda = c(lambda_l1, 0))

## Ridge scale scores
set.seed(42)
l2 <- cv.glmnet(X, train_y, alpha = 0, family = "binomial")
lambda_l2 <- l2$lambda
set.seed(42)
l2 <- cv.glmnet(X, train_y, alpha = 0, lambda = c(lambda_l2, 0), family = "binomial")

## Plot results
par(mfrow = c(1, 2))
## plot(l1, cex = .7, main = "lasso")
plot(l2, cex = .7, main = "ridge")
## l1$lambda.min
## [1] 0
l1$lambda.1se
## [1] 0.007245717
l2$lambda.min
## [1] 0
l2$lambda.1se
## [1] 0.02264805


## Items
varnames <- paste0(rep(c("R", "I", "A", "S", "E", "C"), each = 8), 1:8)
X <- as.matrix(data[train_ids, varnames])

## Lasso
set.seed(42)
l1 <- cv.glmnet(X, train_y, alpha = 1, family = "binomial")
lambda_l1 <- l1$lambda
set.seed(42)
l1 <- cv.glmnet(X, train_y, alpha = 1, family = "binomial",
                lambda = c(lambda_l1, 0))

## Ridge
set.seed(42)
l2 <- cv.glmnet(X, train_y, alpha = 0, family = "binomial")
lambda_l2 <- l2$lambda
set.seed(42)
l2 <- cv.glmnet(X, train_y, alpha = 0, lambda = c(lambda_l2, 0), family = "binomial")

## Plot results
par(mfrow = c(1, 2))
plot(l1, cex = .7, main = "lasso")
plot(l2, cex = .7, main = "ridge")
l1$lambda.min
## [1] 0.0003251157
l1$lambda.1se
## [1] 0.002089868
l2$lambda.min
## [1] 0
l2$lambda.1se
## [1] 0.01656184









##############################
##
## CARET (GBM, RF, PRE, SVM)
##
##

## Load library, set up custom functions
library("caret")
library("ggplot2")
BigSummary <- function (data, lev = NULL, model = NULL) {
  brscore <- try(mean((data[, lev[2]] - ifelse(data$obs == lev[2], 1, 0)) ^ 2),
                 silent = TRUE)
  rocObject <- try(pROC::roc(ifelse(data$obs == lev[2], 1, 0), data[, lev[2]],
                             direction = "<", quiet = TRUE), silent = TRUE)
  if (inherits(brscore, "try-error")) brscore <- NA
  rocAUC <- if (inherits(rocObject, "try-error")) {
    NA
  } else {
    rocObject$auc
  }
  tmp <- unlist(e1071::classAgreement(table(data$obs,
                                            data$pred)))[c("diag", "kappa")]
  out <- c(Acc = tmp[[1]],
           Kappa = tmp[[2]],
           AUCROC = rocAUC,
           Brier = brscore)
  out
}
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 1,
                           ## Estimate class probabilities:
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function:
                           summaryFunction = BigSummary,
                           verboseIter = TRUE)


#############################
##
## Random forest 
##

## Subscales 
rfGrid <-  expand.grid(mtry = c(2:6), 
                       min.node.size = c(10000, 5000, 2500, 1000, 750, 500, 250),
                       splitrule = "gini")
set.seed(825)
rfFit <- train(major ~ Real + Inve + Arti + Soci + Ente + Conv, 
               data = data[train_ids, ], method = "ranger", trControl = fitControl, 
               tuneGrid = rfGrid, metric = "Brier", maximize = FALSE)
which(rfFit$results$AUCROC == max(rfFit$results$AUCROC))
which(rfFit$results$Brier == min(rfFit$results$Brier))
which(rfFit$results$Kappa == max(rfFit$results$Kappa))
which(rfFit$results$Acc == max(rfFit$results$Acc))
rfFit$bestTune
##   mtry splitrule min.node.size
## 9    3      gini           500
#save(rfFit, file = "rfFit.Rda")
load("rfFit.Rda")
ggplot(rfFit) + scale_x_continuous(trans="log")

## Items
rfGrid_i <-  expand.grid(mtry = 2*(1:5), 
                         min.node.size = c(10000, 5000, 2500, 1000, 750, 500, 250),
                         splitrule = "gini")
x <- data[train_ids, paste0(rep(c("R", "I", "A", "S", "E", "C"), each = 8), 1:8)]
y <- data$major[train_ids]
set.seed(825)
rfFit_i <- train(x = x, y = y, method = "ranger", trControl = fitControl, 
                 tuneGrid = rfGrid_i, metric = "Brier", maximize = FALSE)
#save(rfFit_i, file = "rfFit_i.Rda")
load("rfFit_i.Rda")
which(rfFit_i$results$AUCROC == max(rfFit_i$results$AUCROC))
which(rfFit_i$results$Brier == min(rfFit_i$results$Brier))
which(rfFit_i$results$Kappa == max(rfFit_i$results$Kappa))
which(rfFit_i$results$Acc == max(rfFit_i$results$Acc))
rfFit_i$bestTune
##     mtry splitrule min.node.size
## 50    10      gini           250
ggplot(rfFit_i) + scale_x_continuous(trans="log")



#############################
##
## Boosting
##

## Subscales
gbmGrid <-  expand.grid(interaction.depth = 1:5, 
                        n.trees = c((1:10)*50, 500+(1:10)*100, 1500+(1:4)*250), 
                        shrinkage = c(0.001, 0.01, 0.1),
                        n.minobsinnode = 20)
set.seed(825)
gbmFit <- train(major ~ Real + Inve + Arti + Soci + Ente + Conv, 
                data = data[train_ids, ], 
                 method = "gbm", 
                 trControl = fitControl, 
                 tuneGrid = gbmGrid,
                 metric = "Brier",
                 maximize = FALSE)
which(gbmFit$results$AUCROC == max(gbmFit$results$AUCROC))
which(gbmFit$results$Brier == min(gbmFit$results$Brier))
which(gbmFit$results$Kappa == max(gbmFit$results$Kappa))
which(gbmFit$results$Acc == max(gbmFit$results$Acc))
#save(gbmFit, file = "gbmFit.Rda")
load("gbmFit.Rda")
gbmFit$bestTune
##     n.trees interaction.depth shrinkage n.minobsinnode
## 184    1100                 3      0.01             20
ggplot(gbmFit)

## Items (items)
x <- data[train_ids, paste0(rep(c("R", "I", "A", "S", "E", "C"), each = 8), 1:8)]
y <- data$major[train_ids]
gbmGrid_i <-  expand.grid(interaction.depth = 1:5, 
                        n.trees = c((1:10)*50, 500+(1:10)*100, 1500+(1:8)*250), 
                        shrinkage = c(0.001, 0.01, 0.1),
                        n.minobsinnode = 20)
set.seed(825)
gbmFit_i <- train(x = x, y = y, 
                method = "gbm", 
                trControl = fitControl, 
                tuneGrid = gbmGrid_i,
                metric = "Brier",
                maximize = FALSE)
which(gbmFit_i$results$AUCROC == max(gbmFit_i$results$AUCROC))
which(gbmFit_i$results$Brier == min(gbmFit_i$results$Brier))
which(gbmFit_i$results$Kappa == max(gbmFit_i$results$Kappa))
which(gbmFit_i$results$Acc == max(gbmFit_i$results$Acc))
#save(gbmFit_i, file = "gbmFit_i.Rda")
load("gbmFit_i.Rda")
gbmFit_i$bestTune
##     n.trees interaction.depth shrinkage n.minobsinnode
## 280    3500                 5      0.01             20
ggplot(gbmFit_i)




#############################
##
## PRE
##

## Subscales
preGrid <- getModelInfo("pre")[[1]]$grid(
  learnrate = c(.01, .05, .1))
set.seed(825)
preFit <- train(major ~ Real + Inve + Arti + Soci + Ente + Conv, 
                data = data[train_ids, ], 
                method = "pre", 
                trControl = fitControl, 
                tuneGrid = preGrid,
                metric = "Brier",
                maximize = FALSE)
which(preFit$results$AUCROC == max(preFit$results$AUCROC))
which(preFit$results$Brier == min(preFit$results$Brier))
which(preFit$results$Kappa == max(preFit$results$Kappa))
which(preFit$results$Acc == max(preFit$results$Acc))
plot(preFit)
#save(preFit, file = "preFit.Rda")
load("preFit.Rda")
plot(preFit)
preFit$bestTune

## items
varnames <- paste0(rep(c("R", "I", "A", "S", "E", "C"), each = 8), 1:8)
pr_form <- formula(paste("major ~", paste(varnames, collapse = "+")))
set.seed(825)
preFit_i <- train(pr_form, 
                  data = data[train_ids, ], 
                  method = "pre", 
                  trControl = fitControl, 
                  tuneGrid = preGrid,
                  metric = "Brier",
                  maximize = FALSE)
which(preFit_i$results$AUCROC == max(preFit_i$results$AUCROC))
which(preFit_i$results$Brier == min(preFit_i$results$Brier))
which(preFit_i$results$Kappa == max(preFit_i$results$Kappa))
which(preFit_i$results$Acc == max(preFit_i$results$Acc))
plot(preFit_i)
#save(preFit_i, file = "preFit_i.Rda")

load("preFit_i.Rda")
plot(preFit_i)
preFit_i$bestTune



#############################
## 
## SVM
##
##
## Not run: CV-ing SVMs takes very long and I do 
##      not know how to best choose tuning grid
##

## subscales linear
getModelInfo("svmLinear")$svmLinear$grid
## Grid search returns only C = 1???
len <- 10
svmLgrid <- data.frame(C = 2^runif(len, min = -5, max = 10))
set.seed(825)
svmlFit <- train(major ~ Real + Inve + Arti + Soci + Ente + Conv, 
                  data = data[train_ids, ], 
                  method = "svmLinear", 
                  trControl = fitControl, 
                  tuneGrid = svmLgrid,
                  preProcess = c("center","scale"),
                  metric = "Brier",
                  maximize = FALSE)
which(svmrbFit$results$AUCROC == max(svmrbFit$results$AUCROC))
which(svmrbFit$results$Brier == min(svmrbFit$results$Brier))
which(svmrbFit$results$Kappa == max(svmrbFit$results$Kappa))
which(svmrbFit$results$Acc == max(svmrbFit$results$Acc))
#save(svmrbFit, file = "svmrbFit.Rda")
load("svmrbFit.Rda")
plot(svmrbFit)
svmrbFit$bestTune

## Subscales radial
getModelInfo("svmRadial")$svmRadial$grid()
set.seed(825)
svmrbFit <- train(major ~ Real + Inve + Arti + Soci + Ente + Conv, 
                data = data[train_ids, ], 
                method = "svmRadial", 
                trControl = fitControl, 
                tuneLength = 20,
                preProcess = c("center","scale"),
                ## Specify which metric to optimize
                metric = "Brier",
                maximize = FALSE)
which(svmrbFit$results$AUCROC == max(svmrbFit$results$AUCROC))
which(svmrbFit$results$Brier == min(svmrbFit$results$Brier))
which(svmrbFit$results$Kappa == max(svmrbFit$results$Kappa))
which(svmrbFit$results$Acc == max(svmrbFit$results$Acc))
#save(svmrbFit, file = "svmrbFit.Rda")
load("svmrbFit.Rda")
plot(svmrbFit)
svmrbFit$bestTune

## items radial
varnames <- paste0(rep(c("R", "I", "A", "S", "E", "C"), each = 8), 1:8)
pr_form <- formula(paste("major ~", paste(varnames, collapse = "+")))
set.seed(825)
svmrbFit_i <- train(pr_form, 
                  data = data[train_ids, ], 
                  method = "svmRadial", 
                  trControl = fitControl, 
                  tuneLength = 10,
                  preProcess = c("center","scale"),
                  ## Specify which metric to optimize
                  metric = "Brier",
                  maximize = FALSE)
which(svmrbFit_i$results$AUCROC == max(svmrbFit_i$results$AUCROC))
which(svmrbFit_i$results$Brier == min(svmrbFit_i$results$Brier))
which(svmrbFit_i$results$Kappa == max(svmrbFit_i$results$Kappa))
which(svmrbFit_i$results$Acc == max(svmrbFit_i$results$Acc))
#save(svmrbFit_i, file = "svmrbFit_i.Rda")
load("svmrbFit_i.Rda")
plot(svmrbFit_i)
svmrbFit_i$bestTune









#####################
##
## ctree and knn
##

library("partykit")
library("class")
library("glmertree")
dat <- data[train_ids, ]

## Subscales
varnames <- c("Real", "Inve", "Arti", "Soci", "Ente", "Conv")
ct_form <- formula(paste("major ~", paste(varnames, collapse = "+")))
set.seed(42)
fold_ids <- sample(1:10, size = nrow(dat), replace = TRUE)
ct_preds <- data.frame(matrix(rep(NA, times = nrow(dat)*15), nrow = nrow(dat)))
names(ct_preds) <- paste0("m", 1:15)
k <- c(1L, 10L, 25L, 50L, 75L, 100L, 150L, 200L, 250L, 300L, 400L, 500L, 600L)
knn_preds <- data.frame(matrix(rep(NA, times = nrow(dat)*length(k)), 
                                   nrow = nrow(dat)))
names(knn_preds) <- as.character(k)
dat$country[is.na(dat$country)] <- "NA"
dat$country <- factor(dat$country)

set.seed(43)
for (i in 1:10) {
  cat("Fold", i, ". ")
  cat("Fitting knns. ")
  for (j in k) {
    try(
      knn_mod <- knn(dat[fold_ids != i, varnames], dat[fold_ids == i, varnames], 
          cl = as.factor(dat[fold_ids != i, "major"]), 
          k = j, use.all = TRUE, prob = TRUE)
    )
    ## Need to obtain predicted probability for second class
    knn_preds[fold_ids == i, as.character(j)] <- ifelse(
      knn_mod == "psychology", attr(knn_mod, "prob"), 1 - attr(knn_mod, "prob"))
  }
  cat("Fitting ctrees. ")
  for (j in 1:15) {
    ct <- ctree(ct_form, data = dat[fold_ids != i, ], maxdepth = j)
    ct_preds[fold_ids == i, paste0("m", j)] <- predict(
      ct, type = "prob", newdata = dat[fold_ids == i, ])[ , "psychology"]
  }
}
br_ct <- sapply(ct_preds, function(x) mean((x - train_y)^2))
br_ct_se <- sapply(ct_preds, function(x) sd((x - train_y)^2)/sqrt(length(train_ids)))
br_knn <- sapply(knn_preds, function(x) mean((x - train_y)^2))
br_knn_se <- sapply(knn_preds, function(x) sd((x - train_y)^2)/sqrt(length(train_ids)))
par(mfrow = c(1, 3))
plot(br_ct, xlab = "maxdepth", ylab = "Brier score", main = "ctree CV results")
arrows(x0 = 1:15, y0 = br_ct - br_ct_se, y1 = br_ct + br_ct_se, length = 0)
plot(k, br_knn, main = "kNN CV results", ylab = "Brier score")
arrows(x0 = k, y0 = br_knn - br_knn_se, y1 = br_knn + br_knn_se, length = 0)
which(br_ct == min(br_ct))
## m6 
## 6 
which(br_knn == min(br_knn))
## 300 
## 10 


## Items
varnames <- paste0(rep(c("R", "I", "A", "S", "E", "C"), each = 8), 1:8)
ct_form <- formula(paste("major ~", paste(varnames, collapse = "+")))
set.seed(42)
fold_ids <- sample(1:10, size = nrow(dat), replace = TRUE)
ct_preds <- data.frame(matrix(rep(NA, times = nrow(dat)*15), nrow = nrow(dat)))
names(ct_preds) <- paste0("m", 1:15)
k <- c(1L, 10L, 25L, 50L, 75L, 100L, 150L, 200L, 250L, 300L, 400L, 500L, 600L)
knn_preds <- data.frame(matrix(rep(NA, times = nrow(dat)*length(k)), 
                               nrow = nrow(dat)))
names(knn_preds) <- as.character(k)
dat$country[is.na(dat$country)] <- "NA"
dat$country <- factor(dat$country)

set.seed(43)
for (i in 1:10) {
  cat("Fold", i, ". ")
  cat("Fitting knns. ")
  for (j in k) {
    try(
      knn_mod <- knn(dat[fold_ids != i, varnames], dat[fold_ids == i, varnames], 
                     cl = as.factor(dat[fold_ids != i, "major"]), 
                     k = j, use.all = TRUE, prob = TRUE)
    )
    ## Need to obtain predicted probability for second class
    knn_preds[fold_ids == i, as.character(j)] <- ifelse(
      knn_mod == "psychology", attr(knn_mod, "prob"), 1 - attr(knn_mod, "prob"))
  }
  cat("Fitting ctrees. ")
  for (j in 1:15) {
    ct <- ctree(ct_form, data = dat[fold_ids != i, ], maxdepth = j)
    ct_preds[fold_ids == i, paste0("m", j)] <- predict(
      ct, type = "prob", newdata = dat[fold_ids == i, ])[ , "psychology"]
  }
}
br_ct <- sapply(ct_preds, function(x) mean((x - train_y)^2))
br_ct_se <- sapply(ct_preds, function(x) sd((x - train_y)^2)/sqrt(length(train_ids)))
br_knn <- sapply(knn_preds, function(x) mean((x - train_y)^2))
br_knn_se <- sapply(knn_preds, function(x) sd((x - train_y)^2)/sqrt(length(train_ids)))
par(mfrow = c(1, 3))
plot(br_ct, xlab = "maxdepth", ylab = "Brier score", main = "ctree CV results")
arrows(x0 = 1:15, y0 = br_ct - br_ct_se, y1 = br_ct + br_ct_se, length = 0)
plot(k, br_knn, main = "kNN CV results", ylab = "Brier score")
arrows(x0 = k, y0 = br_knn - br_knn_se, y1 = br_knn + br_knn_se, length = 0)
which(br_ct == min(br_ct))
##  m9
## 9
which(br_knn == min(br_knn))
## 100
##   6
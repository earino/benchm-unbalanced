library(h2o)
library(unbalanced)
library(tidyverse)

h2o.init(nthreads=-1)

train_full <- h2o.importFile("train-0.01m.csv")
test <- h2o.importFile("test.csv")

delayed <- train_full[train_full$dep_delayed == "Y",]
on_time <- train_full[train_full$dep_delayed == "N",]

num_delayed <- nrow(delayed)

y = "dep_delayed_15min"
X <- names(train_full)[which(names(train_full)!= y)]

glm_h2o <- h2o.glm(x=X, y=y, training_frame = train_full,
                   validation_frame = test,
                   family = "binomial", alpha = 1, lambda = 0)

overall_glm_performance <- h2o.performance(glm_h2o, test)
print(paste0("glm auc on full set: ", h2o.auc(overall_glm_performance)))

glm_h2o_balanced <- h2o.glm(x=X, y=y, training_frame = train_full,
                            validation_frame = test,
                            balance_classes = TRUE,
                            family = "binomial", alpha = 1, lambda = 0)

overall_balanced_glm_performance <- h2o.performance(glm_h2o_balanced, test)
print(paste0("glm auc on full balanced set: ", h2o.auc(overall_balanced_glm_performance)))

min_class_sizes <- c(2^(2:log2(nrow(delayed))), nrow(delayed))

parameters <- list("h2o.glm"=list(x=X, y=y,
                                  family = "binomial", alpha = 1, lambda = 0),
                   "h2o.gbm"=list(x=X, y=y,
                                  distribution = "bernoulli",
                                  ntrees = 100, stopping_rounds = 10,
                                  max_depth = 16, learn_rate = 0.01, min_rows = 1,
                                  nbins = 100))



results <- data_frame()

for (algorithm in c("h2o.glm", "h2o.gbm")) {
  algorithm_ref <- get(algorithm)
  for (rows_to_sample in min_class_sizes) {
    for (number_of_resamples in 10) {
      indexes <- sort(sample(1:nrow(delayed), rows_to_sample))
      delayed_subsample <- delayed[indexes,]

      train_subsample <- as.h2o(rbind(as.data.frame(delayed_subsample), as.data.frame(on_time)))

      model_subsamp <- do.call(algorithm_ref, c(parameters[[algorithm]],
                                                list(training_frame=train_subsample))) #train a model

      model_subsamp_performance <- h2o.performance(model_subsamp, test)
      print(paste0("auc for ", algorithm," on ", rows_to_sample, "/", num_delayed, " set: ",
                   h2o.auc(model_subsamp_performance)))

      results <- rbind(results,
                       data_frame(algorithm=algorithm,
                                  balanced="N/A",
                                  minority_class_count=rows_to_sample,
                                  auc=h2o.auc(model_subsamp_performance)))

      model_subsamp <- do.call(algorithm_ref, c(parameters[[algorithm]], #train model balance classes
                                                list(training_frame=train_subsample,
                                                     balance_classes=TRUE)))

      model_subsamp_performance <- h2o.performance(model_subsamp, test)
      print(paste0("auc for balanced ", algorithm," on ", rows_to_sample, "/", num_delayed, " set: ",
                   h2o.auc(model_subsamp_performance)))

      results <- rbind(results,
                       data_frame(algorithm=algorithm,
                                  balanced="native",
                                  minority_class_count=rows_to_sample,
                                  auc=h2o.auc(model_subsamp_performance)))

      train_subsample_df <- as.data.frame(train_subsample)

      over_sampled_list <- ubOver(X = train_subsample_df[,X],
                                  Y = as.factor(as.numeric(train_subsample_df[,y]=="Y")))

      train_subsample_oversampled <- cbind(over_sampled_list[["X"]],
                                           x=over_sampled_list[["Y"]])

      train_subsample_oversampled$x <- factor(as.vector(train_subsample_oversampled$x),
                                              labels=c("N", "Y"))

      colnames(train_subsample_oversampled)[9] <- "dep_delayed_15min"
      train_subsample_oversampled <- as.h2o(train_subsample_oversampled)

      model_subsamp <- do.call(algorithm_ref, c(parameters[[algorithm]],
                                                list(training_frame=train_subsample_oversampled))) #train a model

      model_subsamp_performance <- h2o.performance(model_subsamp, test)
      print(paste0("auc for ", algorithm," using oversampling on ",
                   rows_to_sample, "/", num_delayed, " set: ",
                   h2o.auc(model_subsamp_performance)))

      results <- rbind(results,
                       data_frame(algorithm=algorithm,
                                  balanced="ubOver",
                                  minority_class_count=rows_to_sample,
                                  auc=h2o.auc(model_subsamp_performance)))


      }
  }
}

write_csv(results, "results.csv")

###################
# MovieLens Project 
# Lidia Almazan
# 5/06/2019
###################

###################################
# Create edx set and validation set
###################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1) # if using R 3.6.0: set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

############################
# Data analysis from edx set 
############################

# Loading libraries needed
library(ggplot2)
library(caret)
library(tidyverse)

# Exploration of the size of the data
dim(edx)

# Overview of the data
head(edx)

# Summary of the edx set
summary(edx)

# Number of movies and users 
edx %>% summarize(n_movies = n_distinct(edx$movieId),
                  n_users = n_distinct(edx$userId))

# Number of genres and times they are given to a film
edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(Count = n(), mean_rating = mean(rating)) %>%
  arrange(desc(Count)) %>% ggplot(aes(genres)) +
  geom_point(aes(genres, mean_rating, color = Count)) +
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
  labs(x = "", y = "Rating", title = "Mean rating per genre") +
  scale_color_gradient(low = "red", high = "green")

# Evaluation of how many times each rating is given
edx %>%
  group_by(rating) %>%
  summarize(count = n())  %>% 
  ggplot(aes(x = rating, y = count)) +
  geom_area(color = "red", fill = "green", alpha = .2) + 
  labs(x = "Rating", y = "Frequency", title = "Rating distribution")

# Distribution of number of ratings per movie 
edx %>% count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 20, col = "red", fill = "green", alpha = .2) +
  scale_x_log10() +
  labs(x = "Number of ratings", y = "Frequency", title = "Rating per movie")

# Movies rated only once (126 movies)
edx %>%
  group_by(movieId) %>%
  summarize(count = n()) %>%
  filter(count == 1) %>%
  left_join(edx, by = "movieId") %>%
  group_by(title) %>%
  summarize(rating = rating, n_rating = count) 

# Number of ratings given by users
edx %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 20, col = "red", fill = "green", alpha = .2) +
  scale_x_log10() +
  labs(x = "Number of ratings", y = "Number of users", title = "Ratings given by user")

# Mean ratings given by users
edx %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(mean_rating = mean(rating)) %>%
  ggplot(aes(mean_rating)) +
  geom_histogram(bins = 20, col = "red", fill = "green", alpha = .2) +
  labs(x = "Rating", y = "Number of users", title = "Mean ratings given by users")

#################################
# Machine learning algorithm that 
# predicting the rating
#################################

# 1 Model mean

# Mean training set
mu_hat <- mean(edx$rating)
mu_hat

# Test the results with the validation set
model1_rmse <- RMSE(validation$rating, mu_hat)
model1_rmse

# Save the prediction in a data frame
rmse_results <- data_frame(Model = "1 - Mean", RMSE = model1_rmse)
rmse_results %>% knitr::kable()

# 2 Model movie effect

# Improvement of the previous model adding a bias b_i to represent
# average ranking for movie i
movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu_hat))

# Plot the averages of movies
movie_avgs %>% qplot(b_i, geom = "histogram", bins = 20, data = ., 
                     color = I("red"), fill = I("green"), alpha = .1) + 
  theme(legend.position = "none") + xlab(bquote('b'['i']))
 
# Test the results with the validation set
predicted_ratings <- mu_hat + validation %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

model2_rmse <- RMSE(predicted_ratings, validation$rating)

# Adding the data to a data frame
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model="2 - Movie effect",  
                                     RMSE = model2_rmse ))

# Show the results
rmse_results %>% knitr::kable()

# 3 Model movie and user effect

# Improvement of the previous model adding also a bias b_u to represent
# average ranking for user u
user_avgs<- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))

# Plot the averages rating for user u rated over 100 movies
edx %>% group_by(userId) %>% summarize(b_u = mean(rating)) %>%
  filter(n()>=100) %>% qplot(b_u, geom = "histogram", bins = 20, data = ., 
      color = I("red"), fill = I("green"), alpha = .1) + 
  theme(legend.position = "none") + xlab(bquote('b'['u']))

# Test and save rmse results 
predicted_ratings <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  pull(pred)

model3_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model="3 - Movie and user effect",  
                                     RMSE = model3_rmse))

# Show the result
rmse_results %>% knitr::kable()

# 4 Model regularized movie and user effect

# lambda is a tuning parameter
# Use cross-validation to choose it.
lambdas <- seq(0, 10, 0.25)


# For each lambda,find b_i & b_u, followed by rating prediction & testing
# note:the below code could take some time  
rmses <- sapply(lambdas, function(l){

  mu_hat <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_hat)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu_hat)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu_hat + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
})

# Plot of rmse vs lambda                                                          
qplot(lambdas, rmses, color = I("red")) + xlab(expression(lambda)) + ylab("RMSE")

# Choose the optimal lambda                                                             
lambda <- lambdas[which.min(rmses)]
lambda

# Test and save results                                                             
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model="4 - Regularized movie and user effect",  
                                     RMSE = min(rmses)))

# RMSE final results
rmse_results %>% knitr::kable()

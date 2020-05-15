setwd("~/Cloud Storage/Google Drive/+Indiana University/X490 - Research SP20/LING-X490/LING-X490-SP20")

test <- read.delim("../Data/kaggle_data/test.csv", header=FALSE)

# train_dirty <- read.delim("../Data/kaggle_data/train.csv", header=FALSE)
train_clean <- read.delim("../Data/kaggle_data/train.target+comments.tsv", header=FALSE)
train_random <- read.delim("../Data/kaggle_data/train.random.csv", header=FALSE)
train_trump <- read.delim("../Data/kaggle_data/train.trump", header=FALSE) # filtered on Trump
# train_boosted <-

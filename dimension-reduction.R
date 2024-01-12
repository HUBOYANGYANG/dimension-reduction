library(dimensionreduction)

# 使用read.csv()函数导入.data数据
data <- read.csv("./data.csv")
y <- data[, 1]
X <- data[, -1]

#----------------------------------------------------------------------

# 调用自定义的PCA函数
pca_result <- armaPCA(as.matrix(X), num_components = 2)

# 根据解释的方差比例保留 95% 的方差
pca_result2 <- armaPCA(as.matrix(X), variance_percentage = 0.95)

# 使用R内置的prcomp函数
prcomp_result <- prcomp(as.matrix(X), retx = TRUE, rank = 2)
prcomp_result$x

# 使用prcomp计算PCA
prcomp_result2 <- prcomp(as.matrix(X))

#prcomp不能直接根据0.95输出相应主成分，需要自己进行计算
# 选择保留的信息百分比
information_percentage <- 0.95

# 计算每个主成分的方差比例
variance_ratio <- prcomp_result2$sdev^2 / sum(prcomp_result2$sdev^2)

# 计算累积方差比例
cumulative_variance_ratio <- cumsum(variance_ratio)

# 根据保留的信息百分比选择主成分个数
n_components <- sum(cumulative_variance_ratio < information_percentage) + 1

# 使用prcomp重新计算PCA，并指定保留的主成分个数
prcomp_result_ <- prcomp(as.matrix(X), retx = TRUE, rank = n_components)




#-------------------------------------------------------------------------------

library(MASS)

df <- cbind(X, y)
#内置的lda需要数据的列不能全部相同，否则会报错
constant_features <- sapply(df, function(x) length(unique(x))) == 1
df <- df[, !constant_features]

result <- lda(y ~ ., data = df)
projection <- result$scaling
new_data <- as.matrix(df[, -ncol(df)]) %*% projection

# 调用自定义的lda函数
new_data2 <- armalda(as.matrix(X), y, num_components = 1)


library(microbenchmark)
# 测试运行时间
microbenchmark(
  armalda(as.matrix(X), y, num_components = 1),
  lda(y ~ ., data = df),
  times = 10
)



#--------------------------------------------------------------------------

library(Rtsne)
# 运行t-SNE
#tsne_result <- Rtsne(X, dims = 2, perplexity = 30, verbose = TRUE)
# 提取降维结果
#tsne_data <- tsne_result$Y

# 调用自定义的t-SNE函数
#效果不好，运行时间太长，跑很久没有结果
#tsne_data2 <- armaTsne(as.matrix(X), num_dims = 2, num_iters = 1000, learning_rate = 200.0, perplexity = 30.0)

#-------------------------------------------------------------------------------
# #以下自己生成数据集对比
library(microbenchmark)
set.seed(42)
#data2 <- matrix(rnorm(6000 * 100), nrow = 6000, ncol = 100)
#data2 <- matrix(rnorm(1000 * 15), nrow = 1000, ncol = 15)
data2 <- matrix(rnorm(3000), ncol = 10)  # 生成一个随机数据矩阵

# 调用
# result_lle <- armalle(data2,  2, 10)
#result_Tsne <- armaTsne(data, num_dims = 2, num_iters = 1000, learning_rate = 200.0, perplexity = 30.0)

microbenchmark(
  armalle(data2, 2, 10),
  armaTsne(data2, num_dims = 2, num_iters = 1000, learning_rate = 200.0, perplexity = 30.0),
  times = 10
)


'
results3 <- microbenchmark(
  armalle(data, 2, 10),
  armaTsne(data, num_dims = 2, num_iters = 1000, learning_rate = 200.0, perplexity = 30.0),
  times = 10
)

results3$expr <- c("armalle", "armaTsne")

# 打印结果
print(results3)'



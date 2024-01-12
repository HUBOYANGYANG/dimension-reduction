#include <RcppArmadillo.h>
#include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

//----------------------------------------------------------------
//PCA
//----------------------------------------------------------------
//' Perform Principal Component Analysis (PCA) on a dataset.
//' 
//' This function takes a numeric matrix as input and performs PCA on the dataset.
//' The function allows the user to specify the number of principal components to retain
//' or the percentage of variance to be explained by the retained components.
//' 
//' @param data A numeric matrix containing the input data.
//' @param num_components The number of principal components to retain (default: -1).
//'                       If set to -1, all components will be retained.
//' @param variance_percentage The percentage of variance to be explained by the retained components (default: -1).
//'                            If set to -1, all components will be retained.
//' @return A numeric matrix containing the projected data in the principal component space.
//'
// [[Rcpp::export]]
NumericMatrix armaPCA(NumericMatrix data, int num_components = -1, double variance_percentage = -1) {
  // 将输入的NumericMatrix转换为Armadillo矩阵
  arma::mat X = as<arma::mat>(data);
  
  // 标准化数据（中心化）
  arma::mat centered_data = X.each_row() - mean(X);
  
  // 计算数据的协方差矩阵
  arma::mat covariance_matrix = cov(centered_data);
  
  // 奇异值分解 (SVD)
  arma::mat U;
  arma::vec s;
  arma::mat V;
  arma::svd(U, s, V, covariance_matrix);
  
  int num_features = s.n_elem;
  
  // 如果用户指定了保留的主成分数量
  if (num_components != -1 && num_components <= num_features) {
    // 提取前 num_components 个主成分
    arma::mat principal_components = V.head_cols(num_components);
    
    // 将数据投影到主成分空间
    arma::mat projected_data = centered_data * principal_components;
    
    // 将Armadillo矩阵转换为NumericMatrix，并返回
    NumericMatrix result = wrap(projected_data);
    return result;
  }
  
  // 如果指定了解释的方差比例
  else if (variance_percentage != -1 && variance_percentage > 0 && variance_percentage <= 1) {
    // 计算方差的百分比
    double total_var = sum(s);
    double threshold = variance_percentage * total_var;
    
    // 计算要保留的主成分数量
    double cumulative_var = 0.0;
    int num_components = 0;
    for (int i = 0; i < num_features; ++i) {
      cumulative_var += s(i);
      if (cumulative_var >= threshold) {
        num_components = i + 1;
        break;
      }
    }
    
    // 提取主成分
    arma::mat principal_components = V.head_cols(num_components);
    
    // 将数据投影到主成分空间
    arma::mat projected_data = centered_data * principal_components;
    
    // 将Armadillo矩阵转换为NumericMatrix，并返回
    NumericMatrix result = wrap(projected_data);
    return result;
  }
  
  // 如果参数不合法
  else {
    Rcpp::stop("Invalid parameters. Please provide either 'num_components' or 'variance_percentage'.");
  }
}


//__________________________________________________________________
// LDA
//__________________________________________________________________
//' This function implements the Linear Discriminant Analysis (LDA) algorithm to reduce the dimensionality of data to the specified number of components.
//' The input parameters include the data matrix X, the class label vector y, and the number of components num_components.
//'
//'@param X Data matrix, where each row represents a sample and each column represents a feature.
//' @param y Class label vector indicating the class membership of each sample.
//' @param num_components Number of dimensions for LDA dimensionality reduction.
//'
//' @return The reduced-dimensional data matrix, where each row represents a sample and each column represents a reduced feature.
//'
// [[Rcpp::export]]
arma::mat armalda(const arma::mat& X, const arma::vec& y, int num_components) {
  int num_classes = arma::max(y) + 1;  // 类别数量
  int num_features = X.n_cols;         // 特征数量
  
  arma::mat means(num_features, num_classes, arma::fill::zeros);  // 类别均值矩阵
  
  // 计算每个类别的均值向量
  for (int i = 0; i < num_classes; ++i) {
    arma::uvec indices = arma::find(y == i);
    arma::mat class_samples = X.rows(indices);
    means.col(i) = arma::mean(class_samples, 0).t();
  }
  
  // 计算类内散度矩阵
  arma::mat within_class_scatter(num_features, num_features, arma::fill::zeros);
  for (int i = 0; i < num_classes; ++i) {
    arma::uvec indices = arma::find(y == i);
    arma::mat class_samples = X.rows(indices);
    class_samples.each_row() -= means.col(i).t();  // 减去类别均值
    within_class_scatter += class_samples.t() * class_samples;
  }
  
  // 计算类间散度矩阵
  arma::mat between_class_scatter(num_features, num_features, arma::fill::zeros);
  for (int i = 0; i < num_classes; ++i) {
    arma::mat class_mean_diff = means.col(i) - arma::mean(means, 1);
    between_class_scatter += class_mean_diff * class_mean_diff.t();
  }
  
  // 计算特征值和特征向量
  arma::mat eigvec;
  arma::vec eigval;
  arma::eig_sym(eigval, eigvec, arma::pinv(within_class_scatter) * between_class_scatter);
  
  // 提取前num_components个特征向量
  arma::mat w = eigvec.cols(0, num_components - 1);
  
  // 对数据进行降维
  arma::mat X_lda = X * w;
  
  return X_lda;
}


//-----------------------------------------------------------------------
// t-sne
//-----------------------------------------------------------------------
#include <Rcpp.h>
#include <cmath>

// 计算两个数据点之间的欧氏距离
double euclideanDistance(const Rcpp::NumericVector& p1, const Rcpp::NumericVector& p2) {
  double distance = 0.0;
  int n = p1.size();
  
  for (int i = 0; i < n; i++) {
    double diff = p1[i] - p2[i];
    distance += diff * diff;
  }
  
  return std::sqrt(distance);
}

// 计算 t-SNE 嵌入
Rcpp::NumericMatrix tsne(const Rcpp::NumericMatrix& data, int num_dims, int num_iters, double learning_rate, double perplexity) {
  int n = data.nrow();
  
  // 随机初始化嵌入空间
  Rcpp::NumericMatrix embedding(n, num_dims);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < num_dims; j++) {
      embedding(i, j) = R::runif(-1.0, 1.0);
    }
  }
  
  // 执行 t-SNE 迭代
  for (int iter = 0; iter < num_iters; iter++) {
    // 使用当前嵌入计算成对相似度
    Rcpp::NumericMatrix embedding_distances(n, n);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (i == j) {
          embedding_distances(i, j) = 0.0;
        } else {
          double distance = euclideanDistance(embedding(i, Rcpp::_), embedding(j, Rcpp::_));
          embedding_distances(i, j) = distance;
        }
      }
    }
    
    // 计算 t-SNE 相似度
    Rcpp::NumericMatrix tsne_affinities(n, n);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (i == j) {
          tsne_affinities(i, j) = 0.0;
        } else {
          double similarity = std::exp(-embedding_distances(i, j));
          tsne_affinities(i, j) = similarity;
        }
      }
    }
    
    // 计算梯度
    Rcpp::NumericMatrix gradient(n, num_dims);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (i != j) {
          double grad = 4.0 * (1.0 - tsne_affinities(i, j)) * (embedding(i, 0) - embedding(j, 0));
          gradient(i, 0) += grad;
          gradient(j, 0) -= grad;
          
          for (int k = 1; k < num_dims; k++) {
            grad = 4.0 * (1.0 - tsne_affinities(i, j)) * (embedding(i, k) - embedding(j, k));
            gradient(i, k) += grad;
            gradient(j, k) -= grad;
          }
        }
      }
    }
    
    // 更新嵌入空间
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < num_dims; j++) {
        embedding(i, j) -= learning_rate * gradient(i, j);
      }
    }
  }
  
  return embedding;
}

//' R 包接口函数
//' This function serves as the interface between R and the t-SNE algorithm implemented in C++.
//' It takes the input data matrix, along with optional parameters, and returns the t-SNE embedding.
//' @param data Data matrix, where each row represents a sample and each column represents a feature.
//' @param num_dims Number of dimensions for the t-SNE embedding(default: 2).
//' @param num_iters Number of iterations for the t-SNE algorithm(default: 1000).
//' @param learning_rate Learning rate for the t-SNE algorithm(default: 200.0).
//' @param perplexity Perplexity parameter for the t-SNE algorithm(default: 30.0).
//'
//' @return The t-SNE embedding as a numeric matrix, where each row represents a sample and each column represents a reduced feature.
// [[Rcpp::export]]
Rcpp::NumericMatrix armaTsne(Rcpp::NumericMatrix data, int num_dims = 2, int num_iters = 1000, double learning_rate = 200.0, double perplexity = 30.0) {
  return tsne(data, num_dims, num_iters, learning_rate, perplexity);
}



//----------------------------------------------------------------------------------
//LLE
//-----------------------------------------------------------------------------------

//' Function to compute the Locally Linear Embedding (LLE) algorithm
//' This function takes the input data matrix, along with the parameters k and d, and returns the LLE embedding.
//'
//' @param data Data matrix, where each row represents a sample and each column represents a feature.
//' @param k Number of nearest neighbors to consider for each sample.
//' @param d Number of dimensions for the LLE embedding.
//'
//' @return The LLE embedding as a numeric matrix, where each row represents a sample and each column represents a reduced feature.
// [[Rcpp::export]]
NumericMatrix armalle(NumericMatrix data, int k, int d) {
  int n = data.nrow();
  int dim = data.ncol();
  
  // Convert data matrix to arma::mat
  arma::mat armaData = as<arma::mat>(data);
  
  // Initialize weight matrix
  arma::mat W(n, n, arma::fill::zeros);
  
  // Compute pairwise Euclidean distances
  arma::mat dist(n, n, arma::fill::zeros);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int l = 0; l < dim; l++) {
        double diff = armaData(i, l) - armaData(j, l);
        sum += diff * diff;
      }
      dist(i, j) = std::sqrt(sum);
    }
  }
  
  // Compute weight matrix
  for (int i = 0; i < n; i++) {
    // Find k nearest neighbors
    arma::rowvec dist_i = dist.row(i);
    arma::uvec idx = arma::sort_index(dist_i);
    idx = idx.subvec(1, k);
    
    // Compute weights
    arma::mat Z(k, dim, arma::fill::zeros);
    for (int j = 0; j < k; j++) {
      for (int l = 0; l < dim; l++) {
        Z(j, l) = armaData(idx(j), l) - armaData(i, l);
      }
    }
    arma::mat C = Z * Z.t();
    arma::vec w = arma::solve(C, arma::ones(k));
    w /= arma::sum(w);
    
    // Update weight matrix
    for (int j = 0; j < k; j++) {
      W(i, idx(j)) = w(j);
    }
  }
  
  // Compute embedding matrix
  arma::mat M = (arma::eye(n, n) - W) * (arma::eye(n, n)).t();
  arma::mat eigvec;
  arma::vec eigval;
  arma::eig_sym(eigval, eigvec, M);
  arma::mat embedding = eigvec.cols(1, d);
  
  // Convert embedding matrix to NumericMatrix
  NumericMatrix result = as<NumericMatrix>(wrap(embedding));
  
  return result;
}





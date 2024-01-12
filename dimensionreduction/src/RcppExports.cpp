// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// armaPCA
NumericMatrix armaPCA(NumericMatrix data, int num_components, double variance_percentage);
RcppExport SEXP _dimensionreduction_armaPCA(SEXP dataSEXP, SEXP num_componentsSEXP, SEXP variance_percentageSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type data(dataSEXP);
    Rcpp::traits::input_parameter< int >::type num_components(num_componentsSEXP);
    Rcpp::traits::input_parameter< double >::type variance_percentage(variance_percentageSEXP);
    rcpp_result_gen = Rcpp::wrap(armaPCA(data, num_components, variance_percentage));
    return rcpp_result_gen;
END_RCPP
}
// armalda
arma::mat armalda(const arma::mat& X, const arma::vec& y, int num_components);
RcppExport SEXP _dimensionreduction_armalda(SEXP XSEXP, SEXP ySEXP, SEXP num_componentsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< int >::type num_components(num_componentsSEXP);
    rcpp_result_gen = Rcpp::wrap(armalda(X, y, num_components));
    return rcpp_result_gen;
END_RCPP
}
// armaTsne
Rcpp::NumericMatrix armaTsne(Rcpp::NumericMatrix data, int num_dims, int num_iters, double learning_rate, double perplexity);
RcppExport SEXP _dimensionreduction_armaTsne(SEXP dataSEXP, SEXP num_dimsSEXP, SEXP num_itersSEXP, SEXP learning_rateSEXP, SEXP perplexitySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type data(dataSEXP);
    Rcpp::traits::input_parameter< int >::type num_dims(num_dimsSEXP);
    Rcpp::traits::input_parameter< int >::type num_iters(num_itersSEXP);
    Rcpp::traits::input_parameter< double >::type learning_rate(learning_rateSEXP);
    Rcpp::traits::input_parameter< double >::type perplexity(perplexitySEXP);
    rcpp_result_gen = Rcpp::wrap(armaTsne(data, num_dims, num_iters, learning_rate, perplexity));
    return rcpp_result_gen;
END_RCPP
}
// armalle
NumericMatrix armalle(NumericMatrix data, int k, int d);
RcppExport SEXP _dimensionreduction_armalle(SEXP dataSEXP, SEXP kSEXP, SEXP dSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type data(dataSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< int >::type d(dSEXP);
    rcpp_result_gen = Rcpp::wrap(armalle(data, k, d));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_dimensionreduction_armaPCA", (DL_FUNC) &_dimensionreduction_armaPCA, 3},
    {"_dimensionreduction_armalda", (DL_FUNC) &_dimensionreduction_armalda, 3},
    {"_dimensionreduction_armaTsne", (DL_FUNC) &_dimensionreduction_armaTsne, 5},
    {"_dimensionreduction_armalle", (DL_FUNC) &_dimensionreduction_armalle, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_dimensionreduction(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
#include "ml/rf/ml_rf.h"
using namespace rf_old;

void classRF (double *x, int *dimx, int *cl, int *ncl, int *cat,
	      int *maxcat, int *sampsize, int *strata, int *Options,
	      int *ntree, int *nvar, int *ipi, double *classwt,
	      double *cut, int *nodesize, int *outcl, int *counttr,
	      double *prox, double *imprt, double *impsd, double *impmat,
	      int *nrnodes, int *ndbigtree, int *nodestatus, int *bestvar,
	      int *treemap, int *nodeclass, double *xbestsplit,
	      double *errtr, int *testdat, double *xts, int *clts,
	      int *nts, double *countts, int *outclts, int labelts,
	      double *proxts, double *errts, int *inbag,
	      int print_verbose_tree_progression);

void rf_old::train (void** argout, int nargout[], void** argin)
{
  double* _tmp_d = NULL;
  int p_size = getScalar<int>(argin[16]);
  int n_size = getScalar<int>(argin[15]);
  double* x = getPr(argin[0]);
  int* y = (int*)getData(argin[1]);
  int nclass = getScalar<int>(argin[2]);
  int* cat = (int*)getData(argin[5]);
  int maxcat = *(int*)getData(argin[6]);
  int* sampsize = (int*)getData(argin[7]);
  int nsum = *(int*)getData(argin[14]);
  int* strata = (int*)getData(argin[8]);
  int* options = (int*)getData(argin[9]);
  int addclass = options[0];
  int importance = options[1];
  int localImp = options[2];
  int proximity = options[3];
  int oob_prox = options[4];
  int do_trance = options[5];
  int keep_forest = options[6];
  int replace = options[7];
  int stratify = options[8];
  int keep_inbag = options[9];
  int nsample = getScalar<int>(argin[17]);
  int dimx[] = {p_size, n_size};
  int tst_available = getScalar<int>(argin[18]);
  double* xts = NULL;
  int* yts = NULL;
  int nts;
  int* outclts = NULL;
  int labelts = 0;
  double* proxts = NULL;
  double* errts = NULL;
  int ntree = getScalar<int>(argin[3]);
  int mtry = getScalar<int>(argin[4]);
  int nt = ntree;
  int ipi = *((int*)getData(argin[10]));
  argout[10] = createIntScalar(mtry);
  nargout[20] = nargout[21] = 1;
  argout[3] = createNumericMatrix<double>(nclass, 1, 0.0);
  nargout[6] = 1;
  nargout[7] = nclass;
  double* classwt = (double*)getData(argout[3]);
  _tmp_d = (double*)getData(argin[11]);
  memcpy(classwt, _tmp_d, nclass * sizeof(double));
  argout[4] = createNumericMatrix<double>(nclass, 1, 0.0);
  nargout[8] = 1;
  nargout[9] = nclass;
  double* cutoff = (double*)getData(argout[4]);
  _tmp_d = (double*)getData(argin[12]);
  memcpy(cutoff, _tmp_d, nclass * sizeof(double));
  int nodesize = *(int*)getData(argin[13]);
  argout[11] = createNumericMatrix<int>(nsample, 1, 0);
  nargout[22] = 1;
  nargout[23] = nsample;
  int* outcl = (int*)getData(argout[11]);
  argout[12] = createNumericMatrix<int>(nclass, nsample, 0);
  nargout[24] = nsample;
  nargout[25] = nclass;
  int* counttr = (int*)getData(argout[12]);
  double* prox = NULL;
  if (proximity) {
    argout[13] = createNumericMatrix<double>(n_size, n_size, 0.0);
    nargout[26] = nargout[27] = n_size;
    prox = (double*)getData(argout[13]);
  }
  else {
    argout[13] = createNumericMatrix<double>(1, 1, 0.0);
    nargout[26] = nargout[27] = 1;
    prox = (double*)getData(argout[13]);
    prox[0] = 1.0;
  }
  double* impout = NULL;
  double* impmat = NULL;
  double* impSD = NULL;
  if (localImp) {
    if (addclass) {
      argout[14] = createNumericMatrix<double>(n_size * 2, p_size, 0.0);
      nargout[28] = p_size;
      nargout[29] = n_size * 2;
      impmat = (double*)getData(argout[14]);
    }
    else {
      argout[14] = createNumericMatrix<double>(n_size, p_size, 0.0);
      nargout[28] = p_size;
      nargout[29] = n_size;
      impmat = (double*)getData(argout[14]);
    }
  }
  else {
    argout[14] = createNumericMatrix<double>(1, 1, 0.0);
    nargout[28] = nargout[29] = 1;
    impmat = (double*)getData(argout[14]);
    impmat[0] = 1.0;
  }
  if (importance) {
    argout[15] = createNumericMatrix<double>(p_size, nclass + 2, 0.0);
    nargout[30] = nclass + 2;
    nargout[31] = p_size;
    argout[16] = createNumericMatrix<double>(p_size, nclass + 1, 0.0);
    nargout[32] = nclass + 1;
    nargout[33] = p_size;
    impout = (double*)getData(argout[15]);
    impSD = (double*)getData(argout[16]);
  }
  else {
    argout[15] = createNumericMatrix<double>(p_size, 1, 0.0);
    nargout[30] = 1;
    nargout[31] = p_size;
    argout[16] = createNumericMatrix<double>(1, 1, 0.0);
    nargout[32] = nargout[33] = 1;
    impout = (double*)getData(argout[15]);
    impSD = (double*)getData(argout[16]);
  }
  int nrnodes = 2 * (int)((double)nsum / (double)nodesize) + 1;
  argout[9] = createNumericMatrix<int>(1, nt, 0);
  nargout[18] = nt;
  nargout[19] = 1;
  int* ndbigtree = (int*)getData(argout[9]);
  argout[6] = createNumericMatrix<int>(nrnodes, nt, 0);
  nargout[12] = nt;
  nargout[13] = nrnodes;
  int* nodestatus = (int*)getData(argout[6]);
  argout[8] = createNumericMatrix<int>(nrnodes, nt, 0);
  nargout[16] = nt;
  nargout[17] = nrnodes;
  int* bestvar = (int*)getData(argout[8]);
  argout[5] = createNumericMatrix<int>(nrnodes, 2 * nt, 0);
  nargout[10] = 2 * nt;
  nargout[11] = nrnodes;
  int* treemap = (int*)getData(argout[5]);
  argout[7] = createNumericMatrix<int>(nrnodes, nt, 0);
  nargout[14] = nt;
  nargout[15] = nrnodes;
  int* nodepred = (int*)getData(argout[7]);
  argout[2] = createNumericMatrix<double>(nrnodes, nt, 0.0);
  nargout[4] = nt;
  nargout[5] = nrnodes;
  double* xbestsplit = (double*)getData(argout[2]);
  argout[17] = createNumericMatrix<double>(nclass + 1, ntree, 0.0);
  nargout[34] = ntree;
  nargout[35] = nclass + 1;
  double* errtr = (double*)getData(argout[17]);
  int testdat = 0;
  double* countts = NULL;
  int* inbag = NULL;
  if (keep_inbag) {
    argout[18] = createNumericMatrix<int>(n_size, ntree, 0);
    nargout[36] = ntree;
    nargout[37] = n_size;
    inbag = (int*)getData(argout[18]);
  }
  else {
    argout[18] = createNumericMatrix<int>(n_size, 1, 0);
    nargout[36] = 1;
    nargout[37] = n_size;
    inbag = (int*)getData(argout[18]);
  }
  argout[0] = createIntScalar(nrnodes);
  nargout[0] = nargout[1] = 1;
  argout[1] = createIntScalar(ntree);
  nargout[2] = nargout[3] = 1;
  if (tst_available) {
    xts = getPr(argin[19]);
    yts = (int*)getData(argin[20]);
    nts = getScalar<int>(argin[21]);
    argout[19] = createNumericMatrix<int>(nts, 1, 0);
    nargout[38] = 1;
    nargout[39] = nts;
    outclts = (int*)getData(argout[19]);
    countts = createNumericMatrix<double>(nclass, nts, 0.0);
    if (proximity) {
      argout[20] = createNumericMatrix<double>(nts, nts + n_size, 0.0);
      nargout[40] = nts + n_size;
      nargout[41] = nts;
      proxts = (double*)getData(argout[20]);
    }
    else {
      argout[20] = createNumericMatrix<double>(1, 1, 0.0);
      nargout[40] = nargout[41] = 1;
      proxts = (double*)getData(argout[20]);
      proxts[0] = 1;
    }
    argout[21] = createNumericMatrix<double>(nclass + 1, ntree, 0.0);
    nargout[42] = ntree;
    nargout[43] = nclass + 1;
    errts = (double*)getData(argout[21]);
    labelts = 1;
    testdat = 1;
  }
  else {
    xts = new double[1];
    yts = new int[1];
    nts = 1;
    argout[19] = createNumericMatrix<int>(1, 1, 0);
    nargout[38] = nargout[39] = 1;
    outclts = (int*)getData(argout[19]);
    countts = createNumericMatrix<double>(nclass, nts, 0.0);
    if (proximity) {
      argout[20] = createNumericMatrix<double>(1, 1, 0.0);
      nargout[40] = nargout[41] = 1;
      proxts = (double*)getData(argout[20]);
    }
    else {
      argout[20] = createNumericMatrix<double>(1, 1, 0.0);
      nargout[40] = nargout[41] = 1;
      proxts = (double*)getData(argout[20]);
      proxts[0] = 1;
    }
    argout[21] = createNumericMatrix<double>(nclass + 1, ntree, 0.0);
    nargout[42] = ntree;
    nargout[43] = nclass + 1;
    errts = (double*)getData(argout[21]);
    labelts = 0;
    testdat = 0;
  }
  int print_verbose_tree_progression = getScalar<int>(argin[22]);
  classRF(x, dimx, y, &nclass, cat, &maxcat, sampsize, strata, options,
	  &ntree, &mtry, &ipi, classwt, cutoff, &nodesize, outcl, counttr,
	  prox, impout, impSD, impmat, &nrnodes, ndbigtree, nodestatus,
	  bestvar, treemap, nodepred, xbestsplit, errtr, &testdat, xts,
	  yts, &nts, countts, outclts, labelts, proxts, errts, inbag,
	  print_verbose_tree_progression);
  del(&countts);
  if (tst_available == 0) {
    del(&xts);
    del(&yts);
  }
}



void rf_old::train (Model& model, double* X, int* Y, int N, int D,
		TrainExtraOptions& extra_options, int ntree, int mtry,
		double* Xtst, int* Ytst, int Ntst)
{
  int DEFAULTS_ON = 0;
  int tst_available;
  int tst_size;
  bool isXYTstToRelease = false;
  if (Xtst != NULL && Ytst != NULL) {
    std::cerr << "Test data available" << std::endl;
    tst_available = 1;
    tst_size = Ntst;
  }
  else {
    del(&Xtst);
    Xtst = copyNumericMatrix<double>(X, N, D, 0, 0, 0, D - 1);
    del(&Ytst);
    if (N > 1) Ytst = createIntScalar(Y[0]);
    else Ytst = createIntScalar(1);
    Ntst = 1;
    tst_available = 0;
    tst_size = 0;
    isXYTstToRelease = true;
  }
  int* orig_labels = NULL;
  int n_orig_labels;
  sort_unique(orig_labels, n_orig_labels, Y, N);
  int* new_labels = new int[n_orig_labels];
  int n_new_labels = n_orig_labels;
  std::map<int, int> orig_new_label_map;
  for (int i = 0; i < n_orig_labels; i++) {
    new_labels[i] = i + 1;
    orig_new_label_map[orig_labels[i]] = i + 1;
  }
  for (int i = 0; i < N; i++) Y[i] = orig_new_label_map[Y[i]];
  for (int i = 0; i < Ntst; i++) Ytst[i] = orig_new_label_map[Ytst[i]];
  int DEBUG_ON = -1;
  int replace = -1;
  double* classwt = NULL;
  int n_classwt;
  double* cutoff = NULL;
  int n_cutoff;
  int* strata = NULL;
  int n_strata;
  int* sampsize = NULL;
  int n_sampsize;
  int nodesize = -1;
  int importance = -1;
  int localImp = -1;
  int nPerm = -1;
  int proximity = -1;
  int oob_prox = -1;
  int do_trace = -1;
  int keep_inbag = -1;
  int print_verbose_tree_progression = -1;
  if (extra_options.DEBUG_ON >= 0) DEBUG_ON = extra_options.DEBUG_ON;
  if (extra_options.replace >= 0) replace = extra_options.replace;
  if (extra_options.classwt != NULL) {
    classwt = copyNumericVector(extra_options.classwt,
				extra_options.n_classwt);
    n_classwt = extra_options.n_classwt;
  }
  if (extra_options.cutoff != NULL) {
    cutoff = copyNumericVector(extra_options.cutoff,
			       extra_options.n_cutoff);
    n_cutoff = extra_options.n_cutoff;
  }
  if (extra_options.strata != NULL) {
    strata = copyNumericVector(extra_options.strata,
			       extra_options.n_strata);
    n_strata = extra_options.n_strata;
  }
  if (extra_options.sampsize != NULL) {
    sampsize = copyNumericVector(extra_options.sampsize,
				 extra_options.n_sampsize);
    n_sampsize = extra_options.n_sampsize;
  }
  if (extra_options.nodesize >= 0) nodesize = extra_options.nodesize;
  if (extra_options.importance >= 0) importance = extra_options.importance;
  if (extra_options.localImp >= 0) localImp = extra_options.localImp;
  if (extra_options.nPerm >= 0) nPerm = extra_options.nPerm;
  if (extra_options.proximity >= 0) proximity = extra_options.proximity;
  if (extra_options.oob_prox >= 0) oob_prox = extra_options.oob_prox;
  if (extra_options.do_trace >= 0) do_trace = extra_options.do_trace;
  if (extra_options.keep_inbag >= 0) keep_inbag = extra_options.keep_inbag;
  if (extra_options.print_verbose_tree_progression >= 0)
    print_verbose_tree_progression =
      extra_options.print_verbose_tree_progression;
  int keep_forest = 1;
  if (extra_options.DEBUG_ON < 0) DEBUG_ON = 0;
  if (extra_options.replace < 0) replace = 1;
  if (extra_options.sampsize == NULL) {
    if (replace) {
      sampsize = createIntScalar(N);
      n_sampsize = 1;
    }
    else {
      sampsize = createIntScalar((int)ceil(0.632 * (double)N));
      n_sampsize = 1;
    }
  }
  if (extra_options.nodesize < 0) nodesize = 1;
  if (extra_options.importance < 0) importance = 0;
  if (extra_options.localImp < 0) localImp = 0;
  if (extra_options.nPerm < 0) nPerm = 1;
  if (extra_options.do_trace < 0) do_trace = 0;
  if (extra_options.keep_inbag < 0) keep_inbag = 0;
  if (extra_options.print_verbose_tree_progression < 0)
    print_verbose_tree_progression = 0;
  if (ntree <= 0) {
    ntree = 500;
    DEFAULTS_ON = 1;
  }
  if (mtry <= 0 || mtry > D) mtry = (int)floor(sqrt((double)D));
  int addclass = N <= 0? 1: 0;
  if (addclass == 0 && n_orig_labels < 2) {
    std::cerr << "Need at least two classes for classification..."
	      << std::endl;
    exit(EXIT_FAILURE);
  }
  int n_size = N;
  int p_size = D;
  if (N == 0) {
    std::cerr << "Data X has 0 rows..." << std::endl;
    exit(EXIT_FAILURE);
  }
  if (mtry < 1 || mtry > D) DEFAULTS_ON = 1;
  mtry = std::max(1, std::min(D, mtry));
  if (DEFAULTS_ON) std::cerr << "Setting to default ntree = " << ntree
  			     << "and mtry = " << mtry << std::endl;
  if (N > 0) addclass = 0;
  else {
    addclass = 1;
    Y = new int[N * 2];
    fill_mem(Y, 1, N);
    fill_mem(Y + N, 2, N);
    double* Xtmp = X;
    X = new double[N * 2 * D];
    memcpy(X, Xtmp, N * D);
    memcpy(X + N * D, Xtmp, N * D);
    del(&Xtmp);
  }
  std::vector<double*> orig_uniques_in_feature;
  std::vector<int> n_orig_uniques_in_feature;
  std::vector<int*> mapped_uniques_in_feature;
  std::vector<int> n_mapped_uniques_in_feature;
  int* ncat = NULL;
  int rn_ncat = 0;
  int cn_ncat = 0;
  int n_ncat = 0;
  if (extra_options.categorical_feature != NULL) {
    orig_uniques_in_feature.resize(D);
    n_orig_uniques_in_feature.resize(D * 2);
    mapped_uniques_in_feature.resize(D);
    n_mapped_uniques_in_feature.resize(D * 2);
    ncat = createNumericMatrix<int>(1, D, 1);
    rn_ncat = 1;
    cn_ncat = D;
    n_ncat = rn_ncat * cn_ncat;
    for (int i = 0; i < D; i++) {
      if (extra_options.categorical_feature[i]) {
	int n;
	sort_unique(orig_uniques_in_feature[i], n, X + i, D, N);
	n_orig_uniques_in_feature[i] = n;
	mapped_uniques_in_feature[i] = new int[n];
	n_mapped_uniques_in_feature[i] = n;
	for (int j = 0; j < n; j++) {
	  mapped_uniques_in_feature[i][j] = j + 1;
	}
	for (int j = 0; j < n; j++) {
	  for (int k = 0; k < N; k++) {
	    if (X[k * D + i] == mapped_uniques_in_feature[i][j]) {
	      X[k * D + i] = mapped_uniques_in_feature[i][j];
	    }
	  }
	}
	ncat[i] = n;
      }
    }
  }
  else {
    ncat = createNumericMatrix<int>(1, D, 1);
    rn_ncat = 1;
    cn_ncat = D;
    n_ncat = rn_ncat * cn_ncat;
  }
  int maxcat = find_max(ncat, n_ncat);
  if (maxcat > 32) {
    std::cerr
      << "Cannot handle categorical predictors with 32+ categories..."
      << std::endl;
    exit(EXIT_FAILURE);
  }
  int nclass = n_orig_labels;
  if (cutoff == NULL) {
    cutoff = createNumericMatrix<double>(1, nclass, 1.0 / (double)nclass);
    n_cutoff = nclass;
  }
  else {
    double cutoff_sum = find_sum(cutoff, n_cutoff);
    if (cutoff_sum > 1.0 || cutoff_sum < 0.0
	|| is_exist(cutoff, n_cutoff, -DBL_MAX, 0.0)
	|| n_cutoff != nclass) {
      std::cerr << "Incorrect cutoff specified..." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  int ipi;
  if (classwt == NULL) {
    classwt = createNumericMatrix<double>(1, nclass, 1.0);
    n_classwt = nclass;
    ipi = 0;
  }
  else {
    if (n_classwt != nclass) {
      std::cerr << "Length of classwt not equal to number of classes..."
		<< std::endl;
      exit(EXIT_FAILURE);
    }
    if (is_exist(classwt, n_classwt, -DBL_MAX, 0.0)) {
      std::cerr << "classwt must be positive..." << std::endl;
      exit(EXIT_FAILURE);
    }
    ipi = 1;
  }
  if (proximity < 0) {
    proximity = addclass;
    oob_prox = proximity;
  }
  if (oob_prox < 0) oob_prox = proximity;
  if (localImp) importance = 1;
  if (importance && nPerm < 1) nPerm = 1;
  int nsample;
  if (addclass) nsample = 2 * n_size;
  else nsample = n_size;
  int stratify = n_sampsize > 1? 1: 0;
  if (stratify == 0 && *sampsize > N) {
    std::cerr << "Sampsize too large..." << std::endl;
    exit(EXIT_FAILURE);
  }
  int nsum;
  if (stratify) {
    if (n_orig_labels != n_sampsize) {
      std::cerr
	<< "Sampsize should be same as number of unique classes in Y"
	<< std::endl;
      exit(EXIT_FAILURE);
    }
    std::cerr << "Stratify in use" << std::endl;
    for (int i = 0; i < n_orig_labels; i++) {
      std::cerr << "Sampsize for class " << orig_labels[i] << " is "
		<< sampsize[i] << std::endl;
    }
    if (strata == NULL) {
      strata = copyNumericMatrix<int>(Y, N, 1, 0, N - 1, 0, 0);
      n_strata = N;
    }
    else if (is_exist(strata, n_strata, INT_MIN, 0)) {
      std::cerr << "Strata variable should have all values > 0..."
		<< std::endl;
      exit(EXIT_FAILURE);
    }
    nsum = find_sum(sampsize, n_sampsize);
    if (is_exist(sampsize, n_sampsize, INT_MIN, 0) || nsum == 0) {
      std::cerr << "Bad sampsize specification..." << std::endl;
    }
  }
  else {
    nsum = *sampsize;
    strata = createIntScalar(1);
    n_strata = 1;
  }
  int* options = new int[10];
  options[0] = addclass;
  options[1] = importance;
  options[2] = localImp;
  options[3] = proximity;
  options[4] = oob_prox;
  options[5] = do_trace;
  options[6] = keep_forest;
  options[7] = replace;
  options[8] = stratify;
  options[9] = keep_inbag;
  if (DEBUG_ON) {
    std::cerr << "size(X) = [" << N << " " << D << "] " << std::endl;
    std::cerr << "size(Y) = " << N << std::endl;
    std::cerr << "nclass = " << nclass << std::endl;
    std::cerr << "size(ncat) = " << n_ncat << std::endl;
    std::cerr << "maxcat = " << maxcat << std::endl;
    std::cerr << "size(sampsize) = " << n_sampsize << std::endl;
    std::cerr << "sampsize[0] = " << sampsize[0] << std::endl;
    std::cerr << "stratify = " << stratify << std::endl;
    std::cerr << "proximity = " << proximity << std::endl;
    std::cerr << "oob_prox = " << oob_prox << std::endl;
    std::cerr << "size(strata) = " << n_strata << std::endl;
    std::cerr << "*strata = " << *strata << std::endl;
    std::cerr << "ntree = " << ntree << std::endl;
    std::cerr << "mtry = " << mtry << std::endl;
    std::cerr << "ipi = " << ipi << std::endl;
    std::cerr << "classwt = ";
    print_vector(classwt, n_classwt);
    std::cerr << "cutoff = ";
    print_vector(cutoff, n_cutoff);
    std::cerr << "nodesize = " << nodesize << std::endl;
    std::cerr << "print_verbose_tree_progression = "
	      << print_verbose_tree_progression << std::endl;
  }
  void** argout = new void*[22];
  int nargout[44];
  void** argin = new void*[23];
  argin[0] = (void*)X;
  argin[1] = (void*)Y;
  argin[2] = (void*)&n_orig_labels;
  argin[3] = (void*)&ntree;
  argin[4] = (void*)&mtry;
  argin[5] = (void*)ncat;
  argin[6] = (void*)&maxcat;
  argin[7] = (void*)sampsize;
  argin[8] = (void*)strata;
  argin[9] = (void*)options;
  argin[10] = (void*)&ipi;
  argin[11] = (void*)classwt;
  argin[12] = (void*)cutoff;
  argin[13] = (void*)&nodesize;
  argin[14] = (void*)&nsum;
  argin[15] = (void*)&n_size;
  argin[16] = (void*)&p_size;
  argin[17] = (void*)&nsample;
  argin[18] = (void*)&tst_available;
  argin[19] = (void*)Xtst;
  argin[20] = (void*)Ytst;
  argin[21] = (void*)&tst_size;
  argin[22] = (void*)&print_verbose_tree_progression;
  train(argout, nargout, argin);
  int onrnodes = *(int*)argout[0];
  int ontree = *(int*)argout[1];
  double* oxbestsplit = (double*)argout[2];
  int rn_oxbestsplit = nargout[4];
  int cn_oxbestsplit = nargout[5];
  int n_oxbestsplit = rn_oxbestsplit * cn_oxbestsplit;
  double* oclasswt = (double*)argout[3];
  int rn_oclasswt = nargout[6];
  int cn_oclasswt = nargout[7];
  int n_oclasswt = rn_oclasswt * cn_oclasswt;
  double* ocutoff = (double*)argout[4];
  int rn_ocutoff = nargout[8];
  int cn_ocutoff = nargout[9];
  int n_ocutoff = rn_ocutoff * cn_ocutoff;
  int* otreemap = (int*)argout[5];
  int rn_otreemap = nargout[10];
  int cn_otreemap = nargout[11];
  int n_otreemap = rn_otreemap * cn_otreemap;
  int* onodestatus = (int*)argout[6];
  int rn_onodestatus = nargout[12];
  int cn_onodestatus = nargout[13];
  int n_onodestatus = rn_onodestatus * cn_onodestatus;
  int* onodeclass = (int*)argout[7];
  int rn_onodeclass = nargout[14];
  int cn_onodeclass = nargout[15];
  int n_onodeclass = rn_onodeclass * cn_onodeclass;
  int* obestvar = (int*)argout[8];
  int rn_obestvar = nargout[16];
  int cn_obestvar = nargout[17];
  int n_obestvar = rn_obestvar * cn_obestvar;
  int* ondbigtree = (int*)argout[9];
  int rn_ondbigtree = nargout[18];
  int cn_ondbigtree = nargout[19];
  int n_ondbigtree = rn_ondbigtree * cn_ondbigtree;
  int omtry = *(int*)argout[10];
  int* ooutcl = (int*)argout[11];
  int rn_ooutcl = nargout[22];
  int cn_ooutcl = nargout[23];
  int n_ooutcl = rn_ooutcl * cn_ooutcl;
  int* ocounttr = (int*)argout[12];
  int rn_ocounttr = nargout[24];
  int cn_ocounttr = nargout[25];
  int n_ocounttr = rn_ocounttr * cn_ocounttr;
  double* oprox = (double*)argout[13];
  int rn_oprox = nargout[26];
  int cn_oprox = nargout[27];
  int n_oprox = rn_oprox * cn_oprox;
  double* oimpmat = (double*)argout[14];
  int rn_oimpmat = nargout[28];
  int cn_oimpmat = nargout[29];
  int n_oimpmat = rn_oimpmat * cn_oimpmat;
  double* oimpout = (double*)argout[15];
  int rn_oimpout = nargout[30];
  int cn_oimpout = nargout[31];
  int n_oimpout = rn_oimpout * cn_oimpout;
  double* oimpSD = (double*)argout[16];
  int rn_oimpSD = nargout[32];
  int cn_oimpSD = nargout[33];
  int n_oimpSD = rn_oimpSD * cn_oimpSD;
  double* oerrtr = (double*)argout[17];
  int rn_oerrtr = nargout[34];
  int cn_oerrtr = nargout[35];
  int n_oerrtr = rn_oerrtr * cn_oerrtr;
  int* oinbag = (int*)argout[18];
  int rn_oinbag = nargout[36];
  int cn_oinbag = nargout[37];
  int n_oinbag = rn_oinbag * cn_oinbag;
  int* ooutclts = (int*)argout[19];
  int rn_ooutclts = nargout[38];
  int cn_ooutclts = nargout[39];
  int n_ooutclts = rn_ooutclts * cn_ooutclts;
  double* oproxts = (double*)argout[20];
  int rn_oproxts = nargout[40];
  int cn_oproxts = nargout[41];
  int n_oproxts = rn_oproxts * cn_oproxts;
  double* oerrts = (double*)argout[21];
  int rn_oerrts = nargout[42];
  int cn_oerrts = nargout[43];
  int n_oerrts = rn_oerrts * cn_oerrts;
  if (maxcat != 1) {
    int m = orig_uniques_in_feature.size();
    model.orig_uniques_in_feature.resize(m, NULL);
    model.n_orig_uniques_in_feature.resize(2 * m);
    for (int i = 0; i < m; i++) {
      int rn = n_orig_uniques_in_feature[i];
      int cn = 1;
      model.orig_uniques_in_feature[i] =
	copyNumericVector(orig_uniques_in_feature[i], rn * cn);
      model.n_orig_uniques_in_feature[i * 2] = rn;
      model.n_orig_uniques_in_feature[i * 2 + 1] = cn;
    }
    m = mapped_uniques_in_feature.size();
    model.mapped_uniques_in_feature.resize(m, NULL);
    model.n_mapped_uniques_in_feature.resize(m * 2);
    for (int i = 0; i < m; i++) {
      int rn = n_mapped_uniques_in_feature[i];
      int cn = 1;
      model.mapped_uniques_in_feature[i] =
	copyNumericVector(mapped_uniques_in_feature[i], rn * cn);
      model.n_mapped_uniques_in_feature[i * 2] = rn;
      model.n_mapped_uniques_in_feature[i * 2 + 1] = cn;
    }
    model.ncat = copyNumericVector(ncat, n_ncat);
    model.n_ncat[0] = rn_ncat;
    model.n_ncat[1] = cn_ncat;
    model.categorical_feature =
      copyNumericVector(extra_options.categorical_feature,
			extra_options.n_categorical_feature);
    model.n_categorical_feature[0] = extra_options.n_categorical_feature;
    model.n_categorical_feature[1] = 1;
  }
  model.nrnodes = onrnodes;
  model.ntree = ontree;
  model.xbestsplit = transpose(oxbestsplit, rn_oxbestsplit, cn_oxbestsplit);
  model.n_xbestsplit[0] = cn_oxbestsplit;
  model.n_xbestsplit[1] = rn_oxbestsplit;
  model.classwt = transpose(oclasswt, rn_oclasswt, cn_oclasswt);
  model.n_classwt[0] = cn_oclasswt;
  model.n_classwt[1] = rn_oclasswt;
  model.cutoff = transpose(ocutoff, rn_ocutoff, cn_ocutoff);
  model.n_cutoff[0] = cn_ocutoff;
  model.n_cutoff[1] = rn_ocutoff;
  model.treemap = transpose(otreemap, rn_otreemap, cn_otreemap);
  model.n_treemap[0] = cn_otreemap;
  model.n_treemap[1] = rn_otreemap;
  model.nodestatus = transpose(onodestatus, rn_onodestatus, cn_onodestatus);
  model.n_nodestatus[0] = cn_onodestatus;
  model.n_nodestatus[1] = rn_onodestatus;
  model.nodeclass = transpose(onodeclass, rn_onodeclass, cn_onodeclass);
  model.n_nodeclass[0] = cn_onodeclass;
  model.n_nodeclass[1] = rn_onodeclass;
  model.bestvar = transpose(obestvar, rn_obestvar, cn_obestvar);
  model.n_bestvar[0] = cn_obestvar;
  model.n_bestvar[1] = rn_obestvar;
  model.ndbigtree = transpose(ondbigtree, rn_ondbigtree, cn_ondbigtree);
  model.n_ndbigtree[0] = cn_ondbigtree;
  model.n_ndbigtree[1] = rn_ondbigtree;
  model.mtry = omtry;
  model.orig_labels = copyNumericVector(orig_labels, n_orig_labels);
  model.n_orig_labels[0] = n_orig_labels;
  model.n_orig_labels[1] = 1;
  model.new_labels = copyNumericVector(new_labels, n_new_labels);
  model.n_new_labels[0] = n_new_labels;
  model.n_new_labels[1] = 1;
  model.nclass = nclass;
  model.outcl = transpose(ooutcl, rn_ooutcl, cn_ooutcl);
  model.n_outcl[0] = cn_ooutcl;
  model.n_outcl[1] = rn_ooutcl;
  model.outclts = transpose(ooutclts, rn_ooutclts, cn_ooutclts);
  model.n_outclts[0] = cn_ooutclts;
  model.n_outclts[1] = rn_ooutclts;
  model.counttr = transpose(ocounttr, rn_ocounttr, cn_ocounttr);
  model.n_counttr[0] = cn_ocounttr;
  model.n_counttr[1] = rn_ocounttr;
  if (proximity) {
    model.proximity = transpose(oprox, rn_oprox, cn_oprox);
    model.n_proximity[0] = cn_oprox;
    model.n_proximity[1] = rn_oprox;
    if (tst_available) {
      model.proximity_tst = transpose(oproxts, rn_oproxts, cn_oproxts);
      model.n_proximity_tst[0] = cn_oproxts;
      model.n_proximity_tst[1] = rn_oproxts;
    }
    else {
      model.proximity_tst = NULL;
      model.n_proximity_tst[0] = 0;
      model.n_proximity_tst[1] = 0;
    }
  }
  else {
    model.proximity = NULL;
    model.n_proximity[0] = 0;
    model.n_proximity[1] = 0;
  }
  model.localImp = transpose(oimpmat, rn_oimpmat, cn_oimpmat);
  model.n_localImp[0] = cn_oimpmat;
  model.n_localImp[1] = rn_oimpmat;
  model.importance = transpose(oimpout, rn_oimpout, cn_oimpout);
  model.n_importance[0] = cn_oimpout;
  model.n_importance[1] = rn_oimpout;
  model.importanceSD = transpose(oimpSD, rn_oimpSD, cn_oimpSD);
  model.n_importanceSD[0] = cn_oimpSD;
  model.n_importanceSD[1] = rn_oimpSD;
  model.errtr = copyNumericVector(oerrtr, n_oerrtr);
  model.n_errtr[0] = rn_oerrtr;
  model.n_errtr[1] = cn_oerrtr;
  model.errts = copyNumericVector(oerrts, n_oerrts);
  model.n_errts[0] = rn_oerrts;
  model.n_errts[1] = cn_oerrts;
  model.inbag = transpose(oinbag, rn_oinbag, cn_oinbag);
  model.n_inbag[0] = cn_oinbag;
  model.n_inbag[1] = rn_oinbag;
  model.votes = copyNumericVector(ocounttr, n_ocounttr);
  model.n_votes[0] = rn_ocounttr;
  model.n_votes[1] = cn_ocounttr;
  model.oob_times =
    createNumericMatrix<int>(model.n_votes[0], model.n_votes[1], 0);
  model.n_oob_times[0] = model.n_votes[0];
  model.n_oob_times[1] = 1;
  for (int i = 0; i < model.n_oob_times[0]; i++) {
    model.oob_times[i] = find_sum(model.votes + i * model.n_votes[1],
			       model.n_votes[1], 1);
  }
  del(&orig_labels);
  del(&new_labels);
  del(&classwt);
  del(&cutoff);
  del(&strata);
  del(&sampsize);
  del(&ncat);
  del(&options);
  del((int**)&argout[0]);
  del((int**)&argout[1]);
  del((double**)&argout[2]);
  del((double**)&argout[3]);
  del((double**)&argout[4]);
  del((int**)&argout[5]);
  del((int**)&argout[6]);
  del((int**)&argout[7]);
  del((int**)&argout[8]);
  del((int**)&argout[9]);
  del((int**)&argout[10]);
  del((int**)&argout[11]);
  del((int**)&argout[12]);
  del((double**)&argout[13]);
  del((double**)&argout[14]);
  del((double**)&argout[15]);
  del((double**)&argout[16]);
  del((double**)&argout[17]);
  del((int**)&argout[18]);
  del((int**)&argout[19]);
  del((double**)&argout[20]);
  del((double**)&argout[21]);
  del(&argout);
  del(&argin);
  if (isXYTstToRelease) {
    del(&Xtst);
    del(&Ytst);
  }
}

#ifndef _H_RANDOMFOREST
#define _H_RANDOMFOREST

#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <climits>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <sstream>
#include <list>
#include <set>
#include <map>
#include <vector>
#include <string>
#include <cstring>
#include <fstream>
#include <sstream>

namespace rf_old {

  struct TrainExtraOptions {

    int DEBUG_ON;
    int replace;
    double* classwt;
    int n_classwt;
    double* cutoff;
    int n_cutoff;
    int* strata;
    int n_strata;
    int* sampsize;
    int n_sampsize;
    int nodesize;
    int importance;
    int localImp;
    int nPerm;
    int proximity;
    int oob_prox;
    int do_trace;
    int keep_inbag;
    int print_verbose_tree_progression;
    int* categorical_feature;
    int n_categorical_feature;

    TrainExtraOptions () : DEBUG_ON(-1),
      replace(-1),
      classwt(NULL),
      n_classwt(0),
      cutoff(NULL),
      n_cutoff(0),
      strata(NULL),
      n_strata(0),
      sampsize(NULL),
      n_sampsize(0),
      nodesize(-1),
      importance(-1),
      localImp(-1),
      nPerm(-1),
      proximity(-1),
      oob_prox(-1),
      do_trace(-1),
      keep_inbag(-1),
      print_verbose_tree_progression(-1),
      categorical_feature(NULL),
      n_categorical_feature(0) {}

    ~TrainExtraOptions () {
      if (classwt != NULL) { delete[] classwt; classwt = NULL; }
      if (cutoff != NULL) { delete[] cutoff; cutoff = NULL; }
      if (sampsize != NULL) { delete[] sampsize; sampsize = NULL; }
      if (categorical_feature != NULL) {
	delete[] categorical_feature;
	categorical_feature = NULL;
      }

    }
  };



  struct PredictExtraOptions {

    int predict_all;
    int proximity;
    int nodes;

    PredictExtraOptions () : predict_all(-1),
      proximity(-1),
      nodes(-1) {}

  };



  struct Model {

    std::vector<double*> orig_uniques_in_feature;
    std::vector<int> n_orig_uniques_in_feature;
    std::vector<int*> mapped_uniques_in_feature;
    std::vector<int> n_mapped_uniques_in_feature;
    int* ncat;
    int n_ncat[2];
    int* categorical_feature;
    int n_categorical_feature[2];
    int nrnodes;
    int ntree;
    double* xbestsplit;
    int n_xbestsplit[2];
    double* classwt;
    int n_classwt[2];
    double* cutoff;
    int n_cutoff[2];
    int* treemap;
    int n_treemap[2];
    int* nodestatus;
    int n_nodestatus[2];
    int* nodeclass;
    int n_nodeclass[2];
    int* bestvar;
    int n_bestvar[2];
    int* ndbigtree;
    int n_ndbigtree[2];
    int mtry;
    int* orig_labels;
    int n_orig_labels[2];
    int* new_labels;
    int n_new_labels[2];
    int nclass;
    int* outcl;
    int n_outcl[2];
    int* outclts;
    int n_outclts[2];
    int* counttr;
    int n_counttr[2];
    double* proximity;
    int n_proximity[2];
    double* proximity_tst;
    int n_proximity_tst[2];
    double* localImp;
    int n_localImp[2];
    double* importance;
    int n_importance[2];
    double* importanceSD;
    int n_importanceSD[2];
    double* errtr;
    int n_errtr[2];
    double* errts;
    int n_errts[2];
    int* inbag;
    int n_inbag[2];
    int* votes;
    int n_votes[2];
    int* oob_times;
    int n_oob_times[2];

    Model () {
      ncat = NULL;
      n_ncat[0] = n_ncat[1] = 0;
      categorical_feature = NULL;
      n_categorical_feature[0] = n_categorical_feature[1] = 0;
      nrnodes = -1;
      ntree = -1;
      xbestsplit = NULL;
      n_xbestsplit[0] = n_xbestsplit[1] = 0;
      classwt = NULL;
      n_classwt[0] = n_classwt[1] = 0;
      cutoff = NULL;
      n_cutoff[0] = n_cutoff[1] = 0;
      treemap = NULL;
      n_treemap[0] = n_treemap[1] = 0;
      nodestatus = NULL;
      n_nodestatus[0] = n_nodestatus[1] = 0;
      nodeclass = NULL;
      n_nodeclass[0] = n_nodeclass[1] = 0;
      bestvar = NULL;
      n_bestvar[0] = n_bestvar[1] = 0;
      ndbigtree = NULL;
      n_ndbigtree[0] = n_ndbigtree[1] = 0;
      mtry = -1;
      orig_labels = NULL;
      n_orig_labels[0] = n_orig_labels[1] = 0;
      new_labels = NULL;
      n_new_labels[0] = n_new_labels[1] = 0;
      nclass = -1;
      outcl = NULL;
      n_outcl[0] = n_outcl[1] = 0;
      outclts = NULL;
      n_outclts[0] = n_outclts[1] = 0;
      counttr = NULL;
      n_counttr[0] = n_counttr[1] = 0;
      proximity = NULL;
      n_proximity[0] = n_proximity[1] = 0;
      proximity_tst = NULL;
      n_proximity_tst[0] = n_proximity_tst[1] = 0;
      localImp = NULL;
      n_localImp[0] = n_localImp[1] = 0;
      importance = NULL;
      n_importance[0] = n_importance[1] = 0;
      importanceSD = NULL;
      n_importanceSD[0] = n_importanceSD[1] = 0;
      errtr = NULL;
      n_errtr[0] = n_errtr[1] = 0;
      errts = NULL;
      n_errts[0] = n_errts[1] = 0;
      inbag = NULL;
      n_inbag[0] = n_inbag[1] = 0;
      votes = NULL;
      n_votes[0] = n_votes[1] = 0;
      oob_times = NULL;
      n_oob_times[0] = n_oob_times[1] = 0;
    }

    ~Model () {
      for (std::vector<double*>::iterator itr =
	     orig_uniques_in_feature.begin();
	   itr != orig_uniques_in_feature.end(); itr++) {
	if (*itr != NULL) delete[] *itr;
      }
      orig_uniques_in_feature.clear();
      n_orig_uniques_in_feature.clear();
      for (std::vector<int*>::iterator itr =
	     mapped_uniques_in_feature.begin();
	   itr != mapped_uniques_in_feature.end(); itr++) {
	if (*itr != NULL) delete[] *itr;
      }
      mapped_uniques_in_feature.clear();
      n_mapped_uniques_in_feature.clear();
      if (ncat != NULL) { delete[] ncat; ncat = NULL; }
      if (categorical_feature != NULL) {
	delete[] categorical_feature;
	categorical_feature = NULL;
      }
      if (xbestsplit != NULL) { delete[] xbestsplit; xbestsplit = NULL; }
      if (classwt != NULL) { delete[] classwt; classwt = NULL; }
      if (cutoff != NULL) { delete[] cutoff; cutoff = NULL; }
      if (treemap != NULL) { delete[] treemap; treemap = NULL; }
      if (nodestatus != NULL) { delete[] nodestatus; nodestatus = NULL; };
      if (nodeclass != NULL) { delete[] nodeclass; nodeclass = NULL; }
      if (bestvar != NULL) { delete[] bestvar; bestvar = NULL; }
      if (ndbigtree != NULL) { delete[] ndbigtree; ndbigtree = NULL; }
      if (orig_labels != NULL) {
	delete[] orig_labels;
	orig_labels = NULL;
      }
      if (new_labels != NULL) { delete[] new_labels; new_labels = NULL; }
      if (outcl != NULL) { delete[] outcl; outcl = NULL; }
      if (outclts != NULL) { delete[] outclts; outclts = NULL; }
      if (counttr != NULL) { delete[] counttr; counttr = NULL; }
      if (proximity != NULL) { delete[] proximity; proximity = NULL; }
      if (proximity_tst != NULL) {
	delete[] proximity_tst;
	proximity_tst = NULL;
      }
      if (localImp != NULL) { delete[] localImp; localImp = NULL; }
      if (importance != NULL) { delete[] importance; importance = NULL; }
      if (importanceSD != NULL) {
	delete[] importanceSD;
	importanceSD = NULL;
      }
      if (errtr != NULL) { delete[] errtr; errtr = NULL; }
      if (errts != NULL) { delete[] errts; errts = NULL; }
      if (inbag != NULL) { delete[] inbag; inbag = NULL; }
      if (votes != NULL) { delete[] votes; votes = NULL; }
      if (oob_times != NULL) { delete[] oob_times; oob_times = NULL; }
    }

  };

  void* getData (void* p);
  double* getPr (void* p);
  void sort_unique (int*& y, int& n_y, int* x, int n_x, int n_x_inc = 1);
  void sort_unique (double*& y, int& n_y, double* x, int n_x,
		    int n_x_inc = 1);

  template <typename T>
    void fill_mem (T* p, T v, int n)
    {
      for (int i = 0; i < n; i++) p[i] = v;
    }



  template <typename T>
    T getScalar (void* p) { return *(T*)p; }



  template <typename T>
    double* createDoubleScalar (T v) {
    double* ret = new double[1];
    if (ret != NULL) ret[0] = (double)v;
    else {
      std::cerr << "Error creating double scalar..." << std::endl;
      exit(0);
    }
    return ret;
  }



  template <typename T>
    int* createIntScalar (T v)
    {
      int * ret = new int[1];
      if (ret != NULL) ret[0] = (int)v;
      else {
	std::cerr << "Error creating integer scalar..." << std::endl;
	exit(0);
      }
      return ret;
    }



  template <typename T>
    T* createNumericMatrix (int rn, int cn, T val)
    {
      T* ret = new T[rn * cn];
      if (ret != NULL) fill_mem(ret, val, rn * cn);
      return ret;
    }



  template <typename T>
    T* copyNumericMatrix (T* src, int rn_src, int cn_src, int r_st,
			  int r_end, int c_st, int c_end)
    {
      int rn_dst = r_end - r_st + 1;
      int cn_dst = c_end - c_st + 1;
      T* ret = new T[rn_dst * cn_dst];
      for (int r = 0; r < rn_dst; r++)
	memcpy(ret + r * cn_dst, src + r * cn_src + c_st,
	       cn_dst * sizeof(T));
      return ret;
    }



  template <typename T>
    T* copyNumericVector (T* src, int len)
    {
      return copyNumericMatrix(src, len, 1, 0, len - 1, 0, 0);
    }



  template <typename T>
    T* transpose (T* x, int rn, int cn)
    {
      int ret_rn = cn, ret_cn = rn;
      T* ret = new T[ret_rn * ret_cn];
      for (int ret_r = 0; ret_r < ret_rn; ret_r++) {
	for (int ret_c = 0; ret_c < ret_cn; ret_c++) {
	  int r = ret_c, c = ret_r;
	  ret[ret_r * ret_cn + ret_c] = x[r * cn + c];
	}
      }
      return ret;
    }



  template <typename T>
    T find_max (T* x, int n_x)
    {
      T ret = x[0];
      for (int i = 1; i < n_x; i++) if (ret < x[i]) ret = x[i];
      return ret;
    }



  template <typename T>
    T find_sum (T* x, int n_x, int x_inc = 1)
    {
      T ret = 0;
      for (int i = 0; i < n_x; i += x_inc) ret += x[i];
      return ret;
    }



  template <typename T>
    bool is_exist (T* x, int n_x, T lower, T upper)
    {
      for (int i = 0; i < n_x; i++) {
	if (x[i] >= lower && x[i] <= upper) return true;
      }
      return false;
    }



  template <typename T>
    void print_vector (T* vec, int n_vec, bool to_stdout = false)
    {
      if (to_stdout) std::cout << "[";
      else std::cerr << "[";
      for (int i = 0; i < n_vec; i++) {
	if (i != n_vec - 1) {
	  if (to_stdout) std::cout << vec[i] << " ";
	  else std::cerr << vec[i] << " ";
	}
	else {
	  if (to_stdout) std::cout << vec[i];
	  else std::cerr << vec[i];
	}
      }
      if (to_stdout) std::cout << "]" << std::endl;
      else std::cerr << "]" << std::endl;
    }



  template <typename T>
    void del (T** p)
    {
      if (*p != NULL) {
	delete[] (T*)*p;
	*p = NULL;
      }
    }



  void train (void** argout, int nargout[], void** argin);

  void train (Model& model, double* X, int* Y, int N, int D,
	      TrainExtraOptions&, int = 0, int = 0, double* = NULL,
	      int* = NULL, int = 0);

  void train (rf_old::Model& model,
	      std::list<std::list<float> > const& feats,
	      std::list<int> const& labels, int ntree, int mtry,
	      float sampsizeRatio, bool isBalanceClass,
	      bool printClassWeights);

  void writeModelToTextFile (const char*, Model const&);

  void writeModelToBinaryFile (const char*, Model const&);

  void readModelFromTextFile (Model&, const char*);

  void readModelFromBinaryFile (Model&, const char*);

  void deleteModel (Model&);

  void printModel (Model const&);

  void predict (void** argout, int nargout[], void** argin, int N, int D);

  void predict (int*& Y, double*& votes, int*& prediction_per_tree,
                double*& proximity_ts, int*& nodes, double* X,
                int N, int D, Model const& model,
                PredictExtraOptions& extra_options);

  void predict (std::list<double>& probs, Model const& model, int label,
		std::list<std::list<float> > const& feats);

  double predict (Model const& model, int label,
		  std::list<float> const& feat);

  void predict (double* preds, Model const& model, int label,
                double const* feats, int N, int D);

  double predict (Model const& model, int label,
                  double const* feat, int D);

  void readMatrixFromFile (double*&, int&, int&, const char*);

  void readMatrixFromFiles (double*&, int&, int&,
			    std::vector<const char*> const&);

  void readMatrixFromFiles (int*&, int&, int&,
			    std::vector<const char*> const&);

  void countLabel (std::map<int, int>&, int*, int);

};

#endif

#ifndef _glia_ml_rf_rf_hxx_
#define _glia_ml_rf_rf_hxx_

#include "glia_base.hxx"
#include "Eigen/Dense"
#include "ml/rf/ml_rf.h"

void classForest (
    int *mdim, int *ntest, int *nclass, int *maxcat, int *nrnodes,
    int *ntree, double *x, double *xbestsplit, double *pid,
    double *cutoff, double *countts, int *treemap, int *nodestatus,
    int *cat, int *nodeclass, int *jts, int *jet, int *bestvar,
    int *node, int *treeSize, int *keepPred, int *prox, double *proxMat,
    int *nodes);

namespace glia {
namespace ml {
namespace rf {

struct TrainOptions {
  int DEBUG_ON = -1;
  int replace = -1;
  std::vector<double> classwt;
  std::vector<double> cutoff;
  std::vector<int> strata;
  std::vector<int> sampsize;
  int nodesize = -1;
  int importance = -1;
  int localImp = -1;
  int nPerm = -1;
  int proximity = -1;
  int oob_prox = -1;
  int do_trace = -1;
  int keep_inbag = -1;
  int print_verbose_tree_progression = -1;
  std::vector<int> categorical_feature;
};


struct PredictOptions {
  int predict_all = -1;
  int proximity = -1;
  int nodes = -1;
};

class Model;

};
};
};

namespace {  // IO

const int MIN_SPARSE_SIZE = 128;

template <typename T> void
writeArray (std::ofstream& fs, T const* data, int size)
{
  if (size > 0) {
    if (size > MIN_SPARSE_SIZE) {
      std::vector<int> indices;
      indices.reserve(size);
      for (int i = 0; i < size; i++)
      {	if (glia::isfeq(data[i], 0.0)) { indices.push_back(i); } }
      // If more than half is zero, then write sparsely
      bool isSparse = indices.size() < size / 2;
      fs.write((char*)&isSparse, sizeof(bool));
      if (isSparse) {
	int s = indices.size();
	fs.write((char*)&s, sizeof(int));
	for (auto it = indices.begin(); it != indices.end(); it++) {
	  fs.write((char*)&(*it), sizeof(int));
	  fs.write((char*)(data + *it), sizeof(T));
	}
      } else { fs.write((char*)data, size * sizeof(T)); }
    } else { fs.write((char*)data, size * sizeof(T)); }
  }
}

template <typename T> void
writeVector (std::ofstream& fs, std::vector<T> const& data, int nrow)
{
  int ncol = data.size() / ncol;
  fs.write((char*)&nrow, sizeof(int));
  fs.write((char*)&ncol, sizeof(int));
  writeArray(fs, data.data(), data.size());
}

template <typename T> void
writeVectors (std::ofstream& fs, std::vector<std::vector<T>> const& data)
{
  int n = data.size();
  fs.write((char*)&n, sizeof(int));
  for (auto const& v : data) { writeVector(fs, v, 1); }
}

template <typename T> void
writeEigenMatrix (
    std::ofstream& fs,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> const& data)
{
  int n;
  n = data.rows();
  fs.write((char*)&n, sizeof(int));
  n = data.cols();
  fs.write((char*)&n, sizeof(int));
  writeArray(fs, data.data(), data.size());
}

template<typename T> void
readArray (T* data, std::ifstream& fs, int size)
{
  if (size > 0) {
    if (size > MIN_SPARSE_SIZE) {
      bool isSparse;
      fs.read((char*)&isSparse, sizeof(bool));
      if (isSparse) {
	int num;
	fs.read((char*)&num, sizeof(int));
	for (int i = 0; i < num; i++) {
	  int index;
	  fs.read((char*)&index, sizeof(int));
	  fs.read((char*)&data[index], sizeof(T));
	}
      } else { fs.read((char*)data, size * sizeof(T)); }
    } else { fs.read((char*)data, size * sizeof(T)); }
  } else { data = NULL; }
}

template <typename T> void
readVector (std::vector<T>& data, std::ifstream& fs)
{
  int nrow, ncol;
  fs.read((char*)&nrow, sizeof(int));
  fs.read((char*)&ncol, sizeof(int));
  data.resize(nrow * ncol);
  readArray(data.data(), fs, nrow * ncol);
}

template <typename T> void
readVectors (std::vector<std::vector<T>>& data, std::ifstream& fs)
{
  int n;
  fs.read((char*)&n, sizeof(int));
  data.resize(n);
  for (int i = 0; i < n; ++i) { readVector(data[i], fs); }
}

template <typename T> void
readEigenMatrix (
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& data,
    std::ifstream& fs)
{
  int nrow, ncol;
  fs.read((char*)&nrow, sizeof(int));
  fs.read((char*)&ncol, sizeof(int));
  data.resize(nrow, ncol);
  readArray(data.data(), fs, nrow * ncol);
}

};


namespace glia {
namespace ml {
namespace rf {

class Model {
 public:
  std::vector<std::vector<double>> orig_uniques_in_feature;
  std::vector<std::vector<int>> mapped_uniques_in_feature;
  Eigen::MatrixXi cat;
  std::vector<int> categorical_feature;
  int nrnodes = -1;
  int ntree = -1;
  Eigen::MatrixXd xbestsplit;
  Eigen::MatrixXd classwt;
  Eigen::MatrixXd cutoff;
  Eigen::MatrixXi treemap;
  Eigen::MatrixXi nodestatus;
  Eigen::MatrixXi nodeclass;
  Eigen::MatrixXi bestvar;
  Eigen::MatrixXi ndbigtree;
  int mtry = -1;
  std::vector<int> orig_labels;
  std::vector<int> new_labels;
  int nclass = -1;
  Eigen::MatrixXi outcl;
  Eigen::MatrixXi outclts;
  Eigen::MatrixXi counttr;
  Eigen::MatrixXd proximity;
  Eigen::MatrixXd proximity_tst;
  Eigen::MatrixXd localImp;
  Eigen::MatrixXd importance;
  Eigen::MatrixXd importanceSD;
  Eigen::MatrixXd errtr;
  Eigen::MatrixXd errts;
  Eigen::MatrixXi inbag;
  Eigen::MatrixXi votes;
  Eigen::MatrixXi oob_times;

  void write (std::string file) {
    std::ofstream fs(file, std::ios::binary);
    if (fs.is_open()) {
      writeVectors(fs, orig_uniques_in_feature);
      writeVectors(fs, mapped_uniques_in_feature);
      writeEigenMatrix(fs, cat);
      writeVector(fs, categorical_feature, 1);
      fs.write((char*)&nrnodes, sizeof(int));
      fs.write((char*)&ntree, sizeof(int));
      writeEigenMatrix(fs, xbestsplit);
      writeEigenMatrix(fs, classwt);
      writeEigenMatrix(fs, cutoff);
      writeEigenMatrix(fs, treemap);
      writeEigenMatrix(fs, nodestatus);
      writeEigenMatrix(fs, nodeclass);
      writeEigenMatrix(fs, bestvar);
      writeEigenMatrix(fs, ndbigtree);
      fs.write((char*)&mtry, sizeof(int));
      writeVector(fs, orig_labels, 1);
      writeVector(fs, new_labels, 1);
      fs.write((char*)&nclass, sizeof(int));
      writeEigenMatrix(fs, outcl);
      writeEigenMatrix(fs, outclts);
      writeEigenMatrix(fs, counttr);
      writeEigenMatrix(fs, proximity);
      writeEigenMatrix(fs, proximity_tst);
      writeEigenMatrix(fs, localImp);
      writeEigenMatrix(fs, importance);
      writeEigenMatrix(fs, importanceSD);
      writeEigenMatrix(fs, errtr);
      writeEigenMatrix(fs, errts);
      writeEigenMatrix(fs, inbag);
      writeEigenMatrix(fs, votes);
      writeEigenMatrix(fs, oob_times);
      fs.close();
    } else { perr("Error: cannot create model file..."); }
  }

  void read (std::string file) {
    std::ifstream fs(file, std::ios::binary);
    if (fs.is_open()) {
      readVectors(orig_uniques_in_feature, fs);
      readVectors(mapped_uniques_in_feature, fs);
      readEigenMatrix(cat, fs);
      readVector(categorical_feature, fs);
      fs.read((char*)&nrnodes, sizeof(int));
      fs.read((char*)&ntree, sizeof(int));
      readEigenMatrix(xbestsplit, fs);
      readEigenMatrix(classwt, fs);
      readEigenMatrix(cutoff, fs);
      readEigenMatrix(treemap, fs);
      readEigenMatrix(nodestatus, fs);
      readEigenMatrix(nodeclass, fs);
      readEigenMatrix(bestvar, fs);
      readEigenMatrix(ndbigtree, fs);
      fs.read((char*)&mtry, sizeof(int));
      readVector(orig_labels, fs);
      readVector(new_labels, fs);
      fs.read((char*)&nclass, sizeof(int));
      readEigenMatrix(outcl, fs);
      readEigenMatrix(outclts, fs);
      readEigenMatrix(counttr, fs);
      readEigenMatrix(proximity, fs);
      readEigenMatrix(proximity_tst, fs);
      readEigenMatrix(localImp, fs);
      readEigenMatrix(importance, fs);
      readEigenMatrix(importanceSD, fs);
      readEigenMatrix(errtr, fs);
      readEigenMatrix(errts, fs);
      readEigenMatrix(inbag, fs);
      readEigenMatrix(votes, fs);
      readEigenMatrix(oob_times, fs);
      fs.close();
    } else { perr("Error: cannot read model file..."); }
  }

  void readFromOld (std::string file) {
    rf_old::Model om;
    rf_old::readModelFromBinaryFile(om, file.c_str());
    orig_uniques_in_feature.resize(om.orig_uniques_in_feature.size());
    for (int i = 0; i < om.orig_uniques_in_feature.size(); ++i) {
      orig_uniques_in_feature[i].resize(om.n_orig_uniques_in_feature[i]);
      std::memcpy(
          orig_uniques_in_feature[i].data(),
          om.orig_uniques_in_feature[i],
          sizeof(double) * om.n_orig_uniques_in_feature[i]);
    }
    mapped_uniques_in_feature.resize(om.mapped_uniques_in_feature.size());
    for (int i = 0; i < om.mapped_uniques_in_feature.size(); ++i) {
      mapped_uniques_in_feature[i].resize(
          om.n_mapped_uniques_in_feature[i]);
      std::memcpy(
          mapped_uniques_in_feature[i].data(),
          om.mapped_uniques_in_feature[i],
          sizeof(double) * om.n_mapped_uniques_in_feature[i]);
    }
    cat = Eigen::Map<Eigen::MatrixXi>(
        om.ncat, om.n_ncat[0], om.n_ncat[1]);
    categorical_feature.resize(
        om.n_categorical_feature[0] * om.n_categorical_feature[1]);
    std::memcpy(
        categorical_feature.data(), om.categorical_feature,
        sizeof(int) * om.n_categorical_feature[0] *
        om.n_categorical_feature[1]);
    nrnodes = om.nrnodes;
    ntree = om.ntree;
    xbestsplit = Eigen::Map<Eigen::MatrixXd>(
        om.xbestsplit, om.n_xbestsplit[0], om.n_xbestsplit[1]);
    classwt = Eigen::Map<Eigen::MatrixXd>(
        om.classwt, om.n_classwt[0], om.n_classwt[1]);
    cutoff = Eigen::Map<Eigen::MatrixXd>(
        om.cutoff, om.n_cutoff[0], om.n_cutoff[1]);
    treemap = Eigen::Map<Eigen::MatrixXi>(
        om.treemap, om.n_treemap[0], om.n_treemap[1]);
    nodestatus = Eigen::Map<Eigen::MatrixXi>(
        om.nodestatus, om.n_nodestatus[0], om.n_nodestatus[1]);
    nodeclass = Eigen::Map<Eigen::MatrixXi>(
        om.nodeclass, om.n_nodeclass[0], om.n_nodeclass[1]);
    bestvar = Eigen::Map<Eigen::MatrixXi>(
        om.bestvar, om.n_bestvar[0], om.n_bestvar[1]);
    ndbigtree = Eigen::Map<Eigen::MatrixXi>(
        om.ndbigtree, om.n_ndbigtree[0], om.n_ndbigtree[1]);
    mtry = om.mtry;
    orig_labels.resize(om.n_orig_labels[0] * om.n_orig_labels[1]);
    std::memcpy(
        orig_labels.data(), om.orig_labels,
        sizeof(int) * om.n_orig_labels[0] * om.n_orig_labels[1]);
    new_labels.resize(om.n_new_labels[0] * om.n_new_labels[1]);
    std::memcpy(
        new_labels.data(), om.new_labels,
        sizeof(int) * om.n_new_labels[0] * om.n_new_labels[1]);
    nclass = om.nclass;
    outcl = Eigen::Map<Eigen::MatrixXi>(
        om.outcl, om.n_outcl[0], om.n_outcl[1]);
    outclts = Eigen::Map<Eigen::MatrixXi>(
        om.outclts, om.n_outclts[0], om.n_outclts[1]);
    counttr = Eigen::Map<Eigen::MatrixXi>(
        om.counttr, om.n_counttr[0], om.n_counttr[1]);
    proximity = Eigen::Map<Eigen::MatrixXd>(
        om.proximity, om.n_proximity[0], om.n_proximity[1]);
    proximity_tst = Eigen::Map<Eigen::MatrixXd>(
        om.proximity_tst, om.n_proximity_tst[0], om.n_proximity_tst[1]);
    localImp = Eigen::Map<Eigen::MatrixXd>(
        om.localImp, om.n_localImp[0], om.n_localImp[1]);
    importance = Eigen::Map<Eigen::MatrixXd>(
        om.importance, om.n_importance[0], om.n_importance[1]);
    importanceSD = Eigen::Map<Eigen::MatrixXd>(
        om.importanceSD, om.n_importanceSD[0], om.n_importanceSD[1]);
    errtr = Eigen::Map<Eigen::MatrixXd>(
        om.errtr, om.n_errtr[0], om.n_errtr[1]);
    errts = Eigen::Map<Eigen::MatrixXd>(
        om.errts, om.n_errts[0], om.n_errts[1]);
    inbag = Eigen::Map<Eigen::MatrixXi>(
        om.inbag, om.n_inbag[0], om.n_inbag[1]);
    votes = Eigen::Map<Eigen::MatrixXi>(
        om.votes, om.n_votes[0], om.n_votes[1]);
    oob_times = Eigen::Map<Eigen::MatrixXi>(
        om.oob_times, om.n_oob_times[0], om.n_oob_times[1]);
  }

  double predict (double* X, int D, int label) {
    helperPredict(
        _Y, _votes, _prediction_per_tree, _proximity_ts, _nodes, _cat,
        X, 1, D, PredictOptions());
    for (int i = 0; i < orig_labels.size(); ++i) {
      if (label == orig_labels[i])
      { return _votes(i) / (double)ntree; }
    }
    perr("Error: invalid label for random forest predictor");
    return -1.0;
  }

 protected:
  Eigen::MatrixXi _cat;
  Eigen::VectorXi _Y;
  Eigen::MatrixXd _votes;
  Eigen::MatrixXi _prediction_per_tree;
  Eigen::MatrixXd _proximity_ts;
  Eigen::MatrixXi _nodes;

  void helperPredict (
      Eigen::VectorXi& jet, Eigen::MatrixXi& jts, Eigen::MatrixXd& countts,
      Eigen::MatrixXd& proxMat, Eigen::MatrixXi& nodexts,
      double* X, int N, int D, int nrnodes, int ntree,
      Eigen::MatrixXd& xbestsplit, Eigen::MatrixXd& pid,
      Eigen::MatrixXd& cutoff, Eigen::MatrixXi& treemap,
      Eigen::MatrixXi& nodestatus, Eigen::MatrixXi& nodeclass,
      Eigen::MatrixXi& bestvar, Eigen::MatrixXi& ndbigtree,
      int nclass, int keepPred, int proximity, int nodes,
      Eigen::MatrixXi& cat, int maxcat) {
    if (nodes) { nodexts.setZero(N, ntree); }
    else { nodexts.setZero(N, 1); }
    if (proximity) { proxMat.setZero(N, N); }
    else { proxMat.setOnes(1, 1); }
    jet.setZero(N);
    countts.setZero(nclass, N);
    if (keepPred) { jts.setZero(N, ntree); } else { jts.setZero(N, 1); }
    classForest(
        &D, &N, &nclass, &maxcat, &nrnodes, &ntree, X, xbestsplit.data(),
        pid.data(), cutoff.data(), countts.data(), treemap.data(),
        nodestatus.data(), cat.data(), nodeclass.data(), jts.data(),
        jet.data(), bestvar.data(), nodexts.data(), ndbigtree.data(),
        &keepPred, &proximity, proxMat.data(), &nodes);
  }

  void helperPredict (
      Eigen::VectorXi& Y, Eigen::MatrixXd& votes,
      Eigen::MatrixXi& prediction_per_tree, Eigen::MatrixXd& proximity_ts,
      Eigen::MatrixXi& nodes, Eigen::MatrixXi& _cat, double* X, int N,
      int D, glia::ml::rf::PredictOptions const& options) {
    int keepPred_ = std::max(0, options.predict_all);
    int proximity_ = std::max(0, options.proximity);
    int node_ = std::max(0, options.nodes);
    if (!categorical_feature.empty()) {
      for (int i = 0; i < D; ++i) {
        if (categorical_feature[i]) {
          for (int j = 0; j < orig_uniques_in_feature.size(); ++j) {
            for (int k = 0; k < N; ++k) {
              if (X[k * D + i] == orig_uniques_in_feature[i][j])
              { X[k * D + i] = mapped_uniques_in_feature[i][j]; }
            }
          }
        }
      }
      _cat = cat;
    } else { _cat.setOnes(1, D); }
    int maxcat = _cat.maxCoeff();
    helperPredict(
        Y, prediction_per_tree, votes, proximity_ts, nodes, X, N, D,
        nrnodes, ntree, xbestsplit, classwt, cutoff, treemap, nodestatus,
        nodeclass, bestvar, ndbigtree, nclass, keepPred_, proximity_,
        node_, _cat, maxcat);
    for (int j = 0; j < Y.size(); ++j) {
      int& y = Y[j];
      for (int i = 0; i < orig_labels.size(); ++i)
      { if (y == orig_labels[i]) { y = new_labels[i]; } }
    }
  }

};

};
};
};

#endif

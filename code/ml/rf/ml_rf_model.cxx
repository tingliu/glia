#include "ml/rf/ml_rf.h"
using namespace rf_old;

const int MIN_SPARSE_SIZE = 128;

template <typename T>
void writeArray (std::ofstream& fs, T const* data, int size)
{
  if (size > 0) {
    if (size > MIN_SPARSE_SIZE) {
      std::list<int> indices;
      for (int i = 0; i < size; i++) {
	T d = data[i];
	if (fabs((double)d) > 1e-8) indices.push_back(i);
      }
      // If more than half is zero, then write sparsely
      bool isSparse = indices.size() < size / 2;
      fs.write((char*)&isSparse, sizeof(bool));
      if (isSparse) {
	int s = indices.size();
	fs.write((char*)&s, sizeof(int));
	for (std::list<int>::const_iterator itr = indices.begin();
	     itr != indices.end(); itr++) {
	  fs.write((char*)&(*itr), sizeof(int));
	  fs.write((char*)(data + *itr), sizeof(T));
	}
	// // For debug
	// std::cerr << "Write sparsely ("
	// 	  << (float)indices.size() / (float)size
	// 	  << ")..." << std::endl;
	// // ~ For debug
      }
      else {
	fs.write((char*)data, size * sizeof(T));
	// // For debug
	// std::cerr << "Write non-sparsely ("
	// 	  << (float)indices.size() / (float)size
	// 	  << ")..." << std::endl;
	// // ~ For debug
      }
    }
    else fs.write((char*)data, size * sizeof(T));
  }
}



template<typename T>
void readArray (T*& data, std::ifstream& fs, int size)
{
  if (size > 0) {
    data = new T[size]();
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
      }
      else fs.read((char*)data, size * sizeof(T));
    }
    else fs.read((char*)data, size * sizeof(T));
  }
  else data = NULL;
}


void rf_old::deleteModel (Model& model)
{
  for (std::vector<double*>::iterator itr =
	 model.orig_uniques_in_feature.begin();
       itr != model.orig_uniques_in_feature.end(); itr++) {
    del(&(*itr));
  }
  model.orig_uniques_in_feature.clear();
  model.n_orig_uniques_in_feature.clear();
  for (std::vector<int*>::iterator itr =
	 model.mapped_uniques_in_feature.begin();
       itr != model.mapped_uniques_in_feature.end(); itr++) {
    del(&(*itr));
  }
  model.mapped_uniques_in_feature.clear();
  model.n_mapped_uniques_in_feature.clear();
  del(&model.ncat);
  model.n_ncat[0] = model.n_ncat[1] = 0;
  del(&model.categorical_feature);
  model.n_categorical_feature[0] = model.n_categorical_feature[1] = 0;
  del(&model.xbestsplit);
  model.n_xbestsplit[0] = model.n_xbestsplit[1] = 0;
  del(&model.classwt);
  model.n_classwt[0] = model.n_classwt[1] = 0;
  del(&model.cutoff);
  model.n_cutoff[0] = model.n_cutoff[1] = 0;
  del(&model.treemap);
  model.n_treemap[0] = model.n_treemap[1] = 0;
  del(&model.nodestatus);
  model.n_nodestatus[0] = model.n_nodestatus[1] = 0;
  del(&model.nodeclass);
  model.n_nodeclass[0] = model.n_nodeclass[1] = 0;
  del(&model.bestvar);
  model.n_bestvar[0] = model.n_bestvar[1] = 0;
  del(&model.ndbigtree);
  model.n_ndbigtree[0] = model.n_ndbigtree[1] = 0;
  del(&model.orig_labels);
  model.n_orig_labels[0] = model.n_orig_labels[1] = 0;
  del(&model.new_labels);
  model.n_new_labels[0] = model.n_new_labels[1] = 0;
  del(&model.outcl);
  model.n_outcl[0] = model.n_outcl[1] = 0;
  del(&model.outclts);
  model.n_outclts[0] = model.n_outclts[1] = 0;
  del(&model.counttr);
  model.n_counttr[0] = model.n_counttr[1] = 0;
  del(&model.proximity);
  model.n_proximity[0] = model.n_proximity[1] = 0;
  del(&model.proximity_tst);
  model.n_proximity_tst[0] = model.n_proximity_tst[1] = 0;
  del(&model.localImp);
  model.n_localImp[0] = model.n_localImp[1] = 0;
  del(&model.importance);
  model.n_importance[0] = model.n_importance[1] = 0;
  del(&model.importanceSD);
  model.n_importanceSD[0] = model.n_importanceSD[1] = 0;
  del(&model.errtr);
  model.n_errtr[0] = model.n_errtr[1] = 0;
  del(&model.errts);
  model.n_errts[0] = model.n_errts[1] = 0;
  del(&model.inbag);
  model.n_inbag[0] = model.n_inbag[1] = 0;
  del(&model.votes);
  model.n_votes[0] = model.n_votes[1] = 0;
  del(&model.oob_times);
  model.n_oob_times[0] = model.n_oob_times[1] = 0;
}



void rf_old::printModel (Model const& model)
{
  std::cout << "ncat: " << std::endl;
  std::cout << "\tsize = [" << model.n_ncat[0] << " "
	    << model.n_ncat[1] << "]\n\t";
  // print_vector(model.ncat, model.n_ncat[0] * model.n_ncat[1], true);
  if (model.n_ncat[0] * model.n_ncat[1] >= 2) {
    print_vector(model.ncat, 2, true);
  }
  std::cout << std::endl;
  std::cout << "categorical_feature: " << std::endl;
  std::cout << "\tsize = [" << model.n_categorical_feature[0] << " "
	    << model.n_categorical_feature[1] << "]\n\t";
  // print_vector(model.ncat,
  // 	       model.n_categorical_feature[0]
  // 	       * model.n_categorical_feature[1], true);
  if (model.n_categorical_feature[0]
      * model.n_categorical_feature[1] >= 2) {
    print_vector(model.categorical_feature, 2, true);
  }
  std::cout << std::endl;
  std::cout << "nrnodes = " << model.nrnodes << std::endl;
  std::cout << std::endl;
  std::cout << "ntree = " << model.ntree << std::endl;
  std::cout << std::endl;
  std::cout << "xbestsplit: " << std::endl;
  std::cout << "\tsize = [" << model.n_xbestsplit[0] << " "
	    << model.n_xbestsplit[1] << "]\n\t";
  // print_vector(model.xbestsplit,
  // 	       model.n_xbestsplit[0] * model.n_xbestsplit[1], true);
  if (model.n_xbestsplit[0] * model.n_xbestsplit[1] >= 2) {
    print_vector(model.xbestsplit, 2, true);
  }
  std::cout << std::endl;
  std::cout << "classwt:" << std::endl;
  std::cout << "\tsize = [" << model.n_classwt[0] << " "
	    << model.n_classwt[1] << "]\n\t";
  // print_vector(model.classwt,
  // 	       model.n_classwt[0] * model.n_classwt[1], true);
  if (model.n_classwt[0] * model.n_classwt[1] >= 2) {
    print_vector(model.classwt, 2, true);
  }
  std::cout << std::endl;
  std::cout << "cutoff:" << std::endl;
  std::cout << "\tsize = [" << model.n_cutoff[0] << " "
	    << model.n_cutoff[1] << "]\n\t";
  // print_vector(model.cutoff,
  // 	       model.n_cutoff[0] * model.n_cutoff[1], true);
  if (model.n_cutoff[0] * model.n_cutoff[1] >= 2) {
    print_vector(model.cutoff, 2, true);
  }
  std::cout << std::endl;
  std::cout << "treemap:" << std::endl;
  std::cout << "\tsize = [" << model.n_treemap[0] << " "
	    << model.n_treemap[1] << "]\n\t";
  // print_vector(model.treemap,
  // 	       model.n_treemap[0] * model.n_treemap[1], true);
  if (model.n_treemap[0] * model.n_treemap[1] >= 2) {
    print_vector(model.treemap, 2, true);
  }
  std::cout << std::endl;
  std::cout << "nodestatus:" << std::endl;
  std::cout << "\tsize = [" << model.n_nodestatus[0] << " "
	    << model.n_nodestatus[1] << "]\n\t";
  // print_vector(model.nodestatus,
  // 	       model.n_nodestatus[0] * model.n_nodestatus[1], true);
  if (model.n_nodestatus[0] * model.n_nodestatus[1] >= 2) {
    print_vector(model.nodestatus, 2, true);
  }
  std::cout << std::endl;
  std::cout << "nodeclass:" << std::endl;
  std::cout << "\tsize = [" << model.n_nodeclass[0] << " "
	    << model.n_nodeclass[1] << "]\n\t";
  // print_vector(model.nodeclass,
  // 	       model.n_nodeclass[0] * model.n_nodeclass[1], true);
  if (model.n_nodeclass[0] * model.n_nodeclass[1] >= 2) {
    print_vector(model.nodeclass, 2, true);
  }
  std::cout << std::endl;
  std::cout << "bestvar:" << std::endl;
  std::cout << "\tsize = [" << model.n_bestvar[0] << " "
	    << model.n_bestvar[1] << "]\n\t";
  // print_vector(model.bestvar,
  // 	       model.n_bestvar[0] * model.n_bestvar[1], true);
  if (model.n_bestvar[0] * model.n_bestvar[1] >= 2) {
    print_vector(model.bestvar, 2, true);
  }
  std::cout << std::endl;
  std::cout << "ndbigtree:" << std::endl;
  std::cout << "\tsize = [" << model.n_ndbigtree[0] << " "
	    << model.n_ndbigtree[1] << "]\n\t";
  // print_vector(model.ndbigtree,
  // 	       model.n_ndbigtree[0] * model.n_ndbigtree[1], true);
  if (model.n_ndbigtree[0] * model.n_ndbigtree[1] >= 2) {
    print_vector(model.ndbigtree, 2, true);
  }
  std::cout << std::endl;
  std::cout << "mtry = " << model.mtry << std::endl;
  std::cout << std::endl;
  std::cout << "orig_labels:" << std::endl;
  std::cout << "\tsize = [" << model.n_orig_labels[0] << " "
	    << model.n_orig_labels[1] << "]\n\t";
  // print_vector(model.orig_labels,
  // 	       model.n_orig_labels[0] * model.n_orig_labels[1], true);
  if (model.n_orig_labels[0] * model.n_orig_labels[1] >= 2) {
    print_vector(model.orig_labels, 2, true);
  }
  std::cout << std::endl;
  std::cout << "new_labels:" << std::endl;
  std::cout << "\tsize = [" << model.n_new_labels[0] << " "
	    << model.n_new_labels[1] << "]\n\t";
  // print_vector(model.new_labels,
  // 	       model.n_new_labels[0] * model.n_new_labels[1], true);
  if (model.n_new_labels[0] * model.n_new_labels[1] >= 2) {
    print_vector(model.new_labels, 2, true);
  }
  std::cout << std::endl;
  std::cout << "nclass = " << model.nclass << std::endl;
  std::cout << std::endl;
  std::cout << "outcl:" << std::endl;
  std::cout << "\tsize = [" << model.n_outcl[0] << " "
	    << model.n_outcl[1] << "]\n\t";
  // print_vector(model.outcl, model.n_outcl[0] * model.n_outcl[1], true);
  if (model.n_outcl[0] * model.n_outcl[1] >= 2) {
    print_vector(model.outcl, 2, true);
  }
  std::cout << std::endl;
  std::cout << "outclts:" << std::endl;
  std::cout << "\tsize = [" << model.n_outclts[0] << " "
	    << model.n_outclts[1] << "]\n\t";
  // print_vector(model.outclts,
  // 	       model.n_outclts[0] * model.n_outclts[1], true);
  if (model.n_outclts[0] * model.n_outclts[1] >= 2) {
    print_vector(model.outclts, 2, true);
  }
  std::cout << std::endl;
  std::cout << "counttr:" << std::endl;
  std::cout << "\tsize = [" << model.n_counttr[0] << " "
	    << model.n_counttr[1] << "]\n\t";
  // print_vector(model.counttr,
  // 	       model.n_counttr[0] * model.n_counttr[1], true);
  if (model.n_counttr[0] * model.n_counttr[1] >= 2) {
    print_vector(model.counttr, 2, true);
  }
  std::cout << std::endl;
  std::cout << "proximity:" << std::endl;
  std::cout << "\tsize = [" << model.n_proximity[0] << " "
	    << model.n_proximity[1] << "]\n\t";
  // print_vector(model.proximity,
  // 	       model.n_proximity[0] * model.n_proximity[1], true);
  if (model.n_proximity[0] * model.n_proximity[1] >= 2) {
    print_vector(model.proximity, 2, true);
  }
  std::cout << std::endl;
  std::cout << "proximity_tst:" << std::endl;
  std::cout << "\tsize = [" << model.n_proximity_tst[0] << " "
	    << model.n_proximity_tst[1] << "]\n\t";
  // print_vector(model.proximity_tst,
  // 	       model.n_proximity_tst[0] * model.n_proximity_tst[1], true);
  if (model.n_proximity_tst[0] * model.n_proximity_tst[1] >= 2) {
    print_vector(model.proximity_tst, 2, true);
  }
  std::cout << std::endl;
  std::cout << "localImp:" << std::endl;
  std::cout << "\tsize = [" << model.n_localImp[0] << " "
	    << model.n_localImp[1] << "]\n\t";
  // print_vector(model.localImp,
  // 	       model.n_localImp[0] * model.n_localImp[1], true);
  if (model.n_localImp[0] * model.n_localImp[1] >= 2) {
    print_vector(model.localImp, 2, true);
  }
  std::cout << std::endl;
  std::cout << "importance:" << std::endl;
  std::cout << "\tsize = [" << model.n_importance[0] << " "
	    << model.n_importance[1] << "]\n\t";
  // print_vector(model.importance,
  // 	       model.n_importance[0] * model.n_importance[1], true);
  if (model.n_importance[0] * model.n_importance[1] >= 2) {
    print_vector(model.importance, 2, true);
  }
  std::cout << std::endl;
  std::cout << "importanceSD:" << std::endl;
  std::cout << "\tsize = [" << model.n_importanceSD[0] << " "
	    << model.n_importanceSD[1] << "]\n\t";
  // print_vector(model.importanceSD,
  // 	       model.n_importanceSD[0] * model.n_importanceSD[1], true);
  if (model.n_importanceSD[0] * model.n_importanceSD[1] >= 2) {
    print_vector(model.importanceSD, 2, true);
  }
  std::cout << std::endl;
  std::cout << "errtr:" << std::endl;
  std::cout << "\tsize = [" << model.n_errtr[0] << " "
	    << model.n_errtr[1] << "]\n\t";
  // print_vector(model.errtr, model.n_errtr[0] * model.n_errtr[1], true);
  if (model.n_errtr[0] * model.n_errtr[1] >= 2) {
    print_vector(model.errtr, 2, true);
  }
  std::cout << std::endl;
  std::cout << "errts:" << std::endl;
  std::cout << "\tsize = [" << model.n_errts[0] << " "
	    << model.n_errts[1] << "]\n\t";
  // print_vector(model.errts, model.n_errts[0] * model.n_errts[1], true);
  if (model.n_errts[0] * model.n_errts[1] >= 2) {
    print_vector(model.errts, 2, true);
  }
  std::cout << std::endl;
  std::cout << "inbag:" << std::endl;
  std::cout << "\tsize = [" << model.n_inbag[0] << " "
	    << model.n_inbag[1] << "]\n\t";
  // print_vector(model.inbag, model.n_inbag[0] * model.n_inbag[1], true);
  if (model.n_inbag[0] * model.n_inbag[1] >= 2) {
    print_vector(model.inbag, 2, true);
  }
  std::cout << std::endl;
  std::cout << "votes:" << std::endl;
  std::cout << "\tsize = [" << model.n_votes[0] << " "
	    << model.n_votes[1] << "]\n\t";
  // print_vector(model.votes, model.n_votes[0] * model.n_votes[1], true);
  if (model.n_votes[0] * model.n_votes[1] >= 2) {
    print_vector(model.votes, 2, true);
  }
  std::cout << std::endl;
  std::cout << "oob_times: " << std::endl;
  std::cout << "\tsize = [" << model.n_oob_times[0] << " "
	    << model.n_oob_times[1] << "]\n\t";
  // print_vector(model.oob_times,
  // 	       model.n_oob_times[0] * model.n_oob_times[1], true);
  if (model.n_oob_times[0] * model.n_oob_times[1] >= 2) {
    print_vector(model.oob_times, 2, true);
  }
  std::cout << std::endl;
}



void rf_old::writeModelToBinaryFile (
    const char* fileName, Model const& model)
{
  std::ofstream fs(fileName, std::ios::binary);
  if (fs.is_open()) {
    fs.write((char*)&model, sizeof(model));
    if (model.n_orig_uniques_in_feature.size() > 0) {
      fs.write((char*)&model.n_orig_uniques_in_feature.front(),
	       sizeof(int) * model.n_orig_uniques_in_feature.size());
      for (int i = 0; i < model.n_orig_uniques_in_feature.size(); i++) {
	writeArray(fs, model.orig_uniques_in_feature[i],
		   model.n_orig_uniques_in_feature[i]);
      }
    }
    if (model.n_mapped_uniques_in_feature.size() > 0) {
      fs.write((char*)&model.n_mapped_uniques_in_feature.front(),
	       sizeof(int) * model.n_mapped_uniques_in_feature.size());
      for (int i = 0; i < model.n_mapped_uniques_in_feature.size();
	   i++) {
	writeArray(fs, model.mapped_uniques_in_feature[i],
		   model.n_mapped_uniques_in_feature[i]);
      }
    }
    writeArray(fs, model.ncat, model.n_ncat[0] * model.n_ncat[1]);
    writeArray(fs, model.categorical_feature,
	       model.n_categorical_feature[0]
	       * model.n_categorical_feature[1]);
    fs.write((char*)&model.nrnodes, sizeof(int));
    fs.write((char*)&model.ntree, sizeof(int));
    writeArray(fs, model.xbestsplit, model.n_xbestsplit[0]
	       * model.n_xbestsplit[1]);
    writeArray(fs, model.classwt, model.n_classwt[0]
	       * model.n_classwt[1]);
    writeArray(fs, model.cutoff, model.n_cutoff[0] * model.n_cutoff[1]);
    writeArray(fs, model.treemap, model.n_treemap[0]
	       * model.n_treemap[1]);
    writeArray(fs, model.nodestatus, model.n_nodestatus[0]
	       * model.n_nodestatus[1]);
    writeArray(fs, model.nodeclass, model.n_nodeclass[0]
	       * model.n_nodeclass[1]);
    writeArray(fs, model.bestvar, model.n_bestvar[0]
	       * model.n_bestvar[1]);
    writeArray(fs, model.ndbigtree, model.n_ndbigtree[0]
	       * model.n_ndbigtree[1]);
    fs.write((char*)&model.mtry, sizeof(int));
    writeArray(fs, model.orig_labels, model.n_orig_labels[0]
	       * model.n_orig_labels[1]);
    writeArray(fs, model.new_labels, model.n_new_labels[0]
	       * model.n_new_labels[1]);
    fs.write((char*)&model.nclass, sizeof(int));
    writeArray(fs, model.outcl, model.n_outcl[0] * model.n_outcl[1]);
    writeArray(fs, model.outclts, model.n_outclts[0]
	       * model.n_outclts[1]);
    writeArray(fs, model.counttr, model.n_counttr[0]
	       * model.n_counttr[1]);
    writeArray(fs, model.proximity, model.n_proximity[0]
	       * model.n_proximity[1]);
    writeArray(fs, model.proximity_tst, model.n_proximity_tst[0]
	       * model.n_proximity_tst[1]);
    writeArray(fs, model.localImp, model.n_localImp[0]
	       * model.n_localImp[1]);
    writeArray(fs, model.importance, model.n_importance[0]
	       * model.n_importance[1]);
    writeArray(fs, model.importanceSD, model.n_importanceSD[0]
	       * model.n_importanceSD[1]);
    writeArray(fs, model.errtr, model.n_errtr[0] * model.n_errtr[1]);
    writeArray(fs, model.errts, model.n_errts[0] * model.n_errts[1]);
    writeArray(fs, model.inbag, model.n_inbag[0] * model.n_inbag[1]);
    writeArray(fs, model.votes, model.n_votes[0] * model.n_votes[1]);
    writeArray(fs, model.oob_times, model.n_oob_times[0]
	       * model.n_oob_times[1]);
    fs.close();
  }
  else {
    std::cerr << "Error creating model file..." << std::endl;
    exit(EXIT_FAILURE);
  }
}



void rf_old::readModelFromBinaryFile (Model& model, const char* fileName)
{
  std::ifstream fs(fileName, std::ios::binary);
  if (fs.is_open()) {
    fs.read((char*)&model, sizeof(Model));
    int size = model.n_orig_uniques_in_feature.size();
    if (size > 0) {
      model.n_orig_uniques_in_feature.resize(size);
      fs.read((char*)&model.n_orig_uniques_in_feature.front(),
	      sizeof(int) * size);
      for (int i = 0; i < size; i++) {
	readArray(model.orig_uniques_in_feature[i],
		  fs, model.n_orig_uniques_in_feature[i]);
      }
    }
    else {
      model.n_orig_uniques_in_feature.clear();
      model.orig_uniques_in_feature.clear();
    }
    size = model.n_mapped_uniques_in_feature.size();
    if (size > 0) {
      model.n_mapped_uniques_in_feature.resize(size);
      fs.read((char*)&model.n_mapped_uniques_in_feature.front(),
	      sizeof(int) * size);
      for (int i = 0; i < size; i++) {
	readArray(model.mapped_uniques_in_feature[i],
		  fs, model.n_mapped_uniques_in_feature[i]);
      }
    }
    else {
      model.n_mapped_uniques_in_feature.clear();
      model.mapped_uniques_in_feature.clear();
    }
    readArray(model.ncat, fs, model.n_ncat[0] * model.n_ncat[1]);
    readArray(model.categorical_feature, fs,
	      model.n_categorical_feature[0]
	      * model.n_categorical_feature[1]);
    fs.read((char*)&model.nrnodes, sizeof(int));
    fs.read((char*)&model.ntree, sizeof(int));
    readArray(model.xbestsplit, fs, model.n_xbestsplit[0]
	      * model.n_xbestsplit[1]);
    readArray(model.classwt, fs, model.n_classwt[0]
	      * model.n_classwt[1]);
    readArray(model.cutoff, fs, model.n_cutoff[0] * model.n_cutoff[1]);
    readArray(model.treemap, fs, model.n_treemap[0]
	      * model.n_treemap[1]);
    readArray(model.nodestatus, fs, model.n_nodestatus[0]
	      * model.n_nodestatus[1]);
    readArray(model.nodeclass, fs, model.n_nodeclass[0]
	      * model.n_nodeclass[1]);
    readArray(model.bestvar, fs, model.n_bestvar[0]
	      * model.n_bestvar[1]);
    readArray(model.ndbigtree, fs, model.n_ndbigtree[0]
	      * model.n_ndbigtree[1]);
    fs.read((char*)&model.mtry, sizeof(int));
    readArray(model.orig_labels, fs, model.n_orig_labels[0]
	      * model.n_orig_labels[1]);
    readArray(model.new_labels, fs, model.n_new_labels[0]
	      * model.n_new_labels[1]);
    fs.read((char*)&model.nclass, sizeof(int));
    readArray(model.outcl, fs, model.n_outcl[0] * model.n_outcl[1]);
    readArray(model.outclts, fs, model.n_outclts[0]
	      * model.n_outclts[1]);
    readArray(model.counttr, fs, model.n_counttr[0]
	      * model.n_counttr[1]);
    readArray(model.proximity, fs, model.n_proximity[0]
	      * model.n_proximity[1]);
    readArray(model.proximity_tst, fs, model.n_proximity_tst[0]
	      * model.n_proximity_tst[1]);
    readArray(model.localImp, fs, model.n_localImp[0]
	      * model.n_localImp[1]);
    readArray(model.importance, fs, model.n_importance[0]
	      * model.n_importance[1]);
    readArray(model.importanceSD, fs, model.n_importanceSD[0]
	      * model.n_importanceSD[1]);
    readArray(model.errtr, fs, model.n_errtr[0] * model.n_errtr[1]);
    readArray(model.errts, fs, model.n_errts[0] * model.n_errts[1]);
    readArray(model.inbag, fs, model.n_inbag[0] * model.n_inbag[1]);
    readArray(model.votes, fs, model.n_votes[0] * model.n_votes[1]);
    readArray(model.oob_times, fs, model.n_oob_times[0]
	      * model.n_oob_times[1]);
    fs.close();
    // Transpose some matrices
    model.xbestsplit = transpose(model.xbestsplit, model.n_xbestsplit[0],
				 model.n_xbestsplit[1]);
    model.classwt = transpose(model.classwt, model.n_classwt[0],
			      model.n_classwt[1]);
    model.cutoff = transpose(model.cutoff, model.n_cutoff[0],
			     model.n_cutoff[1]);
    model.treemap = transpose(model.treemap, model.n_treemap[0],
			      model.n_treemap[1]);
    model.nodestatus = transpose(model.nodestatus, model.n_nodestatus[0],
				 model.n_nodestatus[1]);
    model.nodeclass = transpose(model.nodeclass, model.n_nodeclass[0],
				model.n_nodeclass[1]);
    model.bestvar = transpose(model.bestvar, model.n_bestvar[0],
			      model.n_bestvar[1]);
    model.ndbigtree = transpose(model.ndbigtree, model.n_ndbigtree[0],
				model.n_ndbigtree[1]);
  }
  else {
    std::cerr << "Error reading model file..." << std::endl;
    exit(EXIT_FAILURE);
  }
}

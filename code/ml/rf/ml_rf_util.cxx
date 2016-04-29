#include "ml/rf/ml_rf.h"
using namespace rf_old;

void* rf_old::getData (void* p)
{
  return p;
}



double* rf_old::getPr (void* p)
{
  return (double*)p;
}



void rf_old::sort_unique (int*& y, int& n_y, int* x, int n_x, int n_x_inc)
{
  std::set<int> s;
  for (int i = 0; i < n_x; i += n_x_inc) s.insert(x[i]);
  n_y = s.size();
  y = new int[n_y];
  int j = 0;
  for (std::set<int>::const_iterator sitr = s.begin();
       sitr != s.end(); sitr++) y[j++] = *sitr;
}



void rf_old::sort_unique (double*& y, int& n_y, double* x, int n_x, int n_x_inc)
{
  std::set<double> s;
  for (int i = 0; i < n_x; i += n_x_inc) s.insert(x[i]);
  n_y = s.size();
  y = new double[n_y];
  int j = 0;
  for (std::set<double>::const_iterator sitr = s.begin();
       sitr != s.end(); sitr++) y[j++] = *sitr;
}



int getSum (std::list<int> const& data)
{
  int ret = 0.0;
  for (std::list<int>::const_iterator itr = data.begin();
       itr != data.end(); itr++) {
    ret += *itr;
  }
  return ret;
}



// All matrices must have same column number
void concatMatrices (double*& matrix, int& rowNum,
		     std::list<double*> const& matrices,
		     std::list<int> const& rowNums, int colNum)
{
  matrix = NULL;
  rowNum = getSum(rowNums);
  if (rowNum > 0) {
    matrix = new double[rowNum * colNum];
    std::list<double*>::const_iterator mitr = matrices.begin();
    std::list<int>::const_iterator ritr = rowNums.begin();
    double* index = matrix;
    while (mitr != matrices.end()) {
      if (*ritr > 0) {
	int length = *ritr * colNum;
	memcpy(index, *mitr, length * sizeof(double));
	// std::copy(*mitr, *mitr + length, index);
	index += length;
      }
      mitr++;
      ritr++;
    }
  }
  else rowNum = 0;
}



int getFileRowNum (const char* fileName)
{
  std::ifstream fs(fileName);
  if (fs.is_open()) {
    int ret = 0;
    std::string str;
    while (fs.good()) {
      if (!getline(fs, str)) break;
      ret++;
    }
    fs.close();
    return ret;
  }
  else {
    std::cerr << "Error reading file..." << std::endl;
    exit(EXIT_FAILURE);
  }
}



// Get column number of first row
// Columns are separated with space
int getFileColumnNum (const char* fileName)
{
  std::ifstream fs(fileName);
  if (fs.is_open()) {
    int ret = 0;
    std::string str;
    if (!getline(fs, str)) return 0;
    std::stringstream ss(str);
    std::string t;
    while (ss >> t) ret++;
    fs.close();
    return ret;
  }
  else {
    std::cerr << "Error reading file..." << std::endl;
    exit(EXIT_FAILURE);
  }
}



// Caution: will allocate memory for matrix
void rf_old::readMatrixFromFile (double*& matrix, int& rowNum, int& colNum,
			     const char* fileName)
{
  std::ifstream fs(fileName);
  if (fs.is_open()) {
    rowNum = getFileRowNum(fileName);
    if (rowNum <= 0) {
      matrix = NULL;
      rowNum = 0;
      colNum = 0;
      return;
    }
    colNum = getFileColumnNum(fileName);
    matrix = new double[rowNum * colNum];
    for (int r = 0; r < rowNum; r++) {
      std::string str;
      if (!getline(fs, str)) {
	std::cerr << "Error: inconsistent matrix row number... "
		  << std::endl;
	exit(EXIT_FAILURE);
      }
      std::stringstream ss(str);
      for (int c = 0; c < colNum; c++) {
	if (!(ss >> matrix[r * colNum + c])) {
	  std::cerr << "Error: inconsistent matrix column number... "
		    << std::endl;
	  exit(EXIT_FAILURE);
	}
      }
    }
  }
  else {
    std::cerr << "Error reading matrix file..." << std::endl;
    exit(EXIT_FAILURE);
  }
}



void rf_old::readMatrixFromFiles (double*& matrix, int& rowNum, int& colNum,
			      std::vector<const char*> const& fileNames)
{
  std::list<double*> matrices;
  std::list<int> rowNums;
  int tmpColNum;
  bool isFirstTime = true;
  for (std::vector<const char*>::const_iterator fitr = fileNames.begin();
       fitr != fileNames.end(); fitr++) {
    double* m;
    int r, c;
    readMatrixFromFile(m, r, c, *fitr);
    if (isFirstTime) {
      isFirstTime = false;
      tmpColNum = c;
    }
    else if (c > 0 && c != tmpColNum) {
      std::cerr << "Error: inconsistent matrix column number..."
		<< std::endl;
      exit(EXIT_FAILURE);
    }
    matrices.push_back(m);
    rowNums.push_back(r);
  }
  colNum = tmpColNum;
  concatMatrices(matrix, rowNum, matrices, rowNums, colNum);
  for (std::list<double*>::iterator itr = matrices.begin();
       itr != matrices.end(); itr++) {
    delete[] *itr;
  }
}



void rf_old::readMatrixFromFiles (int*& matrix, int& rowNum, int& colNum,
			      std::vector<const char*> const& fileNames)
{
  double* m;
  readMatrixFromFiles(m, rowNum, colNum, fileNames);
  int n = rowNum * colNum;
  matrix = new int[n];
  for (int i = 0; i < n; i++) {
    if (m[i] >= 0) matrix[i] = (int)(m[i] + 0.5);
    else matrix[i] = (int)(m[i] - 0.5);
  }
  delete[] m;
}



void rf_old::countLabel (std::map<int, int>& labelCount, int* Y, int N)
{
  for (int i = 0; i < N; i++) {
    if (labelCount.count(Y[i]) > 0) labelCount[Y[i]]++;
    else labelCount[Y[i]] = 1;
  }
}

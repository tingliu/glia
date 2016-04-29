#ifndef _glia_util_text_io_hxx_
#define _glia_util_text_io_hxx_

#include "glia_base.hxx"

namespace glia {

template <typename... Args> inline std::string
prints (std::string const& pattern, Args... args) {
  const int maxBufferSize = 1024;
  char buffer[maxBufferSize];
  snprintf(buffer, maxBufferSize, pattern.c_str(), args...);
  return std::string(buffer);
}


template <typename T, typename U> inline std::ostream&
operator<< (std::ostream& os, std::pair<T, U> const& pa);

template <typename T, typename U> inline std::istream&
operator>> (std::istream& is, std::pair<T, U>& pa);

template <typename T> std::ostream&
operator<< (std::ostream& os, std::vector<T> const& data);

template <typename T> std::istream&
operator>> (std::istream& is, std::vector<T>& data);


template <typename T, typename U> inline std::ostream&
operator<< (std::ostream& os, std::pair<T, U> const& pa)
{ return os << pa.first << " " << pa.second; }


template <typename T, typename U> inline std::istream&
operator>> (std::istream& is, std::pair<T, U>& pa)
{ return is >> pa.first >> pa.second; }


template <typename T> std::ostream&
operator<< (std::ostream& os, std::vector<T> const& data)
{
  os << data.size() << " ";
  for (auto const& x: data) { os << x << " "; }
  return os;
}


template <typename T> std::istream&
operator>> (std::istream& is, std::vector<T>& data)
{
  int n;
  is >> n;
  data.reserve(data.size() + n);
  for (int i = 0; i < n; ++i) {
    data.emplace_back();
    is >> data.back();
  }
  return is;
}


template <typename T, typename Func> void
writer (std::string const& file, T data, Func f, int precision = -1)
{
  std::ofstream os(file);
  if (os) {
    if (precision > 0) { os.precision(precision); }
    f(os, data);
    os.close();
  }
  else { perr(std::string("Error: cannot create file ").append(file)); }
}


template <typename TContainer, typename Func> void
writer (std::string const& file, TContainer const& data, Func f,
        std::string const& delim, int precision = -1)
{
  writer(file, data, [&delim, &f]
         (std::ofstream& os, TContainer const& data)
         {
           for (auto const& x: data) {
             f(os, x);
             os << delim;
           }
         }, precision);
}


template <typename T, typename Func> void
reader (T& data, std::string const& file, Func f)
{
  std::ifstream is(file);
  if (is) {
    f(data, is);
    is.close();
  }
  else { perr(std::string("Error: cannot open file ").append(file)); }
}


template <typename T> void
writeData (std::string const& file, std::vector<T> const& data,
           std::string const& delim, int precision = -1)
{
  writer(file, data, [&delim, precision]
         (std::ofstream& os, T const& x){ os << x; },
         delim, precision);
}


template <typename T> void
writeData (std::string const& file, std::vector<T const*> const& data,
           std::string const& delim, int precision = -1)
{
  writer(file, data, [&delim, precision]
         (std::ofstream& os, T const* p){ os << *p; },
         delim, precision);
}


template <typename T> void
writeData (std::string const& file,
           std::vector<std::vector<T>> const& data,
           std::string const& cdelim, std::string const& rdelim,
           int precision = -1)
{
  writer(file, data, [&cdelim]
         (std::ofstream& os, std::vector<T> const& row)
         { for (auto const& x: row) { os << x << cdelim; } },
         rdelim, precision);
}


template <typename T> void
writeData (std::string const& file,
           std::vector<std::vector<T> const*> const& data,
           std::string const& cdelim, std::string const& rdelim,
           int precision = -1)
{
  writer(file, data, [&cdelim]
         (std::ofstream& os, std::vector<T> const* pRow)
         { for (auto const& x: *pRow) { os << x << cdelim; } },
         rdelim, precision);
}


template <typename T = bool> std::pair<int, int>
getFileSize (std::string const& file)
{
  std::pair<int, int> ret = std::make_pair(-1, -1);
  std::string line;
  std::ifstream is(file);
  if (is) {
    ret.first = 0;
    ret.second = 0;
    if (std::getline(is, line)) {
      std::stringstream ss(line);
      std::string t;
      while (ss >> t) { ++ret.second; }
      do { ++ret.first; } while (std::getline(is, line));
    }
    is.close();
  }
  else { perr(std::string("Error: cannot open file ").append(file)); }
  return ret;
}


template <typename T = bool> std::pair<int, int>
getFileSize (std::vector<std::string> const& files)
{
  std::pair<int, int> size = std::make_pair(0, 0);
  for (auto const& file: files) {
    auto _sz = getFileSize(file);
    if (_sz.first < 0 || _sz.second < 0) {
      perr(std::string("Error: invalid data file dimension in ")
           .append(file));
    }
    if (size.second == 0) { size.second = _sz.second; }
    else if (_sz.second > 0 && size.second != _sz.second) {
      perr(std::string
           ("Error: inconsistent data file column number in ")
           .append(file));
    }
    size.first += _sz.first;
  }
  return size;
}


template <typename T> void
readData (std::vector<T>& data, std::pair<int, int>& size,
          std::string const& file, bool isFile1D)
{
  size = getFileSize(file);
  if (size.first >= 0 && size.second >= 0) {
    std::ifstream is(file);
    uint n = isFile1D? (uint)size.first:
        (uint)size.first * (uint)size.second;
    data.reserve(data.size() + n);
    for (uint i = 0; i < n; ++i) {
      data.emplace_back();
      is >> data.back();
    }
    is.close();
  }
  else {
    perr(std::string("Error: invalid data file dimension in ")
         .append(file));
  }
}


template <typename T> void
readData (std::vector<T>& data, std::string const& file, bool isFile1D)
{
  std::pair<int, int> size;
  readData(data, size, file, isFile1D);
}


template <typename T> void
readData (std::vector<T>& data, std::pair<int, int>& size,
          std::vector<std::string> const& files, bool isFile1D)
{
  size = getFileSize(files);
  uint n = isFile1D? (uint)size.first:
      (uint)size.first * (uint)size.second;
  data.reserve(n);
  for (auto const& file: files) { readData(data, file, isFile1D); }
}


template <typename T> void
readData (std::vector<T>& data, std::vector<std::string> const& files,
          bool isFile1D)
{
  std::pair<int, int> size;
  readData(data, size, files, isFile1D);
}


template <typename T> void
readData (std::vector<std::vector<T>>& data, std::pair<int, int>& size,
          std::string const& file)
{
  size = getFileSize(file);
  if (size.first >= 0 && size.second >= 0) {
    data.reserve(data.size() + size.first);
    std::ifstream is(file);
    for (int i = 0; i < size.first; ++i) {
      data.emplace_back();
      data.back().reserve(size.second);
      for (int j = 0; j < size.second; ++j) {
        data.back().emplace_back();
        is >> data.back().back();
      }
    }
    is.close();
  }
  else {
    perr(std::string("Error: invalid data file dimension in ")
         .append(file));
  }
}


template <typename T> void
readData (std::vector<std::vector<T>>& data, std::string const& file)
{
  std::pair<int, int> size;
  readData(data, size, file);
}


template <typename T> void
readData (std::vector<std::vector<T>>& data, std::pair<int, int>& size,
          std::vector<std::string> const& files)
{
  size = getFileSize(files);
  data.reserve(size.first);
  int n = files.size();
  for (int i = 0; i < n; ++i) { readData(data, files[i]); }
}


template <typename T> void
readData (std::vector<std::vector<T>>& data,
          std::vector<std::string> const& files)
{
  std::pair<int, int> size;
  readData(data, size, files);
}


template <typename T> void
appendData (std::vector<std::vector<T>>& data,
            std::vector<T> const& toAppend)
{
  for (auto& sample : data) {
    sample.insert(sample.end(), toAppend.begin(), toAppend.end());
  }
}

};

#endif

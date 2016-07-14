#ifndef _glia_type_feat_hxx_
#define _glia_type_feat_hxx_

#include "glia_base.hxx"
#include "alg/geometry.hxx"
#include "util/image_stats.hxx"

namespace glia {
namespace feat {

void standardize (
    std::vector<FVal>& feats,
    std::vector<std::vector<FVal>> const& featNormalizer,
    FVal inputFeatNullVal, FVal outputFeatNullVal) {
  int rowMean = 0, rowStd = 1;
  int d = feats.size();
  for (int i = 0; i < d; ++i) {
    feats[i] = isfeq(feats[i], inputFeatNullVal) ?
        outputFeatNullVal : sdivide(
            feats[i] - featNormalizer[rowMean][i],
            featNormalizer[rowStd][i], outputFeatNullVal);
  }
}


// Region shape features
class RegionShapeFeats : public Object {
 public:
  typedef RegionShapeFeats Self;
  uint regionDim = 0, dim = 4;
  double area = 0.0, perim = 0.0, compactness = 0.0, bboxArea = 0.0;
  std::vector<double> bboxSize;

  RegionShapeFeats () {}

  RegionShapeFeats (uint regionDim) { init(regionDim); }

  ~RegionShapeFeats () override {}

  virtual void init (uint regionDim) {
    this->regionDim = regionDim;
    dim = regionDim + 4;
    bboxSize.resize(regionDim, 0.0);
  }

  virtual void log () {
    area = slog(area, 0.0);
    perim = slog(perim, 0.0);
    // compactness = slog(compactness, 0.0);
    bboxArea = slog(bboxArea, 0.0);
    for (auto& x: bboxSize) { x = slog(x, 0.0); }
  }

  virtual void serialize (std::vector<FVal>& feats) const {
    feats.reserve(feats.size() + dim);
    feats.push_back(area);
    feats.push_back(perim);
    feats.push_back(compactness);
    feats.push_back(bboxArea);
    std::copy(
        bboxSize.begin(), bboxSize.end(), std::back_inserter(feats));
  }

  friend std::ostream& operator<< (std::ostream& os, Self const& f) {
    os << f.area << " " << f.perim << " "
       << f.compactness << " " << f.bboxArea;
    for (auto x: f.bboxSize) { os << " " << x; }
    return os;
  }

  template <typename TPoints> void
  generate (TPoints const& region, double normalizingArea,
            double normalizingLength) {
    const uint D = TPoints::Point::Dimension;
    init(D);
    area = region.size();
    perim = region.boundary.size() + region.border.size();
    compactness =
        sdivide(std::pow(perim, (double)D / (D - 1)), area, 0.0);
    area = sdivide(area, normalizingArea, 0.0);
    perim = sdivide(perim, normalizingLength, 0.0);
    BoundingBox<D> bb;
    alg::getBoundingBox(bb, region);
    bboxArea = 1.0;
    for (int i = 0; i < D; ++i) {
      bboxSize[i] = sdivide(bb[i], normalizingLength, 0.0);
      bboxArea *= bb[i];
    }
    bboxArea = sdivide(bboxArea, normalizingArea, 0.0);
  }
};


class RegionShapeDiffFeats : public Object {
 public:
  typedef RegionShapeDiffFeats Self;
  const uint dim = 6;
  double areaDiff = 0.0, rAreaDiff0 = 0.0, rAreaDiff1 = 0.0;
  double perimDiff = 0.0, rPerimDiff0 = 0.0, rPerimDiff1 = 0.0;

  ~RegionShapeDiffFeats () override {}

  virtual void log () {
    areaDiff = slog(areaDiff, 0.0);
    perimDiff = slog(perimDiff, 0.0);
  }

  virtual void serialize (std::vector<FVal>& feats) const {
    feats.reserve(feats.size() + dim);
    feats.push_back(areaDiff);
    feats.push_back(rAreaDiff0);
    feats.push_back(rAreaDiff1);
    feats.push_back(perimDiff);
    feats.push_back(rPerimDiff0);
    feats.push_back(rPerimDiff1);
  }

  friend std::ostream& operator<< (std::ostream& os, Self const& f) {
    return os << f.areaDiff << " " << f.rAreaDiff0 << " "
              << f.rAreaDiff1 << " " << f.perimDiff << " "
              << f.rPerimDiff0 << " " << f.rPerimDiff1;
  }

  void generate (
      RegionShapeFeats const& rf0, RegionShapeFeats const& rf1) {
    areaDiff = std::fabs(rf0.area - rf1.area);
    rAreaDiff0 = sdivide(areaDiff, rf0.area, 0.0);
    rAreaDiff1 = sdivide(areaDiff, rf1.area, 0.0);
    perimDiff = std::fabs(rf0.perim - rf1.perim);
    rPerimDiff0 = sdivide(perimDiff, rf0.perim, 0.0);
    rPerimDiff1 = sdivide(perimDiff, rf1.perim, 0.0);
  }
};


// Region shape difference features
class RegionShapeIntraDiffFeats : public RegionShapeDiffFeats {
 public:
  typedef RegionShapeDiffFeats Super;
  typedef RegionShapeIntraDiffFeats Self;
  const uint dim = Super::dim + 5;
  double boundaryLength = 0.0;
  double rBoundaryLengthArea0 = 0.0, rBoundaryLengthArea1 = 0.0;
  double rBoundaryLengthPerim0 = 0.0, rBoundaryLengthPerim1 = 0.0;

  ~RegionShapeIntraDiffFeats () override {}

  virtual void log () {
    Super::log();
    boundaryLength = slog(boundaryLength, 0.0);
    // rBoundaryLengthArea0 = slog(rBoundaryLengthArea0, LOG_FEPS);
    // rBoundaryLengthArea1 = slog(rBoundaryLengthArea1, LOG_FEPS);
    // rBoundaryLengthPerim0 = slog(rBoundaryLengthPerim0, LOG_FEPS);
    // rBoundaryLengthPerim1 = slog(rBoundaryLengthPerim1, LOG_FEPS);
  }

  virtual void serialize (std::vector<FVal>& feats) const {
    feats.reserve(feats.size() + dim);
    Super::serialize(feats);
    feats.push_back(boundaryLength);
    feats.push_back(rBoundaryLengthArea0);
    feats.push_back(rBoundaryLengthArea1);
    feats.push_back(rBoundaryLengthPerim0);
    feats.push_back(rBoundaryLengthPerim1);
  }

  friend std::ostream& operator<< (std::ostream& os, Self const& f) {
    return os << static_cast<Super const&>(f)
              << f.boundaryLength << " "
              << f.rBoundaryLengthArea0 << " "
              << f.rBoundaryLengthArea1 << " "
              << f.rBoundaryLengthPerim0 << " "
              << f.rBoundaryLengthPerim1;
  }

  template <typename TPoints> void
  generate (TPoints const& boundary, RegionShapeFeats const& rf0,
            RegionShapeFeats const& rf1, double normalizingLength) {
    Super::generate(rf0, rf1);
    boundaryLength = sdivide(
        std::ceil(boundary.size() / 2.0), normalizingLength, 0.0);
    rBoundaryLengthArea0 = sdivide(boundaryLength, rf0.area, 0.0);
    rBoundaryLengthArea1 = sdivide(boundaryLength, rf1.area, 0.0);
    rBoundaryLengthPerim0 = sdivide(boundaryLength, rf0.perim, 0.0);
    rBoundaryLengthPerim1 = sdivide(boundaryLength, rf1.perim, 0.0);
  }
};


// Only support 2D regions for now
class RegionAdvShapeFeats2D : public Object {
 public:
  typedef RegionAdvShapeFeats2D Self;
  const uint dim = 15;
  std::array<double, 7> centralMoments;
  std::array<double, 7> huMoments;
  double eccentricity;

  ~RegionAdvShapeFeats2D () override {}

  virtual void log ()
  { for (double& x : centralMoments) { x = slog(x, 0.0); } }

  virtual void serialize (std::vector<FVal>& feats) const {
    feats.reserve(feats.size() + dim);
    std::copy(
        centralMoments.begin(), centralMoments.end(),
        std::back_inserter(feats));
    std::copy(
        huMoments.begin(), huMoments.end(), std::back_inserter(feats));
    feats.push_back(eccentricity);
  }

  friend std::ostream& operator<< (std::ostream& os, Self const& f) {
    for (double x : f.centralMoments) { os << x << " "; }
    for (double x : f.huMoments) { os << x << " "; }
    return os << f.eccentricity;
  }

  template <typename TKey> void
  generate (
      TRegion<TKey, Point<2>> const& region, fPoint<2> const& centroid,
      double normalizingLength) {
    alg::getCentralMoments(centralMoments, region, centroid);
    std::array<double, 7> sims;
    alg::getScaleInvariantMoments(sims, region.size(), centralMoments);
    if (normalizingLength > 0.0) {
      double normalizingLength2 = normalizingLength * normalizingLength;
      double normalizingLength3 = normalizingLength2 * normalizingLength;
      centralMoments[0] /= normalizingLength2;
      centralMoments[1] /= normalizingLength3;
      centralMoments[2] /= normalizingLength2;
      centralMoments[3] /= normalizingLength3;
      centralMoments[4] /= normalizingLength2;
      centralMoments[5] /= normalizingLength2;
      centralMoments[6] /= normalizingLength3;
    }
    alg::getHuMoments(huMoments, sims);
    eccentricity = alg::getEccentricity(
        centralMoments[0], centralMoments[2], centralMoments[4]);
  }
};


// Only support 2D regions for now
class RegionAdvShapeFeatsDiff2D : public Object {
 public:
  typedef RegionAdvShapeFeatsDiff2D Self;
  const uint dim = 15;
  std::array<double, 7> centralMomentDiff;
  std::array<double, 7> huMomentDiff;
  double eccentricityDiff;

  ~RegionAdvShapeFeatsDiff2D () override {}

  virtual void log ()
  { for (double& x : centralMomentDiff) { x = slog(x, 0.0); } }

  virtual void serialize (std::vector<FVal>& feats) const {
    feats.reserve(feats.size() + dim);
    std::copy(
        centralMomentDiff.begin(), centralMomentDiff.end(),
        std::back_inserter(feats));
    std::copy(
        huMomentDiff.begin(), huMomentDiff.end(),
        std::back_inserter(feats));
    feats.push_back(eccentricityDiff);
  }

  friend std::ostream& operator<< (std::ostream& os, Self const& f) {
    for (double x : f.centralMomentDiff) { os << x << " "; }
    for (double x : f.huMomentDiff) { os << x << " "; }
    return os << f.eccentricityDiff;
  }

  void generate (
      RegionAdvShapeFeats2D const& rf0,
      RegionAdvShapeFeats2D const& rf1) {
    for (int i = 0; i < 7; ++i)  {
      centralMomentDiff[i] = std::fabs(
          rf0.centralMoments[i] - rf1.centralMoments[i]);
      huMomentDiff[i] = std::fabs(
          rf0.huMoments[i] - rf1.huMoments[i]);
    }
    eccentricityDiff = fabs(rf0.eccentricity - rf1.eccentricity);
  }
};


class RegionLocationFeats : public Object {
 public:
  typedef RegionLocationFeats Self;
  uint regionDim = 0, dim = 0;
  std::vector<double> centroid;

  RegionLocationFeats () {}

  RegionLocationFeats (uint regionDim) { init(regionDim); }

  ~RegionLocationFeats () override {}

  virtual void init (uint regionDim) {
    this->regionDim = regionDim;
    dim = regionDim;
    centroid.resize(regionDim, 0.0);
  }

  virtual void serialize (std::vector<FVal>& feats) const {
    feats.reserve(feats.size() + dim);
    std::copy(
        centroid.begin(), centroid.end(), std::back_inserter(feats));
  }

  friend std::ostream& operator<< (std::ostream& os, Self const& f) {
    for (int i = 0; i < f.centroid.size(); ++i) {
      if (i == 0) { os << f.centroid.front(); }
      else { os << " " << f.centroid[i]; }
    }
    return os;
  }

  template <typename TPoints> void
  generate (TPoints const& region, double normalizingLength) {
    std::fill(centroid.begin(), centroid.end(), 0.0);
    region.traverse([this](typename TPoints::Point const& p) {
        for (int i = 0; i < this->regionDim; ++i) { centroid[i] += p[i]; }
      });
    double siz = region.size() * normalizingLength;
    for (int i = 0; i < regionDim; ++i) { centroid[i] /= siz; }
  }

  void generate (
      std::vector<double> const& centroid0, int size0,
      std::vector<double> const& centroid1, int size1) {
    for (int i = 0; i < regionDim; ++i) {
      centroid[i] = sdivide(
          centroid0[i] * size0 + centroid1[i] * size1, size0 + size1, 0.0);
    }
  }
};


class RegionLocationDiffFeats : public Object {
 public:
  typedef RegionLocationDiffFeats Self;
  const uint dim = 1;
  double centroidDistL2 = 0.0;

  ~RegionLocationDiffFeats () override {}

  virtual void log () {
    centroidDistL2 = std::max(0.0, slog(centroidDistL2, 0.0));
  }

  virtual void serialize (std::vector<FVal>& feats) const {
    feats.reserve(feats.size() + dim);
    feats.push_back(centroidDistL2);
  }

  friend std::ostream& operator<< (std::ostream& os, Self const& f) {
    return os << f.centroidDistL2;
  }

  void generate (
      RegionLocationFeats const& rf0, RegionLocationFeats const& rf1) {
    centroidDistL2 = 0.0;
    for (int i = 0; i < rf0.regionDim; ++i) {
      double diff = rf0.centroid[i] - rf1.centroid[i];
      centroidDistL2 += diff * diff;
    }
    centroidDistL2 = ssqrt(centroidDistL2, 0.0);
  }
};


class RegionSetDiffFeats : public Object {
 public:
  typedef RegionSetDiffFeats Self;
  const uint dim = 8;
  double overlap = 0.0, setDiff0 = 0.0, setDiff1 = 0.0, symDiff = 0.0;
  double rOverlapArea0 = 0.0, rOverlapArea1 = 0.0;
  double rSetDiffArea0 = 0.0, rSetDiffArea1 = 0.0;

  ~RegionSetDiffFeats () override {};

  virtual void log () {
    overlap = slog(overlap, 0.0);
    setDiff0 = slog(setDiff0, 0.0);
    setDiff1 = slog(setDiff1, 0.0);
    symDiff = slog(symDiff, 0.0);
  }

  virtual void serialize (std::vector<FVal>& feats) const {
    feats.reserve(feats.size() + dim);
    feats.push_back(overlap);
    feats.push_back(setDiff0);
    feats.push_back(setDiff1);
    feats.push_back(symDiff);
    feats.push_back(rOverlapArea0);
    feats.push_back(rOverlapArea1);
    feats.push_back(rSetDiffArea0);
    feats.push_back(rSetDiffArea1);
  }

  friend std::ostream& operator<< (std::ostream& os, Self const& f) {
    return os << f.overlap << " " << f.setDiff0 << " "
              << f.setDiff1 << " " << f.symDiff << " "
              << f.rOverlapArea0 << " " << f.rOverlapArea1 << " "
              << f.rSetDiffArea0 << " " << f.rSetDiffArea1;
  }

  void generate (double ov, double a0, double a1) {
    overlap = ov;
    setDiff0 = a0 - ov;
    setDiff1 = a1 - ov;
    symDiff = setDiff0 + setDiff1;
    rOverlapArea0 = overlap / a0;
    rOverlapArea1 = overlap / a1;
    rSetDiffArea0 = setDiff0 / a0;
    rSetDiffArea1 = setDiff1 / a1;
  }

  template <typename TPoints, typename TImagePtr> void
  generate (
      TPoints const& region0, TPoints const& region1,
      double normalizingArea, TImagePtr& canvas) {
    canvas->FillBuffer(0);
    region0.traverse(
        [&canvas](typename TPoints::Point const& p) {
          canvas->SetPixel(p, 1); });
    int ov = 0;
    region1.traverse(
        [&canvas, &ov](typename TPoints::Point const& p) {
          if (canvas->GetPixel(p) != 0) { ++ov; } });
    generate(ov, region0.size(), region1.size());
  }
};


// Region shape features with images
class ImageRegionShapeFeats : public virtual RegionShapeFeats {
 public:
  typedef RegionShapeFeats Super;
  typedef ImageRegionShapeFeats Self;
  std::vector<double> validPerims, rValidPerims;
  uint nThreshold = 0, dim = Super::dim;

  ImageRegionShapeFeats () {}

  ImageRegionShapeFeats (uint regionDim, uint nThreshold)
  { init(regionDim, nThreshold); }

  ~ImageRegionShapeFeats () override {}

  virtual void init (uint regionDim, uint nThreshold) {
    Super::init(regionDim);
    this->nThreshold = nThreshold;
    dim = Super::dim + nThreshold * 2;
    validPerims.resize(nThreshold, 0.0);
    rValidPerims.resize(nThreshold, 0.0);
  }

  void log () override {
    Super::log();
    for (auto& x: validPerims) { x = slog(x, 0.0); }
    // for (auto& x: rValidPerims) { x = slog(x, LOG_FEPS); }
  }

  void serialize (std::vector<FVal>& feats) const override {
    feats.reserve(feats.size() + dim);
    Super::serialize(feats);
    std::copy(validPerims.begin(), validPerims.end(),
              std::back_inserter(feats));
    std::copy(rValidPerims.begin(), rValidPerims.end(),
              std::back_inserter(feats));
  }

  friend std::ostream& operator<< (std::ostream& os, Self const& f) {
    os << static_cast<Super const&>(f);
    for (auto x: f.validPerims) { os << " " << x; }
    for (auto x: f.rValidPerims) { os << " " << x; }
    return os;
  }

  template <typename TPoints, typename TImagePtr> void
  generate (TPoints const& region, double normalizingArea,
            double normalizingLength, TImagePtr const& image,
            std::vector<double> const& thresholds) {
    const uint D = TPoints::Point::Dimension;
    uint N = thresholds.size();
    init(D, N);
    Super::generate(region, normalizingArea, normalizingLength);
    for (int i = 0; i < N; ++i) {
      uint vp = 0;
      region.boundary.traverse
          ([&vp, &image, &thresholds, i]
           (typename TPoints::Point const& p)
           { if (image->GetPixel(p) >= thresholds[i]) { ++vp; } });
      validPerims[i] = sdivide(vp, normalizingLength, 0.0);
      rValidPerims[i] = sdivide(vp, region.boundary.size(), 0.0);
    }
  }
};


// Region shape difference features with images
class ImageRegionShapeIntraDiffFeats :
      public virtual RegionShapeIntraDiffFeats {
 public:
  typedef RegionShapeIntraDiffFeats Super;
  typedef ImageRegionShapeIntraDiffFeats Self;
  uint nThreshold = 0, dim = Super::dim;
  std::vector<double> validBoundaryLengths, rValidBoundaryLengths,
    rValidBoundaryLengthPerims0, rValidBoundaryLengthPerims1;

  ImageRegionShapeIntraDiffFeats () {}

  ImageRegionShapeIntraDiffFeats (uint nThreshold) { init(nThreshold); }

  ~ImageRegionShapeIntraDiffFeats () override {}

  virtual void init (uint nThreshold) {
    this->nThreshold = nThreshold;
    dim = Super::dim + nThreshold * 4;
    validBoundaryLengths.resize(nThreshold, 0.0);
    rValidBoundaryLengths.resize(nThreshold, 0.0);
    rValidBoundaryLengthPerims0.resize(nThreshold, 0.0);
    rValidBoundaryLengthPerims1.resize(nThreshold, 0.0);
  }

  void log () override {
    Super::log();
    for (auto& x: validBoundaryLengths) { x = slog(x, 0.0); }
    // for (auto& x: rValidBoundaryLengths) { x = slog(x, LOG_FEPS); }
    // for (auto& x: rValidBoundaryLengthPerims0)
    // { x = slog(x, LOG_FEPS); }
    // for (auto& x: rValidBoundaryLengthPerims1)
    // { x = slog(x, LOG_FEPS); }
  }

  void serialize (std::vector<FVal>& feats) const override {
    feats.reserve(feats.size() + dim);
    Super::serialize(feats);
    std::copy(validBoundaryLengths.begin(),
              validBoundaryLengths.end(),
              std::back_inserter(feats));
    std::copy(rValidBoundaryLengths.begin(),
              rValidBoundaryLengths.end(),
              std::back_inserter(feats));
    std::copy(rValidBoundaryLengthPerims0.begin(),
              rValidBoundaryLengthPerims0.end(),
              std::back_inserter(feats));
    std::copy(rValidBoundaryLengthPerims1.begin(),
              rValidBoundaryLengthPerims1.end(),
              std::back_inserter(feats));
  }

  friend std::ostream& operator<< (std::ostream& os, Self const& f) {
    os << static_cast<Super const&>(f);
    for (auto x: f.validBoundaryLengths) { os << " " << x; }
    for (auto x: f.rValidBoundaryLengths) { os << " " << x; }
    for (auto x: f.rValidBoundaryLengthPerims0) { os << " " << x; }
    for (auto x: f.rValidBoundaryLengthPerims1) { os << " " << x; }
    return os;
  }

  template <typename TPoints, typename TImagePtr> void
  generate (TPoints const& boundary, double normalizingLength,
            RegionShapeFeats const& rf0, RegionShapeFeats const& rf1,
            TImagePtr const& image,
            std::vector<double> const& thresholds) {
    init(thresholds.size());
    Super::generate(boundary, rf0, rf1, normalizingLength);
    for (int i = 0; i < nThreshold; ++i) {
      uint vp = 0;
      boundary.traverse
          ([&vp, &image, &thresholds, i]
           (typename TPoints::Point const& p)
           { if (image->GetPixel(p) >= thresholds[i]) { ++vp; } });
      validBoundaryLengths[i] = sdivide(
          std::ceil(vp / 2.0), normalizingLength, 0.0);
      rValidBoundaryLengths[i] = sdivide(
          validBoundaryLengths[i], Super::boundaryLength, 0.0);
      rValidBoundaryLengthPerims0[i] = sdivide(
          validBoundaryLengths[i], rf0.perim, 0.0);
      rValidBoundaryLengthPerims1[i] = sdivide(
          validBoundaryLengths[i], rf1.perim, 0.0);
    }
  }
};


// Label image features
class ImageLabelFeats : public Object {
 public:
  typedef ImageLabelFeats Self;
  uint histBin = 0, dim = 1;
  std::vector<double> histogram;
  double entropy = 0.0;

  ImageLabelFeats () {}

  ImageLabelFeats (uint histBin) { init(histBin); }

  ~ImageLabelFeats () override {}

  virtual void init (uint histBin) {
#ifdef GLIA_USE_HISTOGRAM_AS_FEATS
    dim = histBin + 1;
#else
    dim = 1;
#endif
    histogram.resize(histBin, 0.0);
  }

  virtual void serialize (std::vector<FVal>& feats) const {
    feats.reserve(feats.size() + dim);
#ifdef GLIA_USE_HISTOGRAM_AS_FEATS
    std::copy(histogram.begin(), histogram.end(),
              std::back_inserter(feats));
#endif
    feats.push_back(entropy);
  }

  friend std::ostream& operator<< (std::ostream& os, Self const& f) {
#ifdef GLIA_USE_HISTOGRAM_AS_FEATS
    for (auto x: f.histogram) { os << x << " "; }
#endif
    return os << f.entropy;
  }

  template <typename TPoints, typename TImagePtr> void
  generate (TPoints const& points, TImagePtr const& image,
            uint histBin, std::pair<double, double> const& histRange) {
    init(histBin);
    stats::hist(histogram, image, points, histBin, histRange);
    entropy = stats::entropy(histogram);
  }
};


// Label image difference features
class ImageLabelDiffFeats : public Object {
 public:
  typedef ImageLabelDiffFeats Self;
  const uint dim = 3;
  double histDistL1 = 0.0, histDistX2 = 0.0, entropyDiff = 0.0;

  ~ImageLabelDiffFeats () override {}

  virtual void serialize (std::vector<FVal>& feats) const {
    feats.reserve(feats.size() + dim);
    feats.push_back(histDistL1);
    feats.push_back(histDistX2);
    feats.push_back(entropyDiff);
  }

  friend std::ostream& operator<< (std::ostream& os, Self const& f) {
    return os << f.histDistL1 << " " << f.histDistX2 << " "
              << f.entropyDiff;
  }

  virtual void generate (ImageLabelFeats const& if0,
                         ImageLabelFeats const& if1)
  {
    histDistL1 = stats::distL1(if0.histogram, if1.histogram);
    histDistX2 = stats::distX2(if0.histogram, if1.histogram);
    entropyDiff = fabs(if0.entropy - if1.entropy);
  }
};


// Real image features
class ImageRealFeats : public Object {
 public:
  typedef ImageRealFeats Self;
#ifdef GLIA_USE_MEDIAN_AS_FEATS
  const uint dim = 5;
  double median = 0.0;
#else
  const uint dim = 4;
#endif
  double mean = 0.0, stddev = 0.0, min = 0.0, max = 0.0;

  ~ImageRealFeats () override {}

  virtual void serialize (std::vector<FVal>& feats) const {
    feats.reserve(feats.size() + dim);
#ifdef GLIA_USE_MEDIAN_AS_FEATS
    feats.push_back(median);
#endif
    feats.push_back(mean);
    feats.push_back(stddev);
    feats.push_back(min);
    feats.push_back(max);
  }

  friend std::ostream& operator<< (std::ostream& os, Self const& f) {
#ifdef GLIA_USE_MEDIAN_AS_FEATS
    os << f.median << " ";
#endif
    return os << f.mean << " " << f.stddev << " " << f.min << " "
              << f.max;
  }

  template <typename TPoints, typename TImagePtr> void
  generate (TPoints const& points, TImagePtr const& image) {
    int n = points.size();
    if (n == 0) { return; }
#ifdef GLIA_USE_MEDIAN_AS_FEATS
    std::vector<TImageVal<TImagePtr>> vals;
    vals.reserve(n);
    points.traverse(
        [&image, &vals, this](typename TPoints::Point const& p) {
          auto val = image->GetPixel(p);
          vals.push_back(val);
        });
    median = stats::amedian(vals);
    mean = stats::mean(vals);
    stddev = ssqrt(stats::var(vals, mean), 0.0);
    min = stats::min(vals);
    max = stats::max(vals);
#else
    mean = 0.0;
    min = FMAX;
    max = -FMAX;
    stddev = 0.0;
    points.traverse([&image, this](typename TPoints::Point const& p) {
        auto val = image->GetPixel(p);
        this->mean += val;
        this->stddev += (double)val * val;
        if (val < this->min) { this->min = val; }
        if (val > this->max) { this->max = val; }
      });
    mean /= n;
    stddev = ssqrt(stddev / n - mean * mean, 0.0);
#endif
  }
  // template <typename TPoints, typename TImagePtr> void
  // generate (TPoints const& points, TImagePtr const& image) {
  //   auto n = points.size();
  //   if (n == 0) {
  //     mean = 0.0;
  //     min = 0.0;
  //     max = 0.0;
  //     stddev = 0.0;
  //     return;
  //   }
  //   mean = 0.0;
  //   min = FMAX;
  //   max = -FMAX;
  //   stddev = 0.0;
  //   points.traverse([&image, this]
  //                   (typename TPoints::Point const& p)
  //                   {
  //                     auto val = image->GetPixel(p);
  //                     this->mean += val;
  //                     this->stddev += (double)val * val;
  //                     if (val < this->min) { this->min = val; }
  //                     if (val > this->max) { this->max = val; }
  //                   });
  //   mean /= n;
  //   stddev = ssqrt(stddev / n - mean * mean, 0.0);
  // }
};


// Real image difference features
class ImageRealDiffFeats : public Object {
 public:
  typedef ImageRealDiffFeats Self;
#ifdef GLIA_USE_MEDIAN_AS_FEATS
  const uint dim = 5;
  double medianDiff = 0.0;
#else
  const uint dim = 4;
#endif
  double meanDiff = 0.0, stdDiff = 0.0, minDiff = 0.0, maxDiff = 0.0;

  ~ImageRealDiffFeats () override {}

  virtual void serialize (std::vector<FVal>& feats) const {
    feats.reserve(feats.size() + dim);
#ifdef GLIA_USE_MEDIAN_AS_FEATS
    feats.push_back(medianDiff);
#endif
    feats.push_back(meanDiff);
    feats.push_back(stdDiff);
    feats.push_back(minDiff);
    feats.push_back(maxDiff);
  }

  friend std::ostream& operator<< (std::ostream& os, Self const& f) {
#ifdef GLIA_USE_MEDIAN_AS_FEATS
    os << f.medianDiff << " ";
#endif
    return os << f.meanDiff << " " << f.stdDiff << " "
              << f.minDiff << " " << f.maxDiff;
  }

  virtual void generate (ImageRealFeats const& if0,
                         ImageRealFeats const& if1) {
#ifdef GLIA_USE_MEDIAN_AS_FEATS
    medianDiff = std::fabs(if0.median - if1.median);
#endif
    meanDiff = std::fabs(if0.mean - if1.mean);
    stdDiff = std::fabs(if0.stddev - if1.stddev);
    minDiff = std::fabs(if0.min - if1.min);
    maxDiff = std::fabs(if0.max - if1.max);
  }
};


// Image features
class ImageFeats :
      public virtual ImageLabelFeats,
      public virtual ImageRealFeats {
 public:
  typedef ImageLabelFeats Super0;
  typedef ImageRealFeats Super1;
  typedef ImageFeats Self;
  uint dim = Super0::dim + Super1::dim;

  ImageFeats () {}

  ImageFeats (uint histBin) { init(histBin); }

  ~ImageFeats () override {}

  void init (uint histBin) override {
    Super0::init(histBin);
    dim = Super0::dim + Super1::dim;
  }

  void serialize (std::vector<FVal>& feats) const override {
    feats.reserve(feats.size() + dim);
    Super0::serialize(feats);
    Super1::serialize(feats);
  }

  friend std::ostream& operator<< (std::ostream& os, Self const& f) {
    return os << static_cast<Super0 const&>(f) << " "
              << static_cast<Super1 const&>(f);
  }

  template <typename TPoints, typename TImagePtr> void
  generate (TPoints const& points, TImagePtr const& image,
            uint histBin, std::pair<double, double> const& histRange) {
    init(histBin);
    Super0::generate(points, image, histBin, histRange);
    Super1::generate(points, image);
  }
};


// Image difference features
class ImageDiffFeats :
      public virtual ImageLabelDiffFeats,
      public virtual ImageRealDiffFeats {
 public:
  typedef ImageLabelDiffFeats Super0;
  typedef ImageRealDiffFeats Super1;
  typedef ImageDiffFeats Self;
  const uint dim = Super0::dim + Super1::dim;

  ~ImageDiffFeats () override {}

  void serialize (std::vector<FVal>& feats) const override {
    feats.reserve(feats.size() + dim);
    Super0::serialize(feats);
    Super1::serialize(feats);
  }

  virtual void generate (ImageFeats const& if0, ImageFeats const& if1) {
    Super0::generate(if0, if1);
    Super1::generate(if0, if1);
  }

  friend std::ostream& operator<< (std::ostream& os, Self const& f) {
    return os << static_cast<Super0 const&>(f) << " "
              << static_cast<Super1 const&>(f);
  }
};

};
};

#endif

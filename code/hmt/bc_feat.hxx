#ifndef _glia_hmt_bc_feat_hxx_
#define _glia_hmt_bc_feat_hxx_

#include "type/feat.hxx"
#include "type/tuple.hxx"

namespace glia {
namespace hmt {

const FVal BC_FEAT_NULL_VAL = -99999.0;

template <typename TKey> void
genSaliencyMap (std::unordered_map<TKey, double>& saliencyMap,
                std::vector<TTriple<TKey>> const& order,
                std::vector<double> const& saliencies,
                double initSaliency, double saliencyBias)
{
  int n = order.size();
  for (int i = 0; i < n; ++i) {
    if (saliencyMap.count(order[i].x0) == 0)
    { saliencyMap[order[i].x0] = initSaliency; }
    if (saliencyMap.count(order[i].x1) == 0)
    { saliencyMap[order[i].x1] = initSaliency; }
    saliencyMap[order[i].x2] = saliencies[i] + saliencyBias;
  }
}


template <typename TImagePtr>
struct ImageHistPair {
  TImagePtr image;
  uint histBin;
  std::pair<double, double> histRange;

  ImageHistPair () {}

  ImageHistPair(TImagePtr const& image, uint histBin,
                std::pair<double, double> const& histRange)
      : image(image), histBin(histBin), histRange(histRange) {}

  virtual ~ImageHistPair () {}
};


// Region features
class RegionFeats : public Object {
 public:
  typedef RegionFeats Self;
  std::shared_ptr<glia::feat::ImageRegionShapeFeats> shape;
  std::vector<std::shared_ptr<glia::feat::ImageFeats>> region;
  std::vector<std::shared_ptr<glia::feat::ImageLabelFeats>> labelRegion;
  std::vector<std::shared_ptr<glia::feat::ImageFeats>> boundary;
  std::shared_ptr<double> saliency;

  ~RegionFeats () override {}

  virtual uint dim () const {
    uint ret = 0;
    if (shape) { ret += shape->dim; }
    for (auto const& p: region) { if (p) { ret += p->dim; } }
    for (auto const& p: labelRegion) { if (p) { ret += p->dim; } }
    for (auto const& p: boundary) { if (p) { ret += p->dim; } }
    if (saliency) { ++ret; }
    return ret;
  }

  virtual void log () { shape->log(); }

  virtual void serialize (std::vector<FVal>& feats) const {
    feats.reserve(feats.size() + dim());
    if (shape) { shape->serialize(feats); }
    for (auto const& p: region) { if (p) { p->serialize(feats); } }
    for (auto const& p: labelRegion)
    { if (p) { p->serialize(feats); } }
    for (auto const& p: boundary) { if (p) { p->serialize(feats); } }
    if (saliency) { feats.push_back(*saliency); }
  }

  friend std::ostream& operator<< (std::ostream& os, Self const& f) {
    if (f.shape) { os << *f.shape; }
    for (auto const& p: f.region) { if (p) { os << " " << *p; } }
    for (auto const& p: f.labelRegion) { if (p) { os << " " << *p; } }
    for (auto const& p: f.boundary) { if (p) { os << " " << *p; } }
    if (f.saliency) { os << " " << *f.saliency; }
    return os;
  }

  template <typename TPoints, typename TRImagePtr, typename TLImagePtr>
  void generate (
      TPoints const& reg, double normalizingArea,
      double normalizingLength, TRImagePtr const& pbImage,
      std::vector<double> const& boundaryThresholds,
      std::vector<ImageHistPair<TRImagePtr>> const& rImages,
      std::vector<ImageHistPair<TLImagePtr>> const& rlImages,
      std::vector<ImageHistPair<TRImagePtr>> const& bImages,
      double const* pSaliency) {
    shape = std::make_shared<glia::feat::ImageRegionShapeFeats>();
    shape->generate(reg, normalizingArea, normalizingLength,
                    pbImage, boundaryThresholds);
    region.reserve(rImages.size());
    for (auto const& ihp: rImages) {
      if (ihp.image.IsNotNull()) {
        region.push_back(std::make_shared<glia::feat::ImageFeats>());
        region.back()->generate
            (reg, ihp.image, ihp.histBin, ihp.histRange);
      }
    }
    labelRegion.reserve(rlImages.size());
    for (auto const& ihp: rlImages) {
      if (ihp.image.IsNotNull()) {
        labelRegion
            .push_back(std::make_shared<glia::feat::ImageLabelFeats>());
        labelRegion.back()->generate
            (reg, ihp.image, ihp.histBin, ihp.histRange);
      }
    }
    boundary.reserve(bImages.size());
    for (auto const& ihp: bImages) {
      if (ihp.image.IsNotNull()) {
        boundary.push_back(std::make_shared<glia::feat::ImageFeats>());
        boundary.back()->generate
            (reg.boundary, ihp.image, ihp.histBin, ihp.histRange);
      }
    }
    if (pSaliency)
    { saliency = std::make_shared<double>(*pSaliency); }
  }
};


// Boundary features
class BoundaryFeats : public Object {
 public:
  typedef BoundaryFeats Self;
  std::shared_ptr<glia::feat::ImageRegionShapeIntraDiffFeats> shape;
  std::vector<std::shared_ptr<glia::feat::ImageDiffFeats>> region;
  std::vector<std::shared_ptr<glia::feat::ImageLabelDiffFeats>>
  labelRegion;
  std::vector<std::shared_ptr<glia::feat::ImageFeats>> boundary;
  std::shared_ptr<std::pair<double, double>> saliency;

  ~BoundaryFeats () override {}

  virtual uint dim () const {
    uint ret = 0;
    if (shape) { ret += shape->dim; }
    for (auto const& p: region) { if (p) { ret += p->dim; } }
    for (auto const& p: labelRegion) { if (p) { ret += p->dim; } }
    for (auto const& p: boundary) { if (p) { ret += p->dim; } }
    if (saliency) { ret += 2; }
    return ret;
  }

  virtual void log () { if (shape) { shape->log(); } }

  virtual void serialize (std::vector<FVal>& feats) const {
    feats.reserve(feats.size() + dim());
    if (shape) { shape->serialize(feats); }
    for (auto const& p: region) { if (p) { p->serialize(feats); } }
    for (auto const& p: labelRegion)
    { if (p) { p->serialize(feats); } }
    for (auto const& p: boundary) { if (p) { p->serialize(feats); } }
    if (saliency) {
      feats.push_back(saliency->first);
      feats.push_back(saliency->second);
    }
  }

  friend std::ostream& operator<< (std::ostream& os, Self const& f) {
    if (f.shape) { os << *f.shape; }
    for (auto const& p: f.region) { if (p) { os << " " << *p; } }
    for (auto const& p: f.labelRegion) { if (p) { os << " " << *p; } }
    for (auto const& p: f.boundary) { if (p) { os << " " << *p; } }
    if (f.saliency)
    { os << " " << f.saliency->first << " " << f.saliency->second; }
    return os;
  }

  template <typename TPoints, typename TImagePtr> void
  generate (TPoints const& b, double normalizingLength,
            RegionFeats const& rf0, RegionFeats const& rf1,
            RegionFeats const& rf2, TImagePtr const& pbImage,
            std::vector<double> const& boundaryThresholds,
            std::vector<ImageHistPair<TImagePtr>> const& bImages) {
    shape = std::make_shared<glia::feat::ImageRegionShapeIntraDiffFeats>();
    shape->generate(b, normalizingLength, *rf0.shape,
                    *rf1.shape, pbImage, boundaryThresholds);
    int n = rf0.region.size();
    region.reserve(n);
    for (int i = 0; i < n; ++i) {
      region.push_back(std::make_shared<glia::feat::ImageDiffFeats>());
      region.back()->generate(*rf0.region[i], *rf1.region[i]);
    }
    n = rf0.labelRegion.size();
    labelRegion.reserve(n);
    for (int i = 0; i < n; ++i) {
      labelRegion.push_back
          (std::make_shared<glia::feat::ImageLabelDiffFeats>());
      labelRegion.back()->generate
          (*rf0.labelRegion[i], *rf1.labelRegion[i]);
    }
    boundary.reserve(bImages.size());
    for (auto const& ihp: bImages) {
      boundary.push_back(std::make_shared<glia::feat::ImageFeats>());
      boundary.back()->generate
          (b, ihp.image, ihp.histBin, ihp.histRange);
    }
    if (rf0.saliency && rf1.saliency && rf2.saliency) {
      double dsal02 = std::fabs(*rf0.saliency - *rf2.saliency);
      double dsal12 = std::fabs(*rf1.saliency - *rf2.saliency);
      saliency = std::make_shared<std::pair<double, double>>
          (std::min(dsal02, dsal12), std::max(dsal02, dsal12));
    }
  }
};


// Combined boundary classifier features
class BoundaryClassificationFeats :
      public virtual TQuad<BoundaryFeats, RegionFeats const*> {
 public:
  typedef TQuad<BoundaryFeats, RegionFeats const*> Super;
  typedef BoundaryClassificationFeats Self;

  ~BoundaryClassificationFeats () override {}

  virtual uint dim () const {
    return Super::x0.dim() + Super::x1->dim() + Super::x2->dim() +
        Super::x3->dim();
  }

  virtual void serialize (std::vector<FVal>& feats) const {
    feats.reserve(feats.size() + dim());
    Super::x0.serialize(feats);
    Super::x1->serialize(feats);
    Super::x2->serialize(feats);
    Super::x3->serialize(feats);
  }

  friend std::ostream& operator<< (std::ostream& os, Self const& f) {
    return os << f.x0 << " " << *f.x1 << " " << *f.x2 << " " << *f.x3;
  }
};


// Follow arXiv paper
void selectFeatures (
    std::vector<FVal>& f, BoundaryClassificationFeats const& bcf)
{
  int size = f.size() + 5 + 4 * bcf.x0.region.size() +
      2 * bcf.x0.labelRegion.size();
#ifdef GLIA_USE_MEDIAN_AS_FEATS
  size += 2 * bcf.x0.boundary.size();
#else
  size += bcf.x0.boundary.size();
#endif
  f.reserve(size);
  f.push_back(bcf.x1->shape->area);
  f.push_back(bcf.x2->shape->area);
  f.push_back(bcf.x1->shape->perim);
  f.push_back(bcf.x2->shape->perim);
  f.push_back(bcf.x0.shape->boundaryLength);
  for (auto const& bf : bcf.x0.boundary) {
    f.push_back(bf->mean);
#ifdef GLIA_USE_MEDIAN_AS_FEATS
    f.push_back(bf->median);
#endif
  }
  for (auto const& rf : bcf.x0.region) {
    f.push_back(rf->meanDiff);
    f.push_back(rf->histDistL1);
    f.push_back(rf->histDistX2);
    f.push_back(rf->entropyDiff);
  }
  for (auto const& rlf : bcf.x0.labelRegion) {
    f.push_back(rlf->histDistL1);
    f.push_back(rlf->histDistX2);
  }
}

};
};

#endif

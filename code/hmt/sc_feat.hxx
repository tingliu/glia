#ifndef _glia_hmt_sc_feat_hxx_
#define _glia_hmt_sc_feat_hxx_

#include "hmt/bc_feat.hxx"

namespace glia {
namespace hmt {

// Do not output location as actual features
class RegionFeatsWithLocation : public RegionFeats {
 public:
  typedef RegionFeatsWithLocation Self;
  typedef RegionFeats Super;
  std::shared_ptr<glia::feat::RegionLocationFeats> location;
  std::shared_ptr<glia::feat::RegionAdvShapeFeats2D> ashape;  // 2D only

  ~RegionFeatsWithLocation () override {}

  virtual uint dim () const {
    uint ret = Super::dim();
    // if (location) { ret += location->dim; }
    if (ashape) { ret += ashape->dim; }
    return ret;
  }

  virtual void serialize (std::vector<FVal>& feats) const {
    feats.reserve(feats.size() + dim());
    Super::serialize(feats);
    // if (location) { location->serialize(feats); }
    if (ashape) { ashape->serialize(feats); }
  }

  friend std::ostream& operator<< (std::ostream& os, Self const& f) {
    os << static_cast<Super const&>(f);
    // if (f.location) { os << " " << *f.location; }
    if (f.ashape) { os << " " << *f.ashape; }
    return os;
  }

  template <typename TPoints, typename TRImagePtr, typename TLImagePtr>
  void generate (
      TPoints const& region, double normalizingArea,
      double normalizingLength, TRImagePtr const& pbImage,
      std::vector<double> const& boundaryThresholds,
      std::vector<ImageHistPair<TRImagePtr>> const& rImages,
      std::vector<ImageHistPair<TLImagePtr>> const& rlImages,
      std::vector<ImageHistPair<TRImagePtr>> const& bImages,
      double const* pSaliency) {
    Super::generate(
        region, normalizingArea, normalizingLength, pbImage,
        boundaryThresholds, rImages, rlImages, bImages, pSaliency);
    location = std::make_shared<glia::feat::RegionLocationFeats>(
        TPoints::Point::Dimension);
    location->generate(region, normalizingLength);
    // Advanced shape features only support 2D for now
    fPoint<2> c{location->centroid[0], location->centroid[1]};
    ashape = std::make_shared<glia::feat::RegionAdvShapeFeats2D>();
    ashape->generate(region, c, normalizingLength);
  }
};


class RegionPairFeats : public Object {
 public:
  typedef RegionPairFeats Self;
  std::shared_ptr<glia::feat::RegionShapeDiffFeats> shapeDiff;
  std::shared_ptr<glia::feat::RegionLocationDiffFeats> locationDiff;
  std::shared_ptr<glia::feat::RegionSetDiffFeats> setDiff;
  std::shared_ptr<glia::feat::RegionAdvShapeFeatsDiff2D> ashapeDiff; // 2D
  std::vector<std::shared_ptr<glia::feat::ImageDiffFeats>> imageRegionDiff;
  std::vector<std::shared_ptr<glia::feat::ImageLabelDiffFeats>>
  labelImageRegionDiff;

  ~RegionPairFeats () override {}

  virtual uint dim () const {
    uint ret = 0;
    if (shapeDiff) { ret += shapeDiff->dim; }
    if (locationDiff) { ret += locationDiff->dim; }
    if (setDiff) { ret += setDiff->dim; }
    if (ashapeDiff) { ret += ashapeDiff->dim; }
    for (auto const& p : imageRegionDiff) { if (p) { ret += p->dim; } }
    for (auto const& p : labelImageRegionDiff)
    { if (p) { ret += p->dim; } }
    return ret;
  }

  virtual void log () {
    if (shapeDiff) { shapeDiff->log(); }
    if (locationDiff) { locationDiff->log(); }
    if (setDiff) { setDiff->log(); }
    if (ashapeDiff) { ashapeDiff->log(); }
  }

  virtual void serialize (std::vector<FVal>& feats) const {
    feats.reserve(feats.size() + dim());
    if (shapeDiff) { shapeDiff->serialize(feats); }
    if (locationDiff) { locationDiff->serialize(feats); }
    if (setDiff) { setDiff->serialize(feats); }
    if (ashapeDiff) { ashapeDiff->serialize(feats); }
    for (auto const& p : imageRegionDiff)
    { if (p) { p->serialize(feats); } }
    for (auto const& p : labelImageRegionDiff)
    { if (p) { p->serialize(feats); } }
  }

  friend std::ostream& operator<< (std::ostream& os, Self const& f) {
    if (f.shapeDiff) { os << *f.shapeDiff; }
    if (f.locationDiff) { os << " " << *f.locationDiff; }
    if (f.setDiff) { os << " " << *f.setDiff; }
    if (f.ashapeDiff) { os << " " << *f.ashapeDiff; }
    for (auto const& p : f.imageRegionDiff) { if (p) { os << " " << *p; } }
    for (auto const& p : f.labelImageRegionDiff)
    { if (p) { os << " " << *p; } }
    return os;
  }

  void generate (
      RegionFeatsWithLocation const& rf0,
      RegionFeatsWithLocation const& rf1, double ov) {
    // Shape
    shapeDiff = std::make_shared<glia::feat::RegionShapeDiffFeats>();
    shapeDiff->generate(*rf0.shape, *rf1.shape);
    locationDiff = std::make_shared<glia::feat::RegionLocationDiffFeats>();
    locationDiff->generate(*rf0.location, *rf1.location);
    setDiff = std::make_shared<glia::feat::RegionSetDiffFeats>();
    setDiff->generate(ov, rf0.shape->area, rf1.shape->area);
    ashapeDiff =
        std::make_shared<glia::feat::RegionAdvShapeFeatsDiff2D>();
    ashapeDiff->generate(*rf0.ashape, *rf1.ashape);
    // Image appearance
    int n = rf0.region.size();
    imageRegionDiff.reserve(n);
    for (int i = 0; i < n; ++i) {
      imageRegionDiff.push_back(
          std::make_shared<glia::feat::ImageDiffFeats>());
      imageRegionDiff.back()->generate(*rf0.region[i], *rf1.region[i]);
    }
    n = rf0.labelRegion.size();
    labelImageRegionDiff.reserve(n);
    for (int i = 0; i < n; ++i) {
      labelImageRegionDiff.push_back(
          std::make_shared<glia::feat::ImageDiffFeats>());
      labelImageRegionDiff.back()->generate(
          *rf0.region[i], *rf1.region[i]);
    }
  }
};


class SectionClassificationFeats
    : public virtual TTriple<
  RegionPairFeats, RegionFeatsWithLocation const*> {
 public:
  typedef TTriple<RegionPairFeats, RegionFeatsWithLocation const*> Super;
  typedef SectionClassificationFeats Self;

  ~SectionClassificationFeats () override {}

  virtual uint dim () const
  { return Super::x0.dim() + Super::x1->dim() + Super::x2->dim(); }

  virtual void serialize (std::vector<FVal>& feats) const {
    feats.reserve(feats.size() + dim());
    Super::x0.serialize(feats);
    Super::x1->serialize(feats);
    Super::x2->serialize(feats);
  }

  friend std::ostream& operator<< (std::ostream& os, Self const& f)
  { return os << f.x0 << " " << *f.x1 << " " << *f.x2; }
};

};
};

#endif

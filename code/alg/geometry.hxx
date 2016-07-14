#ifndef _glia_alg_geometry_hxx_
#define _glia_alg_geometry_hxx_

#include "type/point.hxx"
#include "type/region.hxx"

namespace glia {
namespace alg {

template <typename TRegion> void
getCentroid (fPoint<TRegion::Point::Dimension>& c, TRegion const& region)
{
  typedef typename TRegion::Point Point;
  c.Fill(0.0);
  region.traverse([&c](Point const& p){ c += p; });
  double n = region.size();
  for (int i = 0; i < Point::Dimension; ++i) { c[i] /= n; }
}


template <typename TRegion> void
getBoundingBox (
    BoundingBox<TRegion::Point::Dimension>& bbox, TRegion const& region)
{
  typedef BoundingBox<TRegion::Point::Dimension> BBox;
  itk::Index<BBox::ImageDimension>
      lower(region.begin()->second->front()),
      upper(region.begin()->second->front());
  region.traverse
      ([&lower, &upper](typename TRegion::Point const& p) {
        for (int i = 0; i < BBox::ImageDimension; ++i) {
          if (p[i] < lower[i]) { lower[i] = p[i]; }
          else if (p[i] > upper[i]) { upper[i] = p[i]; }
        }
      });
  bbox.GetModifiableIndex() = lower;
  for (auto i = 0; i < BBox::ImageDimension; ++i)
  { bbox.GetModifiableSize()[i] = upper[i] - lower[i]; }
}


// Compute central moments of certain orders
// Output moments: {m02, m03, m11, m12, m20, m21, m30}
template <typename TKey> inline void
getCentralMoments (
    std::array<double, 7>& ms, TRegion<TKey, Point<2>> const& region,
    fPoint<2> const& c)
{
  ms.fill(0.0);
  region.traverse([&ms, &c](Point<2> const& p) {
      double dx = p[0] - c[0], dy = p[1] - c[1];
      double dx2 = dx * dx, dy2 = dy * dy;
      ms[0] += dy2;
      ms[1] += dy2 * dy;
      ms[2] += dx * dy;
      ms[3] += dx * dy2;
      ms[4] += dx2;
      ms[5] += dx2 * dy;
      ms[6] += dx2 * dx;
    });
}


inline void getScaleInvariantMoments (
    std::array<double, 7>& sims, double m00,
    std::array<double, 7> const& ms)
{
  double m002 = m00 * m00, m003 = std::pow(m00, 2.5);
  sims[0] = sdivide(ms[0], m002, 0.0);
  sims[1] = sdivide(ms[1], m003, 0.0);
  sims[2] = sdivide(ms[2], m002, 0.0);
  sims[3] = sdivide(ms[3], m003, 0.0);
  sims[4] = sdivide(ms[4], m002, 0.0);
  sims[5] = sdivide(ms[5], m003, 0.0);
  sims[6] = sdivide(ms[6], m003, 0.0);
}


// Input moments: {m02, m03, m11, m12, m20, m21, m30}
// Input moments must be scale invariant
inline void getHuMoments (
    std::array<double, 7>& hm, std::array<double, 7> const& sims)
{
  double m02 = sims[0], m03 = sims[1], m11 = sims[2],
      m12 = sims[3], m20 = sims[4], m21 = sims[5], m30 = sims[6];
  hm[0] = m20 + m02;
  hm[1] = std::pow(m20 - m02, 2) + 4.0 * m11 * m11;
  hm[2] = std::pow(m30 - 3.0 * m12, 2) + std::pow(3.0 * m21 - m03, 2);
  hm[3] = std::pow(m30 + m12, 2) + std::pow(m21 + m03, 2);
  hm[4] = (m30 - 3.0 * m12) * (m30 + m12) *
      (std::pow(m30 + m12, 2) - 3.0 * std::pow(m21 + m03, 2)) +
      (3.0 * m21 - m03) * (m21 + m03) *
      (3.0 * std::pow(m30 + m12, 2) - std::pow(m21 + m03, 2));
  hm[5] = (m20 - m02) * (std::pow(m30 + m12, 2) -
                         std::pow(m21 + m03, 2)) +
      4.0 * m11 * (m30 + m12) * (m03 + m21);
  hm[6] = (3.0 * m21 - m03) * (m12 + m30) *
      (std::pow(m30 + m12, 2) - 3.0 * std::pow(m21 + m03, 2)) -
      (m30 - 3.0 * m12) * (m12 + m03) *
      (3.0 * std::pow(m30 + m12, 2) - std::pow(m21 + m03, 2));
}


inline double getEccentricity (double m02, double m11, double m20)
{
  double a = m20 + m02;
  double b = ssqrt(std::pow(m20 - m02, 2) + 4.0 * m11 * m11, 0.0);
  // return (a + b) / (a - b + FEPS);
  return sdivide(a + b, a - b, 0.0);
}

};
};

#endif

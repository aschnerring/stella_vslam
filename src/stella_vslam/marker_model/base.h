#ifndef STELLA_VSLAM_MARKER_MODEL_BASE_H
#define STELLA_VSLAM_MARKER_MODEL_BASE_H

#include "stella_vslam/type.h"

#include <string>
#include <limits>

#include <yaml-cpp/yaml.h>
#include <nlohmann/json_fwd.hpp>

namespace stella_vslam {
namespace marker_model {

class base {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    //! Constructor
    explicit base(double width, int max_id_marker);

    //! Destructor
    virtual ~base();

    //! marker geometry
    const double width_;
    const int max_id_marker;
    eigen_alloc_vector<Vec3_t> corners_pos_;

    //! Encode marker_model information as JSON
    virtual nlohmann::json to_json() const;
};

std::ostream& operator<<(std::ostream& os, const base& params);

} // namespace marker_model
} // namespace stella_vslam

#endif // STELLA_VSLAM_MARKER_MODEL_BASE_H

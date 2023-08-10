#include "stella_vslam/data/marker.h"
#include "stella_vslam/data/keyframe.h"
#include <nlohmann/json.hpp> 

namespace stella_vslam {
namespace data {

marker::marker(const eigen_alloc_vector<Vec3_t>& corners_pos_w, unsigned int id, const std::shared_ptr<marker_model::base>& marker_model)
    : corners_pos_w_(corners_pos_w), id_(id), marker_model_(marker_model) {}

void marker::set_corner_pos(const eigen_alloc_vector<Vec3_t>& corner_pos_w) {
    std::lock_guard<std::mutex> lock(mtx_position_);
    corners_pos_w_ = corner_pos_w;
}

nlohmann::json marker::to_json() const {
    nlohmann::json marker_json;
    marker_json["id"] = id_;

    // Convert corners_pos_w_ vector to a JSON array
    nlohmann::json corners_json = nlohmann::json::array();
    for (const auto& corner : corners_pos_w_) {
        corners_json.push_back({corner(0), corner(1), corner(2)});
    }
    marker_json["corners"] = corners_json;

    // Add keyframe information
    nlohmann::json keyframes_json = nlohmann::json::array();
    for (const auto& keyframe : observations_) {
        nlohmann::json keyframe_json;
        keyframe_json["id"] = keyframe->id_;
        keyframes_json.push_back(keyframe_json);
    }
    marker_json["keyframes"] = keyframes_json;

    return marker_json;
}

} // namespace data
} // namespace stella_vslam

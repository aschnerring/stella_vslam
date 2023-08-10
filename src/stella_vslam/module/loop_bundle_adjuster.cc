#include "stella_vslam/mapping_module.h"
#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/data/landmark.h"
#include "stella_vslam/data/marker.h"
#include "stella_vslam/data/map_database.h"
#include "stella_vslam/marker_model/base.h"
#include "stella_vslam/module/loop_bundle_adjuster.h"
#include "stella_vslam/optimize/global_bundle_adjuster.h"

#include <thread>

#include <spdlog/spdlog.h>

namespace stella_vslam {
namespace module {

loop_bundle_adjuster::loop_bundle_adjuster(data::map_database* map_db, const unsigned int num_iter)
    : map_db_(map_db), num_iter_(num_iter) {}

void loop_bundle_adjuster::set_mapping_module(mapping_module* mapper) {
    mapper_ = mapper;
}

void loop_bundle_adjuster::abort() {
    std::lock_guard<std::mutex> lock(mtx_thread_);
    abort_loop_BA_ = true;
}

bool loop_bundle_adjuster::is_running() const {
    std::lock_guard<std::mutex> lock(mtx_thread_);
    return loop_BA_is_running_;
}

void loop_bundle_adjuster::optimize(const std::shared_ptr<data::keyframe>& curr_keyfrm) {
    spdlog::info("start loop bundle adjustment");

    {
        std::lock_guard<std::mutex> lock(mtx_thread_);
        loop_BA_is_running_ = true;
        abort_loop_BA_ = false;
    }

    std::unordered_set<unsigned int> optimized_keyfrm_ids;
    std::unordered_set<unsigned int> optimized_landmark_ids;
    std::unordered_set<unsigned int> optimized_markers_ids;
    eigen_alloc_unord_map<unsigned int, Vec3_t> lm_to_pos_w_after_global_BA;
    eigen_alloc_unord_map<unsigned int, std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>>> mkr_to_pos_w_after_global_BA;
    eigen_alloc_unord_map<unsigned int, Mat44_t> keyfrm_to_pose_cw_after_global_BA;
    const auto global_BA = optimize::global_bundle_adjuster(num_iter_, false);
    bool ok = global_BA.optimize(curr_keyfrm->graph_node_->get_keyframes_from_root(),
                                 optimized_keyfrm_ids, optimized_landmark_ids, optimized_markers_ids,
                                 lm_to_pos_w_after_global_BA,
                                 mkr_to_pos_w_after_global_BA,
                                 keyfrm_to_pose_cw_after_global_BA, &abort_loop_BA_);

    {
        std::lock_guard<std::mutex> lock1(mtx_thread_);

        // if the loop BA was aborted, cannot update the map
        if (!ok) {
            spdlog::info("abort loop bundle adjustment");
            loop_BA_is_running_ = false;
            abort_loop_BA_ = false;
            return;
        }

        spdlog::info("finish loop bundle adjustment");
        spdlog::info("updating the map with pose propagation");

        // stop mapping module
        auto future_pause = mapper_->async_pause();
        spdlog::debug("loop_bundle_adjuster::optimize: wait for mapper_->async_pause");
        future_pause.get();

        std::lock_guard<std::mutex> lock2(data::map_database::mtx_database_);

        spdlog::debug("update the camera pose along the spanning tree from the root");
        eigen_alloc_unord_map<unsigned int, Mat44_t> keyfrm_to_cam_pose_cw_before_BA;
        std::list<std::shared_ptr<data::keyframe>> keyfrms_to_check;
        keyfrms_to_check.push_back(curr_keyfrm->graph_node_->get_spanning_root());
        while (!keyfrms_to_check.empty()) {
            auto parent = keyfrms_to_check.front();
            const Mat44_t cam_pose_wp = parent->get_pose_wc();

            const auto children = parent->graph_node_->get_spanning_children();
            for (auto child : children) {
                if (!optimized_keyfrm_ids.count(child->id_)) {
                    // if `child` is NOT optimized by the loop BA
                    // propagate the pose correction from the spanning parent

                    // parent->child
                    const Mat44_t cam_pose_cp = child->get_pose_cw() * cam_pose_wp;
                    // world->child AFTER correction = parent->child * world->parent AFTER correction
                    keyfrm_to_pose_cw_after_global_BA[child->id_] = cam_pose_cp * keyfrm_to_pose_cw_after_global_BA.at(parent->id_);
                    // check as `child` has been corrected
                    optimized_keyfrm_ids.insert(child->id_);
                }

                // need updating
                keyfrms_to_check.push_back(child);
            }

            // temporally store the camera pose BEFORE correction (for correction of landmark positions)
            keyfrm_to_cam_pose_cw_before_BA[parent->id_] = parent->get_pose_cw();
            // update the camera pose
            parent->set_pose_cw(keyfrm_to_pose_cw_after_global_BA.at(parent->id_));
            // finish updating
            keyfrms_to_check.pop_front();
        }

        spdlog::debug("update the positions of the landmarks");
        auto keyfrms = curr_keyfrm->graph_node_->get_keyframes_from_root();
        std::unordered_set<unsigned int> already_found_landmark_ids;
        std::vector<std::shared_ptr<data::landmark>> lms;
        for (const auto& keyfrm : keyfrms) {
            for (const auto& lm : keyfrm->get_landmarks()) {
                if (!lm) {
                    continue;
                }
                if (lm->will_be_erased()) {
                    continue;
                }
                if (already_found_landmark_ids.count(lm->id_)) {
                    continue;
                }

                already_found_landmark_ids.insert(lm->id_);
                lms.push_back(lm);
            }
        }

        for (const auto& lm : lms) {
            if (lm->will_be_erased()) {
                continue;
            }

            if (optimized_landmark_ids.count(lm->id_)) {
                // if `lm` is optimized by the loop BA

                // update with the optimized position
                lm->set_pos_in_world(lm_to_pos_w_after_global_BA.at(lm->id_));
            }
            else {
                // if `lm` is NOT optimized by the loop BA

                // correct the position according to the move of the camera pose of the reference keyframe
                auto ref_keyfrm = lm->get_ref_keyframe();

                assert(optimized_keyfrm_ids.count(ref_keyfrm->id_));

                // convert the position to the camera-reference using the camera pose BEFORE the correction
                const Mat44_t pose_cw_before_BA = keyfrm_to_cam_pose_cw_before_BA.at(ref_keyfrm->id_);
                const Mat33_t rot_cw_before_BA = pose_cw_before_BA.block<3, 3>(0, 0);
                const Vec3_t trans_cw_before_BA = pose_cw_before_BA.block<3, 1>(0, 3);
                const Vec3_t pos_c = rot_cw_before_BA * lm->get_pos_in_world() + trans_cw_before_BA;

                // convert the position to the world-reference using the camera pose AFTER the correction
                const Mat44_t cam_pose_wc = ref_keyfrm->get_pose_wc();
                const Mat33_t rot_wc = cam_pose_wc.block<3, 3>(0, 0);
                const Vec3_t trans_wc = cam_pose_wc.block<3, 1>(0, 3);
                lm->set_pos_in_world(rot_wc * pos_c + trans_wc);
            }
            lm->update_mean_normal_and_obs_scale_variance();
        }

        // 2.1. add markers
        std::unordered_set<unsigned int> already_found_marker_ids;
        std::vector<std::shared_ptr<data::marker>> markers;
        for (const auto& keyfrm : keyfrms) {
            const auto ngh_markers = keyfrm->get_markers();
            for (const auto& mkr : ngh_markers) {
                if (!mkr) {
                    continue;
                }
                if (already_found_marker_ids.count(mkr->id_)) {
                    continue;
                }

                already_found_marker_ids.insert(mkr->id_);
                markers.push_back(mkr);
            }
        }

        for (const auto& mkr : markers) {
            if (!mkr) {
                continue;
            }

            if (optimized_markers_ids.count(mkr->id_)) {
                // if `mkr` is optimized by the loop BA

                // update with the optimized position
                mkr->set_corner_pos(mkr_to_pos_w_after_global_BA.at(mkr->id_));
            }
            else{
                // if `mkr` is NOT optimized by the loop BA

                if (mkr->observations_.empty()) {
                    // Handle the case when observations_ is empty, e.g., continue or skip this marker
                    continue;
                }
                // correct the position according to the move of the camera pose of the reference keyframe
                auto ref_keyfrm = mkr->observations_[0];
                assert(optimized_keyfrm_ids.count(ref_keyfrm->id_));

                // convert the position to the camera-reference using the camera pose BEFORE the correction
                const Mat44_t pose_cw_before_BA = keyfrm_to_cam_pose_cw_before_BA.at(ref_keyfrm->id_);
                const Mat33_t rot_cw_before_BA = pose_cw_before_BA.block<3, 3>(0, 0);
                const Vec3_t trans_cw_before_BA = pose_cw_before_BA.block<3, 1>(0, 3);

                // Create a vector to store the updated corner positions
                std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>> corners_pose_w;
                corners_pose_w.reserve(4);

                for (unsigned int corner_id = 0; corner_id < 4; ++corner_id) {
                    const Vec3_t pos_w_before_correction = mkr->corners_pos_w_[corner_id];
                    const Vec3_t pos_c = rot_cw_before_BA * pos_w_before_correction + trans_cw_before_BA;
                    // convert the position to the world-reference using the camera pose AFTER the correction
                    const Mat44_t cam_pose_wc = ref_keyfrm->get_pose_wc();
                    const Mat33_t rot_wc = cam_pose_wc.block<3, 3>(0, 0);
                    const Vec3_t trans_wc = cam_pose_wc.block<3, 1>(0, 3);
                    const Vec3_t pos_w = rot_wc * pos_c + trans_wc;
                    // Add the updated corner position to the vector
                    corners_pose_w.push_back(pos_w);
                }
                // Set the updated corner positions in the marker
                mkr->set_corner_pos(corners_pose_w); //updated_corner_pos 
            }
        }


        mapper_->resume();
        loop_BA_is_running_ = false;

        spdlog::info("updated the map");
    }
}

} // namespace module
} // namespace stella_vslam

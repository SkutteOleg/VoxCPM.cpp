#include "voxcpm/runtime.h"

namespace voxcpm {

bool VoxCPMRuntimeSkeleton::initialize(const SharedWeightStore& weights,
                                       VoxCPMBackend& backend,
                                       const VoxCPMConfig& config,
                                       PlacementPolicy placement) {
    if (!weights || !weights->owns_storage()) {
        return false;
    }

    weights_ = weights;
    backend_ = &backend;
    config_ = config;
    placement_ = placement;
    return true;
}

VoxCPMPersistentState VoxCPMRuntimeSkeleton::create_persistent_state() const {
    VoxCPMPersistentState state;
    if (!backend_) {
        return state;
    }

    const PersistentStateShape shape = {
        config_.base_lm.hidden_size,
        config_.feat_dim,
        config_.patch_size,
    };
    state.initialize(*backend_, shape);
    return state;
}

VoxCPMOutputPool VoxCPMRuntimeSkeleton::create_output_pool(int max_latent_patches) const {
    VoxCPMOutputPool pool;
    if (!backend_ || max_latent_patches <= 0) {
        return pool;
    }

    const OutputPoolShape shape = {
        config_.feat_dim,
        config_.patch_size,
        max_latent_patches,
    };
    pool.initialize(*backend_, shape);
    return pool;
}

}  // namespace voxcpm

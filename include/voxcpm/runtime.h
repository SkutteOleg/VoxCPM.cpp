#ifndef VOXCPM_RUNTIME_H
#define VOXCPM_RUNTIME_H

#include "voxcpm/backend.h"
#include "voxcpm/config.h"
#include "voxcpm/graph-cache.h"
#include "voxcpm/output.h"
#include "voxcpm/state.h"
#include "voxcpm/weight-store.h"

namespace voxcpm {

struct PlacementPolicy {
    BackendType default_backend = BackendType::CPU;
    bool allow_transformer_offload = true;
    bool allow_audiovae_offload = false;
    bool allow_scheduler = true;
    bool require_host_visible_output = true;
};

class VoxCPMRuntimeSkeleton {
public:
    VoxCPMRuntimeSkeleton() = default;

    bool initialize(const SharedWeightStore& weights,
                    VoxCPMBackend& backend,
                    const VoxCPMConfig& config,
                    PlacementPolicy placement = {});

    bool is_initialized() const { return backend_ != nullptr && static_cast<bool>(weights_); }

    const SharedWeightStore& weights() const { return weights_; }
    VoxCPMBackend* backend() const { return backend_; }
    const VoxCPMConfig& config() const { return config_; }
    const PlacementPolicy& placement_policy() const { return placement_; }

    VoxCPMPersistentState create_persistent_state() const;
    VoxCPMOutputPool create_output_pool(int max_latent_patches) const;

    VoxCPMGraphCache& graph_cache() { return graph_cache_; }
    const VoxCPMGraphCache& graph_cache() const { return graph_cache_; }
    void clear_graph_cache() { graph_cache_.clear(); }

private:
    SharedWeightStore weights_;
    VoxCPMBackend* backend_ = nullptr;
    VoxCPMConfig config_;
    PlacementPolicy placement_;
    VoxCPMGraphCache graph_cache_;
};

}  // namespace voxcpm

#endif  // VOXCPM_RUNTIME_H

#ifndef VOXCPM_GRAPH_CACHE_H
#define VOXCPM_GRAPH_CACHE_H

#include "voxcpm/context.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace voxcpm {

struct VoxCPMGraphKey {
    std::string stage;
    int seq_len = 0;
    int n_timesteps = 0;
    int position = -1;
    int cfg_milli = 0;
    bool is_prefill = false;
    bool with_prefix = false;

    bool operator==(const VoxCPMGraphKey& other) const {
        return stage == other.stage &&
               seq_len == other.seq_len &&
               n_timesteps == other.n_timesteps &&
               position == other.position &&
               cfg_milli == other.cfg_milli &&
               is_prefill == other.is_prefill &&
               with_prefix == other.with_prefix;
    }
};

struct VoxCPMGraphKeyHash {
    size_t operator()(const VoxCPMGraphKey& key) const;
};

struct VoxCPMGraphArtifacts {
    std::unique_ptr<VoxCPMContext> context;
    ggml_cgraph* graph = nullptr;
    ggml_tensor* input0 = nullptr;
    ggml_tensor* input1 = nullptr;
    ggml_tensor* input2 = nullptr;
    ggml_tensor* input3 = nullptr;
    ggml_tensor* input4 = nullptr;
    ggml_tensor* output = nullptr;
    ggml_tensor* aux_output0 = nullptr;

    void clear();
};

class VoxCPMGraphCache {
public:
    VoxCPMGraphCache() = default;

    bool contains(const VoxCPMGraphKey& key) const;
    VoxCPMGraphArtifacts* find(const VoxCPMGraphKey& key);
    const VoxCPMGraphArtifacts* find(const VoxCPMGraphKey& key) const;
    VoxCPMGraphArtifacts& get_or_create(const VoxCPMGraphKey& key);
    void clear();
    size_t size() const { return entries_.size(); }

private:
    std::unordered_map<VoxCPMGraphKey, VoxCPMGraphArtifacts, VoxCPMGraphKeyHash> entries_;
};

}  // namespace voxcpm

#endif  // VOXCPM_GRAPH_CACHE_H

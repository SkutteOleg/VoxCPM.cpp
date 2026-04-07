#include "voxcpm/graph-cache.h"

#include <functional>

namespace voxcpm {

namespace {

template <typename T>
inline void hash_combine(size_t& seed, const T& value) {
    seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

}  // namespace

size_t VoxCPMGraphKeyHash::operator()(const VoxCPMGraphKey& key) const {
    size_t seed = 0;
    hash_combine(seed, key.stage);
    hash_combine(seed, key.seq_len);
    hash_combine(seed, key.n_timesteps);
    hash_combine(seed, key.position);
    hash_combine(seed, key.cfg_milli);
    hash_combine(seed, key.is_prefill);
    hash_combine(seed, key.with_prefix);
    return seed;
}

void VoxCPMGraphArtifacts::clear() {
    context.reset();
    graph = nullptr;
    input0 = nullptr;
    input1 = nullptr;
    input2 = nullptr;
    input3 = nullptr;
    input4 = nullptr;
    output = nullptr;
    aux_output0 = nullptr;
}

bool VoxCPMGraphCache::contains(const VoxCPMGraphKey& key) const {
    return entries_.find(key) != entries_.end();
}

VoxCPMGraphArtifacts* VoxCPMGraphCache::find(const VoxCPMGraphKey& key) {
    auto it = entries_.find(key);
    return it == entries_.end() ? nullptr : &it->second;
}

const VoxCPMGraphArtifacts* VoxCPMGraphCache::find(const VoxCPMGraphKey& key) const {
    auto it = entries_.find(key);
    return it == entries_.end() ? nullptr : &it->second;
}

VoxCPMGraphArtifacts& VoxCPMGraphCache::get_or_create(const VoxCPMGraphKey& key) {
    return entries_[key];
}

void VoxCPMGraphCache::clear() {
    entries_.clear();
}

}  // namespace voxcpm

#include "test_config.h"
#include "voxcpm/audio-vae.h"
#include "voxcpm/audio_io.h"
#include "voxcpm/backend.h"
#include "voxcpm/context.h"
#include "voxcpm/graph-cache.h"
#include "voxcpm/output.h"
#include "voxcpm/runtime.h"
#include "voxcpm/state.h"
#include "voxcpm/tokenizer.h"
#include "voxcpm/voxcpm.h"
#include "voxcpm/weight-store.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <memory>

using namespace voxcpm;

namespace {

struct ScopedEnvVar {
    explicit ScopedEnvVar(const char* name) : name(name) {
        const char* current = std::getenv(name);
        if (current != nullptr) {
            had_value = true;
            value = current;
        }
    }

    ~ScopedEnvVar() {
        if (had_value) {
            setenv(name, value.c_str(), 1);
        } else {
            unsetenv(name);
        }
    }

    void set(const char* next) const {
        setenv(name, next, 1);
    }

    const char* name = nullptr;
    bool had_value = false;
    std::string value;
};

std::vector<float> read_persistent_tensor(VoxCPMBackend& backend,
                                          const VoxCPMPersistentState& state,
                                          ggml_tensor* tensor) {
    std::vector<float> out(static_cast<size_t>(ggml_nelements(tensor)), 0.0f);
    if (tensor == state.lm_hidden()) {
        REQUIRE(state.get_lm_hidden_to_host(backend, out.data(), out.size()));
    } else if (tensor == state.residual_hidden()) {
        REQUIRE(state.get_residual_hidden_to_host(backend, out.data(), out.size()));
    } else {
        REQUIRE(tensor == state.prefix_patch());
        REQUIRE(state.get_prefix_patch_to_host(backend, out.data(), out.size()));
    }
    return out;
}

std::vector<float> slice_last_column_major_2d(const std::vector<float>& input,
                                              int row_dim,
                                              int col_idx) {
    std::vector<float> out(static_cast<size_t>(row_dim), 0.0f);
    const size_t offset = static_cast<size_t>(col_idx) * static_cast<size_t>(row_dim);
    std::copy_n(input.data() + offset, row_dim, out.data());
    return out;
}

std::vector<float> extract_prompt_patch_range(const std::vector<float>& feat,
                                              const std::vector<int32_t>& feat_mask,
                                              size_t patch_elem_count) {
    std::vector<float> out;
    for (size_t t = 0; t < feat_mask.size(); ++t) {
        if (feat_mask[t] == 0) {
            continue;
        }
        const size_t offset = t * patch_elem_count;
        out.insert(out.end(), feat.begin() + static_cast<std::ptrdiff_t>(offset),
                   feat.begin() + static_cast<std::ptrdiff_t>(offset + patch_elem_count));
    }
    return out;
}

std::vector<float> apply_feat_mask_columns(const std::vector<float>& input,
                                           const std::vector<float>& feat_mask,
                                           int hidden_size) {
    REQUIRE(input.size() == feat_mask.size() * static_cast<size_t>(hidden_size));

    std::vector<float> output(input.size(), 0.0f);
    for (size_t t = 0; t < feat_mask.size(); ++t) {
        const float scale = feat_mask[t];
        const size_t offset = t * static_cast<size_t>(hidden_size);
        for (int h = 0; h < hidden_size; ++h) {
            output[offset + static_cast<size_t>(h)] = scale * input[offset + static_cast<size_t>(h)];
        }
    }
    return output;
}

std::vector<float> apply_residual_bridge(VoxCPMRuntime& runtime,
                                         VoxCPMBackend& backend,
                                         const std::vector<float>& blended,
                                         const std::vector<float>& feat_embed_part,
                                         int seq_len) {
    const int hidden_size = runtime.base_lm().config().hidden_size;
    REQUIRE(blended.size() == feat_embed_part.size());
    REQUIRE(blended.size() == static_cast<size_t>(hidden_size * seq_len));

    if (runtime.components()->fusion_concat_proj() == nullptr) {
        std::vector<float> residual_inputs = blended;
        for (size_t i = 0; i < residual_inputs.size(); ++i) {
            residual_inputs[i] += feat_embed_part[i];
        }
        return residual_inputs;
    }

    VoxCPMContext graph_ctx(ContextType::Graph, 4096, 32768);
    ggml_tensor* input = graph_ctx.new_tensor_2d(GGML_TYPE_F32, hidden_size * 2, seq_len);
    ggml_set_input(input);
    ggml_tensor* output = runtime.components()->fusion_concat_proj()->forward(graph_ctx, input);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, output);
    backend.reserve_compute_memory(graph, "test.runtime.residual_bridge");
    backend.alloc_graph(graph, "test.runtime.residual_bridge");

    std::vector<float> fused_input(static_cast<size_t>(hidden_size * 2 * seq_len), 0.0f);
    for (int t = 0; t < seq_len; ++t) {
        const size_t src_offset = static_cast<size_t>(t) * static_cast<size_t>(hidden_size);
        const size_t dst_offset = static_cast<size_t>(t) * static_cast<size_t>(hidden_size * 2);
        std::copy_n(blended.data() + src_offset, hidden_size, fused_input.data() + dst_offset);
        std::copy_n(feat_embed_part.data() + src_offset,
                    hidden_size,
                    fused_input.data() + dst_offset + static_cast<size_t>(hidden_size));
    }

    backend.tensor_set(input, fused_input.data(), 0, fused_input.size() * sizeof(float));
    REQUIRE(backend.compute(graph) == GGML_STATUS_SUCCESS);

    std::vector<float> output_data(blended.size(), 0.0f);
    backend.tensor_get(output, output_data.data(), 0, output_data.size() * sizeof(float));
    return output_data;
}

std::vector<float> build_expected_residual_inputs(VoxCPMRuntime& runtime,
                                                  VoxCPMBackend& backend,
                                                  const std::vector<float>& combined_embed,
                                                  const std::vector<float>& blended,
                                                  const std::vector<float>& feat_mask,
                                                  int seq_len) {
    const int hidden_size = runtime.base_lm().config().hidden_size;
    const std::vector<float> feat_embed_part = apply_feat_mask_columns(combined_embed, feat_mask, hidden_size);
    return apply_residual_bridge(runtime, backend, blended, feat_embed_part, seq_len);
}

std::vector<uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    REQUIRE(in.is_open());
    const std::streamsize size = in.tellg();
    REQUIRE(size >= 0);
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(size), 0);
    in.read(reinterpret_cast<char*>(bytes.data()), size);
    REQUIRE((in.good() || in.eof()));
    return bytes;
}

struct RealPromptInputs {
    std::vector<int32_t> full_text_tokens;
    std::vector<int32_t> text_mask;
    std::vector<int32_t> feat_mask;
    std::vector<float> feat;
    int seq_len = 0;
};

enum class TestPaddingMode {
    Left,
    Right,
};

std::vector<float> encode_test_audio_features(VoxCPMRuntime& runtime,
                                              AudioVAE& audio_vae,
                                              VoxCPMBackend& backend,
                                              const std::string& audio_path,
                                              TestPaddingMode padding_mode) {
    const std::vector<uint8_t> wav_bytes = read_file_bytes(audio_path);
    const DecodedAudio decoded = decode_audio_from_memory(wav_bytes.data(), wav_bytes.size());
    REQUIRE(decoded.sample_rate > 0);
    std::vector<float> mono = convert_to_mono(decoded);
    mono = resample_audio_to_rate(mono, decoded.sample_rate, audio_vae.config().sample_rate);
    mono = trim_audio_silence_vad(mono, audio_vae.config().sample_rate);

    const int patch_len = runtime.config().patch_size * audio_vae.config().hop_length();
    if (mono.size() % static_cast<size_t>(patch_len) != 0) {
        const size_t padding = static_cast<size_t>(patch_len) - (mono.size() % static_cast<size_t>(patch_len));
        if (padding_mode == TestPaddingMode::Left) {
            mono.insert(mono.begin(), padding, 0.0f);
        } else {
            mono.insert(mono.end(), padding, 0.0f);
        }
    }

    VoxCPMContext graph_ctx(ContextType::Graph, 32768, 262144);
    ggml_tensor* latent = audio_vae.encode(graph_ctx, backend, mono, audio_vae.config().sample_rate);
    REQUIRE(latent != nullptr);
    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, latent);
    backend.reserve_compute_memory(graph, "test.audio_vae.encode");
    backend.alloc_graph(graph, "test.audio_vae.encode");
    const auto& preprocessed = audio_vae.last_preprocessed_audio();
    backend.tensor_set(audio_vae.last_input_tensor(), preprocessed.data(), 0, preprocessed.size() * sizeof(float));
    REQUIRE(backend.compute(graph) == GGML_STATUS_SUCCESS);

    const int total_patches = static_cast<int>(latent->ne[0]);
    const int feat_dim = runtime.config().feat_dim;
    const int patch_size = runtime.config().patch_size;
    REQUIRE(static_cast<int>(latent->ne[1]) == feat_dim);
    REQUIRE(total_patches % patch_size == 0);

    std::vector<float> encoded(static_cast<size_t>(total_patches) * feat_dim, 0.0f);
    backend.tensor_get(latent, encoded.data(), 0, encoded.size() * sizeof(float));

    const int prompt_audio_length = total_patches / patch_size;
    std::vector<float> prompt_feat(static_cast<size_t>(prompt_audio_length) * patch_size * feat_dim, 0.0f);
    for (int t = 0; t < prompt_audio_length; ++t) {
        for (int p = 0; p < patch_size; ++p) {
            const int patch_index = t * patch_size + p;
            for (int d = 0; d < feat_dim; ++d) {
                const size_t src = static_cast<size_t>(d) * static_cast<size_t>(total_patches) + static_cast<size_t>(patch_index);
                const size_t dst = (static_cast<size_t>(t) * patch_size + static_cast<size_t>(p)) * feat_dim + static_cast<size_t>(d);
                prompt_feat[dst] = encoded[src];
            }
        }
    }

    return prompt_feat;
}

RealPromptInputs make_real_prompt_inputs(VoxCPMRuntime& runtime,
                                         AudioVAE& audio_vae,
                                         VoxCPMBackend& backend,
                                         const VoxCPMWeightStore& store,
                                         const std::string& prompt_audio_path,
                                         const std::string& prompt_text,
                                         const std::string& text) {
    VoxCPMTokenizer tokenizer;
    REQUIRE(tokenizer.load_from_store(store));
    ChineseCharSplitTokenizer split_tokenizer(tokenizer);

    const int feat_dim = runtime.config().feat_dim;
    const int patch_size = runtime.config().patch_size;
    const std::vector<float> prompt_feat =
        encode_test_audio_features(runtime, audio_vae, backend, prompt_audio_path, TestPaddingMode::Left);
    const int prompt_audio_length =
        static_cast<int>(prompt_feat.size() / static_cast<size_t>(patch_size * feat_dim));

    std::vector<int32_t> text_tokens = split_tokenizer.encode(prompt_text + text, false);
    text_tokens.push_back(101);

    RealPromptInputs out;
    out.full_text_tokens = text_tokens;
    out.full_text_tokens.resize(text_tokens.size() + static_cast<size_t>(prompt_audio_length), 0);
    out.seq_len = static_cast<int>(out.full_text_tokens.size());
    out.feat.assign(static_cast<size_t>(out.seq_len) * patch_size * feat_dim, 0.0f);
    std::copy(prompt_feat.begin(),
              prompt_feat.end(),
              out.feat.begin() + static_cast<std::ptrdiff_t>(text_tokens.size()) * patch_size * feat_dim);
    out.text_mask.assign(text_tokens.size(), 1);
    out.text_mask.resize(static_cast<size_t>(out.seq_len), 0);
    out.feat_mask.assign(text_tokens.size(), 0);
    out.feat_mask.resize(static_cast<size_t>(out.seq_len), 1);
    return out;
}

RealPromptInputs make_real_reference_prompt_inputs(VoxCPMRuntime& runtime,
                                                   AudioVAE& audio_vae,
                                                   VoxCPMBackend& backend,
                                                   const VoxCPMWeightStore& store,
                                                   const std::string& reference_audio_path,
                                                   const std::string& prompt_audio_path,
                                                   const std::string& prompt_text,
                                                   const std::string& text) {
    constexpr int32_t kAudioStartToken = 101;
    constexpr int32_t kRefAudioStartToken = 103;
    constexpr int32_t kRefAudioEndToken = 104;

    VoxCPMTokenizer tokenizer;
    REQUIRE(tokenizer.load_from_store(store));
    ChineseCharSplitTokenizer split_tokenizer(tokenizer);

    const int feat_dim = runtime.config().feat_dim;
    const int patch_size = runtime.config().patch_size;
    const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;
    const std::vector<float> reference_feat =
        encode_test_audio_features(runtime, audio_vae, backend, reference_audio_path, TestPaddingMode::Right);
    const std::vector<float> prompt_feat =
        encode_test_audio_features(runtime, audio_vae, backend, prompt_audio_path, TestPaddingMode::Left);
    const int reference_audio_length =
        static_cast<int>(reference_feat.size() / static_cast<size_t>(patch_size * feat_dim));
    const int prompt_audio_length =
        static_cast<int>(prompt_feat.size() / static_cast<size_t>(patch_size * feat_dim));

    std::vector<int32_t> text_tokens = split_tokenizer.encode(prompt_text + text, false);
    text_tokens.push_back(kAudioStartToken);

    RealPromptInputs out;
    const size_t total_frames = static_cast<size_t>(reference_audio_length) + 2 +
                                static_cast<size_t>(text_tokens.size()) +
                                static_cast<size_t>(prompt_audio_length);
    out.full_text_tokens.reserve(total_frames);
    out.text_mask.reserve(total_frames);
    out.feat_mask.reserve(total_frames);
    out.feat.reserve(total_frames * frame_stride);

    const auto append_zero_frame = [&]() {
        out.feat.insert(out.feat.end(), frame_stride, 0.0f);
    };
    const auto append_feat_frames = [&](const std::vector<float>& frames, int frame_count) {
        out.feat.insert(out.feat.end(),
                        frames.begin(),
                        frames.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(frame_count) * frame_stride));
    };

    out.full_text_tokens.push_back(kRefAudioStartToken);
    out.text_mask.push_back(1);
    out.feat_mask.push_back(0);
    append_zero_frame();

    for (int i = 0; i < reference_audio_length; ++i) {
        out.full_text_tokens.push_back(0);
        out.text_mask.push_back(0);
        out.feat_mask.push_back(1);
    }
    append_feat_frames(reference_feat, reference_audio_length);

    out.full_text_tokens.push_back(kRefAudioEndToken);
    out.text_mask.push_back(1);
    out.feat_mask.push_back(0);
    append_zero_frame();

    for (int32_t token : text_tokens) {
        out.full_text_tokens.push_back(token);
        out.text_mask.push_back(1);
        out.feat_mask.push_back(0);
        append_zero_frame();
    }

    for (int i = 0; i < prompt_audio_length; ++i) {
        out.full_text_tokens.push_back(0);
        out.text_mask.push_back(0);
        out.feat_mask.push_back(1);
    }
    append_feat_frames(prompt_feat, prompt_audio_length);

    out.seq_len = static_cast<int>(out.full_text_tokens.size());
    return out;
}

std::vector<float> make_deterministic_noise_patch(int feat_dim, int patch_size, int step) {
    const size_t size = static_cast<size_t>(feat_dim) * patch_size;
    std::vector<float> noise(size, 0.0f);
    for (size_t i = 0; i < size; ++i) {
        const float phase = 0.173f * static_cast<float>(i + 1) + 0.619f * static_cast<float>(step + 1);
        noise[i] = 0.75f * std::sin(phase);
    }
    return noise;
}

std::vector<float> build_decode_frames_for_test(const std::vector<float>& prompt_feat,
                                                int prompt_audio_length,
                                                const std::vector<float>& generated_steps,
                                                int streaming_prefix_len,
                                                int patch_size,
                                                int feat_dim,
                                                int* prepended_context_frames = nullptr) {
    const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;
    int context_frames = 0;
    if (!prompt_feat.empty() && prompt_audio_length > 0 && streaming_prefix_len > 1) {
        context_frames = std::min(streaming_prefix_len - 1, prompt_audio_length);
    }
    if (prepended_context_frames != nullptr) {
        *prepended_context_frames = context_frames;
    }

    std::vector<float> decode_frames;
    decode_frames.reserve(static_cast<size_t>(context_frames) * frame_stride + generated_steps.size());
    if (context_frames > 0) {
        const size_t context_offset = static_cast<size_t>(prompt_audio_length - context_frames) * frame_stride;
        decode_frames.insert(decode_frames.end(),
                             prompt_feat.begin() + static_cast<std::ptrdiff_t>(context_offset),
                             prompt_feat.end());
    }
    decode_frames.insert(decode_frames.end(), generated_steps.begin(), generated_steps.end());
    return decode_frames;
}

std::vector<float> patch_major_frames_to_latent_for_test(const std::vector<float>& frames,
                                                         int patch_size,
                                                         int feat_dim) {
    const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;
    REQUIRE((frames.size() % frame_stride) == 0);
    const int total_frames = static_cast<int>(frames.size() / frame_stride);
    const int total_patches = total_frames * patch_size;

    std::vector<float> latent(static_cast<size_t>(total_patches) * feat_dim, 0.0f);
    for (int t = 0; t < total_frames; ++t) {
        for (int p = 0; p < patch_size; ++p) {
            const int patch_index = t * patch_size + p;
            for (int d = 0; d < feat_dim; ++d) {
                const size_t src = (static_cast<size_t>(t) * patch_size + static_cast<size_t>(p)) * feat_dim +
                                   static_cast<size_t>(d);
                const size_t dst = static_cast<size_t>(d) * static_cast<size_t>(total_patches) +
                                   static_cast<size_t>(patch_index);
                latent[dst] = frames[src];
            }
        }
    }
    return latent;
}

}  // namespace

TEST_CASE("Persistent state allocates separate state tensors", "[runtime][state]") {
    VoxCPMBackend backend(BackendType::CPU, 2);

    VoxCPMPersistentState state;
    REQUIRE(state.initialize(backend, PersistentStateShape{256, 80, 12}));

    REQUIRE(state.is_initialized());
    REQUIRE(state.context() != nullptr);
    REQUIRE(state.buffer() != nullptr);
    REQUIRE(state.lm_hidden() != nullptr);
    REQUIRE(state.residual_hidden() != nullptr);
    REQUIRE(state.prefix_patch() != nullptr);
    REQUIRE(state.shape().hidden_size == 256);
    REQUIRE(state.shape().feat_dim == 80);
    REQUIRE(state.shape().patch_size == 12);
}

TEST_CASE("Output pool allocates dedicated output tensors", "[runtime][output]") {
    VoxCPMBackend backend(BackendType::CPU, 2);

    VoxCPMOutputPool pool;
    REQUIRE(pool.initialize(backend, OutputPoolShape{80, 12, 64}));

    REQUIRE(pool.is_initialized());
    REQUIRE(pool.context() != nullptr);
    REQUIRE(pool.buffer() != nullptr);
    REQUIRE(pool.patch_output() != nullptr);
    REQUIRE(pool.stop_logits() != nullptr);
    REQUIRE(pool.latent_seq() != nullptr);
    REQUIRE(pool.shape().max_latent_patches == 64);
}

TEST_CASE("Graph cache stores entries by explicit graph key", "[runtime][graph-cache]") {
    VoxCPMGraphCache cache;
    VoxCPMGraphKey key;
    key.stage = "decode.front_half";
    key.seq_len = 1;
    key.n_timesteps = 10;
    key.cfg_milli = 2000;
    key.with_prefix = true;

    REQUIRE_FALSE(cache.contains(key));
    VoxCPMGraphArtifacts& entry = cache.get_or_create(key);
    REQUIRE(cache.contains(key));
    REQUIRE(cache.find(key) == &entry);
    REQUIRE(cache.size() == 1);

    cache.clear();
    REQUIRE(cache.size() == 0);
}

TEST_CASE("Persistent state syncs host buffers through backend storage", "[runtime][state][sync]") {
    VoxCPMBackend backend(BackendType::CPU, 2);

    VoxCPMPersistentState state;
    REQUIRE(state.initialize(backend, PersistentStateShape{4, 3, 2}));

    const std::vector<float> lm_hidden = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> residual_hidden = {5.0f, 6.0f, 7.0f, 8.0f};
    const std::vector<float> prefix_patch = {9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f};

    REQUIRE(state.set_lm_hidden_from_host(backend, lm_hidden.data(), lm_hidden.size()));
    REQUIRE(state.set_residual_hidden_from_host(backend, residual_hidden.data(), residual_hidden.size()));
    REQUIRE(state.set_prefix_patch_from_host(backend, prefix_patch.data(), prefix_patch.size()));

    std::vector<float> lm_hidden_out(lm_hidden.size(), 0.0f);
    std::vector<float> residual_hidden_out(residual_hidden.size(), 0.0f);
    std::vector<float> prefix_patch_out(prefix_patch.size(), 0.0f);

    REQUIRE(state.get_lm_hidden_to_host(backend, lm_hidden_out.data(), lm_hidden_out.size()));
    REQUIRE(state.get_residual_hidden_to_host(backend, residual_hidden_out.data(), residual_hidden_out.size()));
    REQUIRE(state.get_prefix_patch_to_host(backend, prefix_patch_out.data(), prefix_patch_out.size()));

    REQUIRE(lm_hidden_out == lm_hidden);
    REQUIRE(residual_hidden_out == residual_hidden);
    REQUIRE(prefix_patch_out == prefix_patch);
}

TEST_CASE("Output pool publishes host decode outputs", "[runtime][output][sync]") {
    VoxCPMBackend backend(BackendType::CPU, 2);

    VoxCPMOutputPool pool;
    REQUIRE(pool.initialize(backend, OutputPoolShape{3, 2, 8}));

    const std::vector<float> patch = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::array<float, 2> stop = {0.25f, 0.75f};

    REQUIRE(pool.publish_decode_outputs_from_host(backend,
                                                  patch.data(),
                                                  patch.size(),
                                                  stop.data(),
                                                  stop.size()));

    const HostDecodeOutput host = pool.export_decode_output_to_host(backend);
    REQUIRE(host.patch == patch);
    REQUIRE(host.stop_logits == stop);

    const std::vector<float> patch_only = pool.export_patch_to_host(backend);
    const std::array<float, 2> stop_only = pool.export_stop_logits_to_host(backend);
    REQUIRE(patch_only == patch);
    REQUIRE(stop_only == stop);
}

TEST_CASE("Output pool supports split patch and stop publication", "[runtime][output][split-sync]") {
    VoxCPMBackend backend(BackendType::CPU, 2);

    VoxCPMOutputPool pool;
    REQUIRE(pool.initialize(backend, OutputPoolShape{3, 2, 8}));

    const std::vector<float> patch = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    const std::array<float, 2> stop = {0.9f, 0.1f};

    REQUIRE(pool.publish_patch_output_from_host(backend, patch.data(), patch.size()));
    REQUIRE(pool.publish_stop_logits_from_host(backend, stop.data(), stop.size()));

    REQUIRE(pool.export_patch_to_host(backend) == patch);
    REQUIRE(pool.export_stop_logits_to_host(backend) == stop);
}

TEST_CASE("Output pool tracks latent sequence patches", "[runtime][output][latent-seq]") {
    VoxCPMBackend backend(BackendType::CPU, 2);

    VoxCPMOutputPool pool;
    REQUIRE(pool.initialize(backend, OutputPoolShape{3, 2, 8}));

    const std::vector<float> patch0 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> patch1 = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    REQUIRE(pool.write_patch_to_latent_seq_from_host(backend, patch0.data(), patch0.size(), 0));
    REQUIRE(pool.write_patch_to_latent_seq_from_host(backend, patch1.data(), patch1.size(), 1));

    const std::vector<float> latent = pool.export_latent_seq_to_host(backend, 2);
    REQUIRE(latent.size() == patch0.size() + patch1.size());
    REQUIRE(std::equal(patch0.begin(), patch0.end(), latent.begin()));
    REQUIRE(std::equal(patch1.begin(), patch1.end(), latent.begin() + static_cast<std::ptrdiff_t>(patch0.size())));

    const std::vector<float> audio_vae_latent = pool.export_audio_vae_latent_to_host(backend, 0, 2);
    const std::vector<float> expected_audio_vae = {
        1.0f, 4.0f, 6.0f, 3.0f,
        2.0f, 5.0f, 5.0f, 2.0f,
        3.0f, 6.0f, 4.0f, 1.0f,
    };
    REQUIRE(audio_vae_latent == expected_audio_vae);
}

TEST_CASE("Output pool can export latent sequence subranges without fetching the full prefix",
          "[runtime][output][latent-seq][range][transfer]") {
    VoxCPMBackend backend(BackendType::CPU, 2);

    VoxCPMOutputPool pool;
    REQUIRE(pool.initialize(backend, OutputPoolShape{3, 2, 8}));

    const std::vector<float> patch0 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> patch1 = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    const std::vector<float> patch2 = {7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f};
    REQUIRE(pool.write_patch_to_latent_seq_from_host(backend, patch0.data(), patch0.size(), 0));
    REQUIRE(pool.write_patch_to_latent_seq_from_host(backend, patch1.data(), patch1.size(), 1));
    REQUIRE(pool.write_patch_to_latent_seq_from_host(backend, patch2.data(), patch2.size(), 2));

    backend.reset_transfer_stats();
    const std::vector<float> patch_range = pool.export_latent_seq_range_to_host(backend, 1, 2);
    const BackendTransferStats range_stats = backend.transfer_stats();

    REQUIRE(patch_range.size() == patch1.size() + patch2.size());
    REQUIRE(std::equal(patch1.begin(), patch1.end(), patch_range.begin()));
    REQUIRE(std::equal(patch2.begin(),
                       patch2.end(),
                       patch_range.begin() + static_cast<std::ptrdiff_t>(patch1.size())));
    REQUIRE(range_stats.device_to_host_bytes == patch_range.size() * sizeof(float));

    backend.reset_transfer_stats();
    const std::vector<float> audio_vae_latent = pool.export_audio_vae_latent_to_host(backend, 1, 2);
    const BackendTransferStats audio_vae_stats = backend.transfer_stats();
    const std::vector<float> expected_audio_vae = {
        6.0f, 3.0f, 7.0f, 1.0f,
        5.0f, 2.0f, 8.0f, 2.0f,
        4.0f, 1.0f, 9.0f, 3.0f,
    };
    REQUIRE(audio_vae_latent == expected_audio_vae);
    REQUIRE(audio_vae_stats.device_to_host_bytes == patch_range.size() * sizeof(float));
}

TEST_CASE("Output pool can publish contiguous patch ranges into latent sequence",
          "[runtime][output][latent-seq][span]") {
    VoxCPMBackend backend(BackendType::CPU, 2);

    VoxCPMOutputPool pool;
    REQUIRE(pool.initialize(backend, OutputPoolShape{3, 2, 8}));

    const std::vector<float> patches = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    };
    REQUIRE(pool.write_patch_range_to_latent_seq_from_host(backend, patches.data(), 1, 2));

    const std::vector<float> latent = pool.export_latent_seq_to_host(backend, 3);
    REQUIRE(latent.size() == patches.size() * 3 / 2);
    REQUIRE(std::equal(patches.begin(),
                       patches.end(),
                       latent.begin() + static_cast<std::ptrdiff_t>(patches.size() / 2)));
}

TEST_CASE("Output pool can publish patch tensors into latent sequence without host round-trip",
          "[runtime][output][latent-seq][d2d]") {
    VoxCPMBackend backend(BackendType::CPU, 2);

    VoxCPMOutputPool pool;
    REQUIRE(pool.initialize(backend, OutputPoolShape{3, 2, 8}));

    const std::vector<float> patch = {1.5f, -2.0f, 3.25f, 4.5f, -5.0f, 6.75f};
    REQUIRE(pool.publish_patch_output_from_host(backend, patch.data(), patch.size()));
    REQUIRE(pool.publish_patch_to_latent_seq(backend, pool.patch_output(), 2));

    const std::vector<float> latent = pool.export_latent_seq_to_host(backend, 3);
    REQUIRE(latent.size() == patch.size() * 3);
    REQUIRE(std::equal(patch.begin(),
                       patch.end(),
                       latent.begin() + static_cast<std::ptrdiff_t>(patch.size() * 2)));
}

TEST_CASE("Output pool can publish backend-resident patch ranges into latent sequence",
          "[runtime][output][latent-seq][span][d2d]") {
    VoxCPMBackend backend(BackendType::CPU, 2);

    VoxCPMOutputPool pool;
    REQUIRE(pool.initialize(backend, OutputPoolShape{3, 2, 8}));

    VoxCPMContext src_ctx(ContextType::Graph, 8, 8);
    ggml_tensor* src = src_ctx.new_tensor_2d(GGML_TYPE_F32, 3, 4);
    REQUIRE(src != nullptr);
    ggml_backend_buffer_t src_buffer = backend.alloc_buffer(src_ctx.raw_context(), BufferUsage::Compute);
    REQUIRE(src_buffer != nullptr);

    const std::vector<float> patches = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    };
    backend.tensor_set(src, patches.data(), 0, patches.size() * sizeof(float));

    backend.reset_transfer_stats();
    REQUIRE(pool.publish_patch_range_to_latent_seq(backend, src, 1, 2));
    const BackendTransferStats stats = backend.transfer_stats();

    const std::vector<float> latent = pool.export_latent_seq_to_host(backend, 3);
    REQUIRE(latent.size() == patches.size() * 3 / 2);
    REQUIRE(std::equal(patches.begin(),
                       patches.end(),
                       latent.begin() + static_cast<std::ptrdiff_t>(patches.size() / 2)));
    REQUIRE(stats.host_to_device_bytes == 0);
    REQUIRE(stats.device_to_device_bytes >= patches.size() * sizeof(float));

    backend.free_buffer(src_buffer);
}

TEST_CASE("Output pool exposes AudioVAE latent view with the documented layout",
          "[runtime][output][latent-seq][audio-vae-view]") {
    VoxCPMBackend backend(BackendType::CPU, 2);

    VoxCPMOutputPool pool;
    REQUIRE(pool.initialize(backend, OutputPoolShape{3, 2, 8}));

    const std::vector<float> patch0 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> patch1 = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    REQUIRE(pool.write_patch_to_latent_seq_from_host(backend, patch0.data(), patch0.size(), 0));
    REQUIRE(pool.write_patch_to_latent_seq_from_host(backend, patch1.data(), patch1.size(), 1));

    VoxCPMContext graph_ctx(ContextType::Graph, 1024, 8192);
    ggml_tensor* latent = pool.make_audio_vae_latent_view(graph_ctx.raw_context(), 0, 2);
    REQUIRE(latent != nullptr);
    ggml_set_output(latent);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, latent);
    backend.reserve_compute_memory(graph, "test.output_pool.audio_vae_latent_view");
    backend.alloc_graph(graph, "test.output_pool.audio_vae_latent_view");
    REQUIRE(backend.compute(graph) == GGML_STATUS_SUCCESS);

    std::vector<float> latent_from_view(static_cast<size_t>(ggml_nelements(latent)), 0.0f);
    backend.tensor_get(latent, latent_from_view.data(), 0, latent_from_view.size() * sizeof(float));

    const std::vector<float> latent_from_host = pool.export_audio_vae_latent_to_host(backend, 0, 2);
    REQUIRE(latent_from_view == latent_from_host);
}

TEST_CASE("prefill prompt timeline benchmark can consume backend-resident prompt patch ranges",
          "[runtime][prefill][prompt-timeline][tensor]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    const int frame_count = 3;
    const size_t patch_elem_count = static_cast<size_t>(runtime.config().patch_size * runtime.config().feat_dim);
    std::vector<float> prompt_patches(frame_count * patch_elem_count, 0.0f);
    for (size_t i = 0; i < prompt_patches.size(); ++i) {
        prompt_patches[i] = static_cast<float>((static_cast<int>(i % 29) - 14)) * 0.01f;
    }

    VoxCPMContext src_ctx(ContextType::Graph, 8, 8);
    ggml_tensor* prompt_patch_tensor =
        src_ctx.new_tensor_2d(GGML_TYPE_F32, runtime.config().feat_dim, frame_count * runtime.config().patch_size);
    REQUIRE(prompt_patch_tensor != nullptr);
    ggml_backend_buffer_t src_buffer = backend.alloc_buffer(src_ctx.raw_context(), BufferUsage::Compute);
    REQUIRE(src_buffer != nullptr);
    backend.tensor_set(prompt_patch_tensor, prompt_patches.data(), 0, prompt_patches.size() * sizeof(float));

    backend.reset_transfer_stats();
    VoxCPMDecodeState host_state = runtime.benchmark_stage_prefill_prompt_timeline(prompt_patches, frame_count, true);
    const BackendTransferStats host_stats = backend.transfer_stats();

    backend.reset_transfer_stats();
    VoxCPMDecodeState tensor_state =
        runtime.benchmark_stage_prefill_prompt_timeline_from_tensor(prompt_patch_tensor, frame_count, true);
    const BackendTransferStats tensor_stats = backend.transfer_stats();

    REQUIRE(tensor_state.audio_frame_count == host_state.audio_frame_count);
    REQUIRE(tensor_state.output_pool != nullptr);
    REQUIRE(host_state.output_pool != nullptr);
    REQUIRE(tensor_state.output_pool->export_latent_seq_to_host(backend, frame_count) ==
            host_state.output_pool->export_latent_seq_to_host(backend, frame_count));
    REQUIRE(tensor_state.output_pool->export_patch_to_host(backend) ==
            host_state.output_pool->export_patch_to_host(backend));

    std::vector<float> host_prefix(patch_elem_count, 0.0f);
    std::vector<float> tensor_prefix(patch_elem_count, 0.0f);
    REQUIRE(host_state.persistent_state != nullptr);
    REQUIRE(tensor_state.persistent_state != nullptr);
    REQUIRE(host_state.persistent_state->get_prefix_patch_to_host(backend, host_prefix.data(), host_prefix.size()));
    REQUIRE(tensor_state.persistent_state->get_prefix_patch_to_host(backend, tensor_prefix.data(), tensor_prefix.size()));
    REQUIRE(tensor_prefix == host_prefix);
    REQUIRE(tensor_state.prefix_feat_cond == host_state.prefix_feat_cond);

    REQUIRE(tensor_stats.host_to_device_bytes < host_stats.host_to_device_bytes);
    REQUIRE(host_stats.host_to_device_bytes - tensor_stats.host_to_device_bytes >=
            prompt_patches.size() * sizeof(float));
    REQUIRE(tensor_stats.device_to_device_bytes >= prompt_patches.size() * sizeof(float));

    backend.free_buffer(src_buffer);
}

TEST_CASE("stage-two module benchmark helpers preserve expected shapes and finite outputs",
          "[runtime][modules][stage2]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    const int seq_len = 2;
    const int hidden_size = runtime.base_lm().config().hidden_size;
    const int feat_hidden_size = runtime.feat_encoder().config().hidden_size;
    const int patch_size = runtime.config().patch_size;
    const int feat_dim = runtime.config().feat_dim;

    const std::vector<int32_t> tokens = {1, 2, 3};
    const std::vector<float> embedding = runtime.benchmark_run_embedding(tokens);
    REQUIRE(embedding.size() == static_cast<size_t>(tokens.size() * hidden_size));
    REQUIRE(std::all_of(embedding.begin(), embedding.end(), [](float value) {
        return std::isfinite(value);
    }));

    std::vector<float> feat(static_cast<size_t>(seq_len * patch_size * feat_dim), 0.0f);
    for (size_t i = 0; i < feat.size(); ++i) {
        feat[i] = static_cast<float>((static_cast<int>(i % 13) - 6)) * 0.0625f;
    }

    const std::vector<float> encoded = runtime.benchmark_encode_feature_sequence(feat, seq_len);
    REQUIRE(encoded.size() == static_cast<size_t>(seq_len * feat_hidden_size));
    REQUIRE(std::all_of(encoded.begin(), encoded.end(), [](float value) {
        return std::isfinite(value);
    }));

    const std::vector<float> projected = runtime.benchmark_run_enc_to_lm_projection(encoded, seq_len);
    REQUIRE(projected.size() == static_cast<size_t>(seq_len * hidden_size));
    REQUIRE(std::all_of(projected.begin(), projected.end(), [](float value) {
        return std::isfinite(value);
    }));

    const std::vector<float> fsq = runtime.benchmark_run_fsq_2d(projected, seq_len);
    REQUIRE(fsq.size() == projected.size());
    REQUIRE(std::all_of(fsq.begin(), fsq.end(), [](float value) {
        return std::isfinite(value);
    }));

    const std::vector<float> last_hidden = slice_last_column_major_2d(projected, hidden_size, seq_len - 1);
    const std::vector<float> stop = runtime.benchmark_run_stop_predictor(last_hidden);
    REQUIRE(stop.size() == 2);
    REQUIRE(std::all_of(stop.begin(), stop.end(), [](float value) {
        return std::isfinite(value);
    }));

    const std::vector<float> patch(feat.begin(), feat.begin() + static_cast<std::ptrdiff_t>(patch_size * feat_dim));
    const std::vector<float> locenc = runtime.benchmark_run_locenc_patch_to_lm_embed(patch);
    REQUIRE(locenc.size() == static_cast<size_t>(hidden_size));
    REQUIRE(std::all_of(locenc.begin(), locenc.end(), [](float value) {
        return std::isfinite(value);
    }));
}

TEST_CASE("embedding benchmark can consume backend-resident token tensors directly with lower h2d",
          "[runtime][modules][stage2][embedding]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    const std::vector<int32_t> tokens = {1, 2, 3, 4};
    VoxCPMContext token_ctx(ContextType::Graph, 8, 64);
    ggml_tensor* token_tensor = token_ctx.new_tensor_1d(GGML_TYPE_I32, static_cast<int64_t>(tokens.size()));
    REQUIRE(token_tensor != nullptr);
    ggml_backend_buffer_t token_buffer = backend.alloc_buffer(token_ctx.raw_context(), BufferUsage::State);
    REQUIRE(token_buffer != nullptr);
    backend.tensor_set(token_tensor, tokens.data(), 0, tokens.size() * sizeof(int32_t));

    backend.reset_transfer_stats();
    const std::vector<float> host_embedding = runtime.benchmark_run_embedding(tokens);
    const BackendTransferStats host_stats = backend.transfer_stats();

    backend.reset_transfer_stats();
    const std::vector<float> tensor_embedding = runtime.benchmark_run_embedding_from_tensor(token_tensor,
                                                                                            static_cast<int>(tokens.size()));
    const BackendTransferStats tensor_stats = backend.transfer_stats();

    REQUIRE(host_embedding.size() == tensor_embedding.size());
    for (size_t i = 0; i < host_embedding.size(); ++i) {
        REQUIRE(host_embedding[i] == Catch::Approx(tensor_embedding[i]));
    }
    REQUIRE(tensor_stats.host_to_device_bytes < host_stats.host_to_device_bytes);
    REQUIRE(host_stats.host_to_device_bytes - tensor_stats.host_to_device_bytes >=
            tokens.size() * sizeof(int32_t));

    backend.free_buffer(token_buffer);
}

TEST_CASE("enc_to_lm projection benchmark can consume backend-resident feature tensors directly with lower h2d",
          "[runtime][modules][stage2][projection]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    const int seq_len = 2;
    const int feat_hidden_size = runtime.feat_encoder().config().hidden_size;
    std::vector<float> encoded(static_cast<size_t>(feat_hidden_size * seq_len), 0.0f);
    for (size_t i = 0; i < encoded.size(); ++i) {
        encoded[i] = static_cast<float>((static_cast<int>(i % 23) - 11)) * 0.03125f;
    }

    VoxCPMContext encoded_ctx(ContextType::Graph, 8, 64);
    ggml_tensor* encoded_tensor = encoded_ctx.new_tensor_2d(GGML_TYPE_F32, feat_hidden_size, seq_len);
    REQUIRE(encoded_tensor != nullptr);
    ggml_backend_buffer_t encoded_buffer = backend.alloc_buffer(encoded_ctx.raw_context(), BufferUsage::State);
    REQUIRE(encoded_buffer != nullptr);
    backend.tensor_set(encoded_tensor, encoded.data(), 0, encoded.size() * sizeof(float));

    backend.reset_transfer_stats();
    const std::vector<float> host_projection = runtime.benchmark_run_enc_to_lm_projection(encoded, seq_len);
    const BackendTransferStats host_stats = backend.transfer_stats();

    backend.reset_transfer_stats();
    const std::vector<float> tensor_projection =
        runtime.benchmark_run_enc_to_lm_projection_from_tensor(encoded_tensor, seq_len);
    const BackendTransferStats tensor_stats = backend.transfer_stats();

    REQUIRE(host_projection.size() == tensor_projection.size());
    for (size_t i = 0; i < host_projection.size(); ++i) {
        REQUIRE(host_projection[i] == Catch::Approx(tensor_projection[i]));
    }
    REQUIRE(tensor_stats.host_to_device_bytes < host_stats.host_to_device_bytes);
    REQUIRE(host_stats.host_to_device_bytes - tensor_stats.host_to_device_bytes >=
            encoded.size() * sizeof(float));

    backend.free_buffer(encoded_buffer);
}

TEST_CASE("enc_to_lm projection plus fsq benchmark can compose backend-resident feature tensors with lower h2d",
          "[runtime][modules][stage2][projection][fsq]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    const int seq_len = 2;
    const int feat_hidden_size = runtime.feat_encoder().config().hidden_size;
    const int lm_hidden_size = runtime.base_lm().config().hidden_size;
    std::vector<float> encoded(static_cast<size_t>(feat_hidden_size * seq_len), 0.0f);
    for (size_t i = 0; i < encoded.size(); ++i) {
        encoded[i] = static_cast<float>((static_cast<int>(i % 29) - 14)) * 0.0234375f;
    }

    VoxCPMContext encoded_ctx(ContextType::Graph, 8, 64);
    ggml_tensor* encoded_tensor = encoded_ctx.new_tensor_2d(GGML_TYPE_F32, feat_hidden_size, seq_len);
    REQUIRE(encoded_tensor != nullptr);
    ggml_backend_buffer_t encoded_buffer = backend.alloc_buffer(encoded_ctx.raw_context(), BufferUsage::State);
    REQUIRE(encoded_buffer != nullptr);
    backend.tensor_set(encoded_tensor, encoded.data(), 0, encoded.size() * sizeof(float));

    backend.reset_transfer_stats();
    const std::vector<float> host_fused = runtime.benchmark_run_enc_to_lm_projection_fsq(encoded, seq_len);
    const BackendTransferStats host_stats = backend.transfer_stats();

    backend.reset_transfer_stats();
    const std::vector<float> tensor_fused =
        runtime.benchmark_run_enc_to_lm_projection_fsq_from_tensor(encoded_tensor, seq_len);
    const BackendTransferStats tensor_stats = backend.transfer_stats();

    REQUIRE(host_fused.size() == static_cast<size_t>(lm_hidden_size * seq_len));
    REQUIRE(host_fused.size() == tensor_fused.size());
    for (size_t i = 0; i < host_fused.size(); ++i) {
        REQUIRE(host_fused[i] == Catch::Approx(tensor_fused[i]));
    }
    REQUIRE(tensor_stats.host_to_device_bytes < host_stats.host_to_device_bytes);
    REQUIRE(host_stats.host_to_device_bytes - tensor_stats.host_to_device_bytes >=
            (encoded.size() + host_fused.size()) * sizeof(float));

    backend.free_buffer(encoded_buffer);
}

TEST_CASE("locenc sequence plus enc_to_lm projection benchmark can compose backend-resident feature tensors with lower h2d",
          "[runtime][modules][stage2][locenc][projection]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    const int seq_len = 2;
    const int hidden_size = runtime.base_lm().config().hidden_size;
    const int patch_size = runtime.config().patch_size;
    const int feat_dim = runtime.config().feat_dim;
    std::vector<float> feat(static_cast<size_t>(seq_len * patch_size * feat_dim), 0.0f);
    for (size_t i = 0; i < feat.size(); ++i) {
        feat[i] = static_cast<float>((static_cast<int>(i % 31) - 15)) * 0.015625f;
    }

    VoxCPMContext feat_ctx(ContextType::Graph, 8, 64);
    ggml_tensor* feat_tensor = feat_ctx.new_tensor_3d(GGML_TYPE_F32, feat_dim, patch_size, seq_len);
    REQUIRE(feat_tensor != nullptr);
    ggml_backend_buffer_t feat_buffer = backend.alloc_buffer(feat_ctx.raw_context(), BufferUsage::State);
    REQUIRE(feat_buffer != nullptr);
    backend.tensor_set(feat_tensor, feat.data(), 0, feat.size() * sizeof(float));

    backend.reset_transfer_stats();
    const std::vector<float> host_fused = runtime.benchmark_run_locenc_sequence_to_lm_projection(feat, seq_len);
    const BackendTransferStats host_stats = backend.transfer_stats();

    backend.reset_transfer_stats();
    const std::vector<float> tensor_fused =
        runtime.benchmark_run_locenc_sequence_to_lm_projection_from_tensor(feat_tensor, seq_len);
    const BackendTransferStats tensor_stats = backend.transfer_stats();

    REQUIRE(host_fused.size() == static_cast<size_t>(hidden_size * seq_len));
    REQUIRE(host_fused.size() == tensor_fused.size());
    for (size_t i = 0; i < host_fused.size(); ++i) {
        REQUIRE(host_fused[i] == Catch::Approx(tensor_fused[i]));
    }
    REQUIRE(tensor_stats.host_to_device_bytes < host_stats.host_to_device_bytes);
    REQUIRE(host_stats.host_to_device_bytes - tensor_stats.host_to_device_bytes >=
            (feat.size() + host_fused.size()) * sizeof(float));

    backend.free_buffer(feat_buffer);
}

TEST_CASE("locenc sequence plus enc_to_lm projection plus fsq benchmark can compose backend-resident feature tensors with lower h2d",
          "[runtime][modules][stage2][locenc][projection][fsq]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    const int seq_len = 2;
    const int hidden_size = runtime.base_lm().config().hidden_size;
    const int patch_size = runtime.config().patch_size;
    const int feat_dim = runtime.config().feat_dim;
    std::vector<float> feat(static_cast<size_t>(seq_len * patch_size * feat_dim), 0.0f);
    for (size_t i = 0; i < feat.size(); ++i) {
        feat[i] = static_cast<float>((static_cast<int>(i % 37) - 18)) * 0.01171875f;
    }

    VoxCPMContext feat_ctx(ContextType::Graph, 8, 64);
    ggml_tensor* feat_tensor = feat_ctx.new_tensor_3d(GGML_TYPE_F32, feat_dim, patch_size, seq_len);
    REQUIRE(feat_tensor != nullptr);
    ggml_backend_buffer_t feat_buffer = backend.alloc_buffer(feat_ctx.raw_context(), BufferUsage::State);
    REQUIRE(feat_buffer != nullptr);
    backend.tensor_set(feat_tensor, feat.data(), 0, feat.size() * sizeof(float));

    backend.reset_transfer_stats();
    const std::vector<float> host_fused =
        runtime.benchmark_run_locenc_sequence_to_lm_projection_fsq(feat, seq_len);
    const BackendTransferStats host_stats = backend.transfer_stats();

    backend.reset_transfer_stats();
    const std::vector<float> tensor_fused =
        runtime.benchmark_run_locenc_sequence_to_lm_projection_fsq_from_tensor(feat_tensor, seq_len);
    const BackendTransferStats tensor_stats = backend.transfer_stats();

    REQUIRE(host_fused.size() == static_cast<size_t>(hidden_size * seq_len));
    REQUIRE(host_fused.size() == tensor_fused.size());
    for (size_t i = 0; i < host_fused.size(); ++i) {
        REQUIRE(host_fused[i] == Catch::Approx(tensor_fused[i]));
    }
    REQUIRE(tensor_stats.host_to_device_bytes < host_stats.host_to_device_bytes);
    REQUIRE(host_stats.host_to_device_bytes - tensor_stats.host_to_device_bytes >=
            (feat.size() + host_fused.size()) * sizeof(float));

    backend.free_buffer(feat_buffer);
}

TEST_CASE("embedding plus mask plus locenc sequence to lm projection benchmark can compose backend-resident inputs with lower h2d",
          "[runtime][modules][stage2][embedding][locenc][projection]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    const int seq_len = 4;
    const int hidden_size = runtime.base_lm().config().hidden_size;
    const int patch_size = runtime.config().patch_size;
    const int feat_dim = runtime.config().feat_dim;
    const std::vector<int32_t> tokens = {11, 12, 13, 14};
    std::vector<float> feat(static_cast<size_t>(seq_len * patch_size * feat_dim), 0.0f);
    for (size_t i = 0; i < feat.size(); ++i) {
        feat[i] = static_cast<float>((static_cast<int>(i % 41) - 20)) * 0.0078125f;
    }
    const std::vector<float> text_mask = {1.0f, 1.0f, 0.0f, 0.0f};
    const std::vector<float> feat_mask = {0.0f, 0.0f, 1.0f, 1.0f};

    VoxCPMContext token_ctx(ContextType::Graph, 8, 64);
    ggml_tensor* token_tensor = token_ctx.new_tensor_1d(GGML_TYPE_I32, seq_len);
    REQUIRE(token_tensor != nullptr);
    ggml_backend_buffer_t token_buffer = backend.alloc_buffer(token_ctx.raw_context(), BufferUsage::State);
    REQUIRE(token_buffer != nullptr);
    backend.tensor_set(token_tensor, tokens.data(), 0, tokens.size() * sizeof(int32_t));

    VoxCPMContext feat_ctx(ContextType::Graph, 8, 64);
    ggml_tensor* feat_tensor = feat_ctx.new_tensor_3d(GGML_TYPE_F32, feat_dim, patch_size, seq_len);
    REQUIRE(feat_tensor != nullptr);
    ggml_backend_buffer_t feat_buffer = backend.alloc_buffer(feat_ctx.raw_context(), BufferUsage::State);
    REQUIRE(feat_buffer != nullptr);
    backend.tensor_set(feat_tensor, feat.data(), 0, feat.size() * sizeof(float));

    VoxCPMContext mask_ctx(ContextType::Graph, 8, 64);
    ggml_tensor* text_mask_tensor = mask_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    ggml_tensor* feat_mask_tensor = mask_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    REQUIRE(text_mask_tensor != nullptr);
    REQUIRE(feat_mask_tensor != nullptr);
    ggml_backend_buffer_t mask_buffer = backend.alloc_buffer(mask_ctx.raw_context(), BufferUsage::State);
    REQUIRE(mask_buffer != nullptr);
    backend.tensor_set(text_mask_tensor, text_mask.data(), 0, text_mask.size() * sizeof(float));
    backend.tensor_set(feat_mask_tensor, feat_mask.data(), 0, feat_mask.size() * sizeof(float));

    backend.reset_transfer_stats();
    const std::vector<float> host_fused =
        runtime.benchmark_run_embedding_masked_locenc_sequence_to_lm_projection(tokens,
                                                                                feat,
                                                                                text_mask,
                                                                                feat_mask,
                                                                                seq_len);
    const BackendTransferStats host_stats = backend.transfer_stats();

    backend.reset_transfer_stats();
    const std::vector<float> tensor_fused =
        runtime.benchmark_run_embedding_masked_locenc_sequence_to_lm_projection_from_tensors(token_tensor,
                                                                                              feat_tensor,
                                                                                              text_mask_tensor,
                                                                                              feat_mask_tensor,
                                                                                              seq_len);
    const BackendTransferStats tensor_stats = backend.transfer_stats();

    REQUIRE(host_fused.size() == static_cast<size_t>(hidden_size * seq_len));
    REQUIRE(host_fused.size() == tensor_fused.size());
    for (size_t i = 0; i < host_fused.size(); ++i) {
        REQUIRE(host_fused[i] == Catch::Approx(tensor_fused[i]));
    }
    REQUIRE(tensor_stats.host_to_device_bytes < host_stats.host_to_device_bytes);
    REQUIRE(host_stats.host_to_device_bytes - tensor_stats.host_to_device_bytes >=
            tokens.size() * sizeof(int32_t) + (feat.size() + text_mask.size() + feat_mask.size()) * sizeof(float));

    backend.free_buffer(token_buffer);
    backend.free_buffer(feat_buffer);
    backend.free_buffer(mask_buffer);
}

TEST_CASE("embedding plus mask plus locenc sequence to lm projection plus fsq benchmark can compose backend-resident inputs with lower h2d",
          "[runtime][modules][stage2][embedding][locenc][projection][fsq]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    const int seq_len = 4;
    const int hidden_size = runtime.base_lm().config().hidden_size;
    const int patch_size = runtime.config().patch_size;
    const int feat_dim = runtime.config().feat_dim;
    const std::vector<int32_t> tokens = {21, 22, 23, 24};
    std::vector<float> feat(static_cast<size_t>(seq_len * patch_size * feat_dim), 0.0f);
    for (size_t i = 0; i < feat.size(); ++i) {
        feat[i] = static_cast<float>((static_cast<int>(i % 43) - 21)) * 0.00625f;
    }
    const std::vector<float> text_mask = {1.0f, 1.0f, 0.0f, 0.0f};
    const std::vector<float> feat_mask = {0.0f, 0.0f, 1.0f, 1.0f};

    VoxCPMContext token_ctx(ContextType::Graph, 8, 64);
    ggml_tensor* token_tensor = token_ctx.new_tensor_1d(GGML_TYPE_I32, seq_len);
    REQUIRE(token_tensor != nullptr);
    ggml_backend_buffer_t token_buffer = backend.alloc_buffer(token_ctx.raw_context(), BufferUsage::State);
    REQUIRE(token_buffer != nullptr);
    backend.tensor_set(token_tensor, tokens.data(), 0, tokens.size() * sizeof(int32_t));

    VoxCPMContext feat_ctx(ContextType::Graph, 8, 64);
    ggml_tensor* feat_tensor = feat_ctx.new_tensor_3d(GGML_TYPE_F32, feat_dim, patch_size, seq_len);
    REQUIRE(feat_tensor != nullptr);
    ggml_backend_buffer_t feat_buffer = backend.alloc_buffer(feat_ctx.raw_context(), BufferUsage::State);
    REQUIRE(feat_buffer != nullptr);
    backend.tensor_set(feat_tensor, feat.data(), 0, feat.size() * sizeof(float));

    VoxCPMContext mask_ctx(ContextType::Graph, 8, 64);
    ggml_tensor* text_mask_tensor = mask_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    ggml_tensor* feat_mask_tensor = mask_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    REQUIRE(text_mask_tensor != nullptr);
    REQUIRE(feat_mask_tensor != nullptr);
    ggml_backend_buffer_t mask_buffer = backend.alloc_buffer(mask_ctx.raw_context(), BufferUsage::State);
    REQUIRE(mask_buffer != nullptr);
    backend.tensor_set(text_mask_tensor, text_mask.data(), 0, text_mask.size() * sizeof(float));
    backend.tensor_set(feat_mask_tensor, feat_mask.data(), 0, feat_mask.size() * sizeof(float));

    backend.reset_transfer_stats();
    const std::vector<float> host_fused =
        runtime.benchmark_run_embedding_masked_locenc_sequence_to_lm_projection_fsq(tokens,
                                                                                    feat,
                                                                                    text_mask,
                                                                                    feat_mask,
                                                                                    seq_len);
    const BackendTransferStats host_stats = backend.transfer_stats();

    backend.reset_transfer_stats();
    const std::vector<float> tensor_fused =
        runtime.benchmark_run_embedding_masked_locenc_sequence_to_lm_projection_fsq_from_tensors(token_tensor,
                                                                                                  feat_tensor,
                                                                                                  text_mask_tensor,
                                                                                                  feat_mask_tensor,
                                                                                                  seq_len);
    const BackendTransferStats tensor_stats = backend.transfer_stats();

    REQUIRE(host_fused.size() == static_cast<size_t>(hidden_size * seq_len));
    REQUIRE(host_fused.size() == tensor_fused.size());
    for (size_t i = 0; i < host_fused.size(); ++i) {
        REQUIRE(host_fused[i] == Catch::Approx(tensor_fused[i]));
    }
    REQUIRE(tensor_stats.host_to_device_bytes < host_stats.host_to_device_bytes);
    REQUIRE(host_stats.host_to_device_bytes - tensor_stats.host_to_device_bytes >=
            tokens.size() * sizeof(int32_t) + (feat.size() + text_mask.size() + feat_mask.size() + host_fused.size()) * sizeof(float));

    backend.free_buffer(token_buffer);
    backend.free_buffer(feat_buffer);
    backend.free_buffer(mask_buffer);
}

TEST_CASE("prefill front-half fused replacement matches legacy assembly while reducing d2h staging",
          "[runtime][prefill][front-half][transfer][regression]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    const int seq_len = 4;
    const int hidden_size = runtime.base_lm().config().hidden_size;
    const std::vector<int32_t> tokens = {31, 32, 33, 34};
    std::vector<float> feat(static_cast<size_t>(seq_len * runtime.config().patch_size * runtime.config().feat_dim), 0.0f);
    for (size_t i = 0; i < feat.size(); ++i) {
        feat[i] = static_cast<float>((static_cast<int>(i % 53) - 26)) * 0.004f;
    }
    const std::vector<float> text_mask = {1.0f, 1.0f, 0.0f, 0.0f};
    const std::vector<float> feat_mask = {0.0f, 0.0f, 1.0f, 1.0f};

    auto blend_with_masks = [&](const std::vector<float>& text_part,
                                const std::vector<float>& feat_part) {
        REQUIRE(text_part.size() == feat_part.size());
        std::vector<float> combined(text_part.size(), 0.0f);
        for (int t = 0; t < seq_len; ++t) {
            const float text_scale = text_mask[static_cast<size_t>(t)];
            const float feat_scale = feat_mask[static_cast<size_t>(t)];
            for (int h = 0; h < hidden_size; ++h) {
                const size_t idx = static_cast<size_t>(t) * static_cast<size_t>(hidden_size) + static_cast<size_t>(h);
                combined[idx] = text_scale * text_part[idx] + feat_scale * feat_part[idx];
            }
        }
        return combined;
    };

    MiniCPMKVCache legacy_cache(runtime.base_lm().config().n_layer,
                                runtime.base_lm().config().n_kv_heads,
                                runtime.config().max_length,
                                runtime.base_lm().config().head_dim());
    MiniCPMKVCache fused_cache(runtime.base_lm().config().n_layer,
                               runtime.base_lm().config().n_kv_heads,
                               runtime.config().max_length,
                               runtime.base_lm().config().head_dim());
    legacy_cache.init(backend);
    fused_cache.init(backend);

    backend.reset_transfer_stats();
    const std::vector<float> text_embed = runtime.benchmark_run_embedding(tokens);
    const std::vector<float> feat_embed = runtime.benchmark_run_locenc_sequence_to_lm_projection(feat, seq_len);
    const std::vector<float> legacy_combined = blend_with_masks(text_embed, feat_embed);
    const std::vector<float> legacy_base = runtime.benchmark_run_base_lm_forward(legacy_combined, seq_len, legacy_cache, true);
    const std::vector<float> legacy_fsq = runtime.benchmark_run_fsq_2d(legacy_base, seq_len);
    const std::vector<float> legacy_blended = blend_with_masks(legacy_base, legacy_fsq);
    const BackendTransferStats legacy_stats = backend.transfer_stats();

    backend.reset_transfer_stats();
    const std::vector<float> fused_combined =
        runtime.benchmark_run_embedding_masked_locenc_sequence_to_lm_projection(tokens,
                                                                                feat,
                                                                                text_mask,
                                                                                feat_mask,
                                                                                seq_len);
    const std::vector<float> fused_base = runtime.benchmark_run_base_lm_forward(fused_combined, seq_len, fused_cache, true);
    const std::vector<float> fused_blended =
        runtime.benchmark_run_masked_fsq_blend(fused_base, text_mask, feat_mask, seq_len);
    const BackendTransferStats fused_stats = backend.transfer_stats();

    REQUIRE(legacy_combined.size() == fused_combined.size());
    for (size_t i = 0; i < legacy_combined.size(); ++i) {
        REQUIRE(legacy_combined[i] == Catch::Approx(fused_combined[i]));
    }

    REQUIRE(legacy_blended.size() == fused_blended.size());
    for (size_t i = 0; i < legacy_blended.size(); ++i) {
        REQUIRE(legacy_blended[i] == Catch::Approx(fused_blended[i]));
    }

    const size_t expected_saved_d2h = static_cast<size_t>(hidden_size * seq_len) * sizeof(float);
    REQUIRE(fused_stats.device_to_host_bytes < legacy_stats.device_to_host_bytes);
    REQUIRE(legacy_stats.device_to_host_bytes - fused_stats.device_to_host_bytes >= expected_saved_d2h);

    const size_t legacy_total_bytes = legacy_stats.host_to_device_bytes + legacy_stats.device_to_host_bytes;
    const size_t fused_total_bytes = fused_stats.host_to_device_bytes + fused_stats.device_to_host_bytes;
    REQUIRE(fused_total_bytes < legacy_total_bytes);
}

TEST_CASE("prefill base-to-residual fused replacement matches legacy assembly while reducing transfer staging",
          "[runtime][prefill][residual-inputs][transfer][regression]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    const int seq_len = 4;
    const int hidden_size = runtime.base_lm().config().hidden_size;
    const std::vector<int32_t> tokens = {41, 42, 43, 44};
    std::vector<float> feat(static_cast<size_t>(seq_len * runtime.config().patch_size * runtime.config().feat_dim), 0.0f);
    for (size_t i = 0; i < feat.size(); ++i) {
        feat[i] = static_cast<float>((static_cast<int>(i % 47) - 23)) * 0.005f;
    }
    const std::vector<float> text_mask = {1.0f, 0.0f, 1.0f, 0.0f};
    const std::vector<float> feat_mask = {0.0f, 1.0f, 0.0f, 1.0f};

    const std::vector<float> combined_embed =
        runtime.benchmark_run_embedding_masked_locenc_sequence_to_lm_projection(tokens,
                                                                                feat,
                                                                                text_mask,
                                                                                feat_mask,
                                                                                seq_len);

    MiniCPMKVCache legacy_cache(runtime.base_lm().config().n_layer,
                                runtime.base_lm().config().n_kv_heads,
                                runtime.config().max_length,
                                runtime.base_lm().config().head_dim());
    MiniCPMKVCache fused_cache(runtime.base_lm().config().n_layer,
                               runtime.base_lm().config().n_kv_heads,
                               runtime.config().max_length,
                               runtime.base_lm().config().head_dim());
    legacy_cache.init(backend);
    fused_cache.init(backend);

    backend.reset_transfer_stats();
    const std::vector<float> legacy_base = runtime.benchmark_run_base_lm_forward(combined_embed, seq_len, legacy_cache, true);
    const std::vector<float> legacy_blended =
        runtime.benchmark_run_masked_fsq_blend(legacy_base, text_mask, feat_mask, seq_len);
    const BackendTransferStats legacy_stats = backend.transfer_stats();
    const std::vector<float> legacy_residual_inputs =
        build_expected_residual_inputs(runtime, backend, combined_embed, legacy_blended, feat_mask, seq_len);

    backend.reset_transfer_stats();
    const auto [fused_blended, fused_residual_inputs] =
        runtime.benchmark_run_prefill_base_to_residual_inputs(combined_embed,
                                                              text_mask,
                                                              feat_mask,
                                                              seq_len,
                                                              fused_cache,
                                                              true);
    const BackendTransferStats fused_stats = backend.transfer_stats();

    REQUIRE(legacy_blended.size() == fused_blended.size());
    for (size_t i = 0; i < legacy_blended.size(); ++i) {
        REQUIRE(legacy_blended[i] == Catch::Approx(fused_blended[i]));
    }

    REQUIRE(legacy_residual_inputs.size() == fused_residual_inputs.size());
    for (size_t i = 0; i < legacy_residual_inputs.size(); ++i) {
        REQUIRE(legacy_residual_inputs[i] == Catch::Approx(fused_residual_inputs[i]));
    }

    const size_t expected_saved_h2d = static_cast<size_t>(hidden_size * seq_len) * sizeof(float);
    REQUIRE(fused_stats.host_to_device_bytes < legacy_stats.host_to_device_bytes);
    REQUIRE(legacy_stats.host_to_device_bytes - fused_stats.host_to_device_bytes >= expected_saved_h2d);

    const size_t fused_total_bytes = fused_stats.host_to_device_bytes + fused_stats.device_to_host_bytes;
    const size_t legacy_total_bytes = legacy_stats.host_to_device_bytes + legacy_stats.device_to_host_bytes;
    REQUIRE(fused_total_bytes < legacy_total_bytes);
}

TEST_CASE("prefill direct-input residual path avoids combined-embed host staging while matching legacy results",
          "[runtime][prefill][residual-inputs][combined-embed][transfer]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    const int seq_len = 4;
    const int hidden_size = runtime.base_lm().config().hidden_size;
    const std::vector<int32_t> tokens = {51, 52, 53, 54};
    std::vector<float> feat(static_cast<size_t>(seq_len * runtime.config().patch_size * runtime.config().feat_dim), 0.0f);
    for (size_t i = 0; i < feat.size(); ++i) {
        feat[i] = static_cast<float>((static_cast<int>(i % 59) - 29)) * 0.0035f;
    }
    const std::vector<float> text_mask = {1.0f, 0.0f, 1.0f, 0.0f};
    const std::vector<float> feat_mask = {0.0f, 1.0f, 0.0f, 1.0f};

    MiniCPMKVCache legacy_cache(runtime.base_lm().config().n_layer,
                                runtime.base_lm().config().n_kv_heads,
                                runtime.config().max_length,
                                runtime.base_lm().config().head_dim());
    MiniCPMKVCache direct_cache(runtime.base_lm().config().n_layer,
                                runtime.base_lm().config().n_kv_heads,
                                runtime.config().max_length,
                                runtime.base_lm().config().head_dim());
    legacy_cache.init(backend);
    direct_cache.init(backend);

    backend.reset_transfer_stats();
    const std::vector<float> combined_embed =
        runtime.benchmark_run_embedding_masked_locenc_sequence_to_lm_projection(tokens,
                                                                                feat,
                                                                                text_mask,
                                                                                feat_mask,
                                                                                seq_len);
    const std::vector<float> legacy_base =
        runtime.benchmark_run_base_lm_forward(combined_embed, seq_len, legacy_cache, true);
    const std::vector<float> legacy_blended =
        runtime.benchmark_run_masked_fsq_blend(legacy_base, text_mask, feat_mask, seq_len);
    const std::vector<float> legacy_lm_hidden =
        slice_last_column_major_2d(legacy_blended, hidden_size, seq_len - 1);
    const BackendTransferStats legacy_stats = backend.transfer_stats();
    const std::vector<float> legacy_residual_inputs =
        build_expected_residual_inputs(runtime, backend, combined_embed, legacy_blended, feat_mask, seq_len);

    backend.reset_transfer_stats();
    const auto [direct_lm_hidden, direct_residual_inputs] =
        runtime.benchmark_run_prefill_inputs_to_residual_inputs(tokens,
                                                                feat,
                                                                text_mask,
                                                                feat_mask,
                                                                seq_len,
                                                                direct_cache,
                                                                true);
    const BackendTransferStats direct_stats = backend.transfer_stats();

    REQUIRE(legacy_lm_hidden.size() == static_cast<size_t>(hidden_size));
    REQUIRE(legacy_lm_hidden.size() == direct_lm_hidden.size());
    for (size_t i = 0; i < legacy_lm_hidden.size(); ++i) {
        REQUIRE(legacy_lm_hidden[i] == Catch::Approx(direct_lm_hidden[i]));
    }

    REQUIRE(legacy_residual_inputs.size() == direct_residual_inputs.size());
    for (size_t i = 0; i < legacy_residual_inputs.size(); ++i) {
        REQUIRE(legacy_residual_inputs[i] == Catch::Approx(direct_residual_inputs[i]));
    }

    const size_t combined_embed_bytes = static_cast<size_t>(hidden_size * seq_len) * sizeof(float);
    REQUIRE(direct_stats.device_to_host_bytes < legacy_stats.device_to_host_bytes);
    REQUIRE(legacy_stats.device_to_host_bytes - direct_stats.device_to_host_bytes >= combined_embed_bytes);
    REQUIRE(direct_stats.host_to_device_bytes < legacy_stats.host_to_device_bytes);
    REQUIRE(legacy_stats.host_to_device_bytes - direct_stats.host_to_device_bytes >= combined_embed_bytes);
}

TEST_CASE("prefill direct-input hidden-state path avoids residual-input host staging while matching legacy results",
          "[runtime][prefill][hidden-states][residual-inputs][transfer]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    const int seq_len = 4;
    const int hidden_size = runtime.base_lm().config().hidden_size;
    const std::vector<int32_t> tokens = {61, 62, 63, 64};
    std::vector<float> feat(static_cast<size_t>(seq_len * runtime.config().patch_size * runtime.config().feat_dim), 0.0f);
    for (size_t i = 0; i < feat.size(); ++i) {
        feat[i] = static_cast<float>((static_cast<int>(i % 61) - 30)) * 0.003f;
    }
    const std::vector<float> text_mask = {1.0f, 0.0f, 1.0f, 0.0f};
    const std::vector<float> feat_mask = {0.0f, 1.0f, 0.0f, 1.0f};

    MiniCPMKVCache old_base_cache(runtime.base_lm().config().n_layer,
                                  runtime.base_lm().config().n_kv_heads,
                                  runtime.config().max_length,
                                  runtime.base_lm().config().head_dim());
    MiniCPMKVCache old_residual_cache(runtime.residual_lm().config().n_layer,
                                      runtime.residual_lm().config().n_kv_heads,
                                      runtime.config().max_length,
                                      runtime.residual_lm().config().head_dim());
    MiniCPMKVCache new_base_cache(runtime.base_lm().config().n_layer,
                                  runtime.base_lm().config().n_kv_heads,
                                  runtime.config().max_length,
                                  runtime.base_lm().config().head_dim());
    MiniCPMKVCache new_residual_cache(runtime.residual_lm().config().n_layer,
                                      runtime.residual_lm().config().n_kv_heads,
                                      runtime.config().max_length,
                                      runtime.residual_lm().config().head_dim());
    old_base_cache.init(backend);
    old_residual_cache.init(backend);
    new_base_cache.init(backend);
    new_residual_cache.init(backend);

    backend.reset_transfer_stats();
    const auto [old_lm_hidden, old_residual_inputs] =
        runtime.benchmark_run_prefill_inputs_to_residual_inputs(tokens,
                                                                feat,
                                                                text_mask,
                                                                feat_mask,
                                                                seq_len,
                                                                old_base_cache,
                                                                true);
    const std::vector<float> old_residual_hidden =
        runtime.benchmark_run_residual_lm_forward_last_hidden(old_residual_inputs, seq_len, old_residual_cache, true);
    const BackendTransferStats old_stats = backend.transfer_stats();

    backend.reset_transfer_stats();
    const auto [new_lm_hidden, new_residual_hidden] =
        runtime.benchmark_run_prefill_inputs_to_hidden_states(tokens,
                                                              feat,
                                                              text_mask,
                                                              feat_mask,
                                                              seq_len,
                                                              new_base_cache,
                                                              new_residual_cache,
                                                              true);
    const BackendTransferStats new_stats = backend.transfer_stats();

    REQUIRE(old_lm_hidden.size() == new_lm_hidden.size());
    for (size_t i = 0; i < old_lm_hidden.size(); ++i) {
        REQUIRE(old_lm_hidden[i] == Catch::Approx(new_lm_hidden[i]));
    }

    REQUIRE(old_residual_hidden.size() == new_residual_hidden.size());
    for (size_t i = 0; i < old_residual_hidden.size(); ++i) {
        REQUIRE(old_residual_hidden[i] == Catch::Approx(new_residual_hidden[i]));
    }

    const size_t residual_inputs_bytes = static_cast<size_t>(hidden_size * seq_len) * sizeof(float);
    REQUIRE(new_stats.device_to_host_bytes < old_stats.device_to_host_bytes);
    REQUIRE(old_stats.device_to_host_bytes - new_stats.device_to_host_bytes >= residual_inputs_bytes);
    REQUIRE(new_stats.host_to_device_bytes < old_stats.host_to_device_bytes);
    REQUIRE(old_stats.host_to_device_bytes - new_stats.host_to_device_bytes >= residual_inputs_bytes);
}

TEST_CASE("stop predictor benchmark can consume persistent state directly with lower h2d",
          "[runtime][modules][stage2][stop]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    VoxCPMDecodeState state = runtime.create_decode_state();
    state.lm_hidden.assign(static_cast<size_t>(runtime.base_lm().config().hidden_size), 0.0f);
    for (size_t i = 0; i < state.lm_hidden.size(); ++i) {
        state.lm_hidden[i] = static_cast<float>((static_cast<int>(i % 11) - 5)) * 0.125f;
    }
    REQUIRE(state.persistent_state->set_lm_hidden_from_host(backend, state.lm_hidden.data(), state.lm_hidden.size()));

    backend.reset_transfer_stats();
    const std::vector<float> host_stop = runtime.benchmark_run_stop_predictor(state.lm_hidden);
    const BackendTransferStats host_stats = backend.transfer_stats();

    backend.reset_transfer_stats();
    const std::array<float, 2> state_stop = runtime.benchmark_run_stop_predictor_from_state(state, false);
    const BackendTransferStats state_stats = backend.transfer_stats();

    REQUIRE(host_stop.size() == state_stop.size());
    REQUIRE(host_stop[0] == Catch::Approx(state_stop[0]));
    REQUIRE(host_stop[1] == Catch::Approx(state_stop[1]));
    REQUIRE(state_stats.host_to_device_bytes < host_stats.host_to_device_bytes);
    REQUIRE(host_stats.host_to_device_bytes - state_stats.host_to_device_bytes >=
            state.lm_hidden.size() * sizeof(float));
}

TEST_CASE("lm_to_dit projection benchmark can consume persistent state directly with lower h2d",
          "[runtime][modules][stage2][projection]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    VoxCPMDecodeState state = runtime.create_decode_state();
    state.lm_hidden.assign(static_cast<size_t>(runtime.base_lm().config().hidden_size), 0.0f);
    for (size_t i = 0; i < state.lm_hidden.size(); ++i) {
        state.lm_hidden[i] = static_cast<float>((static_cast<int>(i % 7) - 3)) * 0.2f;
    }
    REQUIRE(state.persistent_state->set_lm_hidden_from_host(backend, state.lm_hidden.data(), state.lm_hidden.size()));

    backend.reset_transfer_stats();
    const std::vector<float> host_projection = runtime.benchmark_run_lm_to_dit_projection(state.lm_hidden);
    const BackendTransferStats host_stats = backend.transfer_stats();

    backend.reset_transfer_stats();
    const std::vector<float> state_projection = runtime.benchmark_run_lm_to_dit_projection_from_state(state);
    const BackendTransferStats state_stats = backend.transfer_stats();

    REQUIRE(host_projection.size() == state_projection.size());
    for (size_t i = 0; i < host_projection.size(); ++i) {
        REQUIRE(host_projection[i] == Catch::Approx(state_projection[i]));
    }
    REQUIRE(state_stats.host_to_device_bytes < host_stats.host_to_device_bytes);
    REQUIRE(host_stats.host_to_device_bytes - state_stats.host_to_device_bytes >=
            state.lm_hidden.size() * sizeof(float));
}

TEST_CASE("res_to_dit projection benchmark can consume persistent state directly with lower h2d",
          "[runtime][modules][stage2][projection]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    VoxCPMDecodeState state = runtime.create_decode_state();
    state.residual_hidden.assign(static_cast<size_t>(runtime.residual_lm().config().hidden_size), 0.0f);
    for (size_t i = 0; i < state.residual_hidden.size(); ++i) {
        state.residual_hidden[i] = static_cast<float>((static_cast<int>(i % 9) - 4)) * 0.15f;
    }
    REQUIRE(state.persistent_state->set_residual_hidden_from_host(backend,
                                                                  state.residual_hidden.data(),
                                                                  state.residual_hidden.size()));

    backend.reset_transfer_stats();
    const std::vector<float> host_projection = runtime.benchmark_run_res_to_dit_projection(state.residual_hidden);
    const BackendTransferStats host_stats = backend.transfer_stats();

    backend.reset_transfer_stats();
    const std::vector<float> state_projection = runtime.benchmark_run_res_to_dit_projection_from_state(state);
    const BackendTransferStats state_stats = backend.transfer_stats();

    REQUIRE(host_projection.size() == state_projection.size());
    for (size_t i = 0; i < host_projection.size(); ++i) {
        REQUIRE(host_projection[i] == Catch::Approx(state_projection[i]));
    }
    REQUIRE(state_stats.host_to_device_bytes < host_stats.host_to_device_bytes);
    REQUIRE(host_stats.host_to_device_bytes - state_stats.host_to_device_bytes >=
            state.residual_hidden.size() * sizeof(float));
}

TEST_CASE("fsq benchmark can consume persistent state directly with lower h2d",
          "[runtime][modules][stage2][fsq]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    VoxCPMDecodeState state = runtime.create_decode_state();
    state.lm_hidden.assign(static_cast<size_t>(runtime.base_lm().config().hidden_size), 0.0f);
    for (size_t i = 0; i < state.lm_hidden.size(); ++i) {
        state.lm_hidden[i] = static_cast<float>((static_cast<int>(i % 13) - 6)) * 0.1f;
    }
    REQUIRE(state.persistent_state->set_lm_hidden_from_host(backend, state.lm_hidden.data(), state.lm_hidden.size()));

    backend.reset_transfer_stats();
    const std::vector<float> host_fsq = runtime.benchmark_run_fsq_2d(state.lm_hidden, 1);
    const BackendTransferStats host_stats = backend.transfer_stats();

    backend.reset_transfer_stats();
    const std::vector<float> state_fsq = runtime.benchmark_run_fsq_from_state(state);
    const BackendTransferStats state_stats = backend.transfer_stats();

    REQUIRE(host_fsq.size() == state_fsq.size());
    for (size_t i = 0; i < host_fsq.size(); ++i) {
        REQUIRE(host_fsq[i] == Catch::Approx(state_fsq[i]));
    }
    REQUIRE(state_stats.host_to_device_bytes < host_stats.host_to_device_bytes);
    REQUIRE(host_stats.host_to_device_bytes - state_stats.host_to_device_bytes >=
            state.lm_hidden.size() * sizeof(float));
}

TEST_CASE("locenc patch benchmark can consume output-pool patch views directly with lower h2d",
          "[runtime][modules][stage2][locenc]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    VoxCPMDecodeState state = runtime.create_decode_state();
    std::vector<float> patch(static_cast<size_t>(runtime.config().patch_size * runtime.config().feat_dim), 0.0f);
    for (size_t i = 0; i < patch.size(); ++i) {
        patch[i] = static_cast<float>((static_cast<int>(i % 19) - 9)) * 0.05f;
    }
    REQUIRE(state.output_pool != nullptr);
    REQUIRE(state.output_pool->is_initialized());
    REQUIRE(state.output_pool->write_patch_to_latent_seq_from_host(backend, patch.data(), patch.size(), 0));

    backend.reset_transfer_stats();
    const std::vector<float> host_hidden = runtime.benchmark_run_locenc_patch(patch);
    const BackendTransferStats host_stats = backend.transfer_stats();

    backend.reset_transfer_stats();
    const std::vector<float> state_hidden = runtime.benchmark_run_locenc_patch_from_output_pool(state, 0);
    const BackendTransferStats state_stats = backend.transfer_stats();

    REQUIRE(host_hidden.size() == state_hidden.size());
    for (size_t i = 0; i < host_hidden.size(); ++i) {
        REQUIRE(host_hidden[i] == Catch::Approx(state_hidden[i]));
    }
    REQUIRE(state_stats.host_to_device_bytes < host_stats.host_to_device_bytes);
    REQUIRE(host_stats.host_to_device_bytes - state_stats.host_to_device_bytes >=
            patch.size() * sizeof(float));
}

TEST_CASE("locenc patch-to-lm benchmark can consume output-pool patch views directly with lower h2d",
          "[runtime][modules][stage2][locenc]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    VoxCPMDecodeState state = runtime.create_decode_state();
    std::vector<float> patch(static_cast<size_t>(runtime.config().patch_size * runtime.config().feat_dim), 0.0f);
    for (size_t i = 0; i < patch.size(); ++i) {
        patch[i] = static_cast<float>((static_cast<int>(i % 17) - 8)) * 0.075f;
    }
    REQUIRE(state.output_pool != nullptr);
    REQUIRE(state.output_pool->is_initialized());
    REQUIRE(state.output_pool->write_patch_to_latent_seq_from_host(backend, patch.data(), patch.size(), 0));

    backend.reset_transfer_stats();
    const std::vector<float> host_embed = runtime.benchmark_run_locenc_patch_to_lm_embed(patch);
    const BackendTransferStats host_stats = backend.transfer_stats();

    backend.reset_transfer_stats();
    const std::vector<float> state_embed = runtime.benchmark_run_locenc_patch_to_lm_embed_from_output_pool(state, 0);
    const BackendTransferStats state_stats = backend.transfer_stats();

    REQUIRE(host_embed.size() == state_embed.size());
    for (size_t i = 0; i < host_embed.size(); ++i) {
        REQUIRE(host_embed[i] == Catch::Approx(state_embed[i]));
    }
    REQUIRE(state_stats.host_to_device_bytes < host_stats.host_to_device_bytes);
    REQUIRE(host_stats.host_to_device_bytes - state_stats.host_to_device_bytes >=
            patch.size() * sizeof(float));
}

TEST_CASE("decode can skip host patch export while keeping output pool current",
          "[runtime][decode-state][no-host-patch-export]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    VoxCPMDecodeState seed = runtime.create_decode_state();
    seed.current_position = 4;
    seed.audio_frame_count = 1;
    seed.streaming_prefix_len = 3;
    seed.lm_hidden.assign(static_cast<size_t>(runtime.base_lm().config().hidden_size), 0.125f);
    seed.residual_hidden.assign(static_cast<size_t>(runtime.residual_lm().config().hidden_size), -0.25f);
    seed.prefix_feat_cond.assign(static_cast<size_t>(runtime.config().feat_dim * runtime.config().patch_size), 0.5f);
    REQUIRE(seed.output_pool->write_patch_to_latent_seq_from_host(backend,
                                                                  seed.prefix_feat_cond.data(),
                                                                  seed.prefix_feat_cond.size(),
                                                                  0));

    const std::vector<float> z(static_cast<size_t>(runtime.config().feat_dim * runtime.config().patch_size), 0.1f);

    VoxCPMDecodeOptions eager_options;
    eager_options.export_patch_to_host = true;
    backend.reset_transfer_stats();
    VoxCPMDecodeResult eager = runtime.decode(runtime.benchmark_clone_state(seed), z, 4, 1.5f, eager_options);
    const BackendTransferStats eager_stats = backend.transfer_stats();

    VoxCPMDecodeOptions no_host_options;
    no_host_options.export_patch_to_host = false;
    backend.reset_transfer_stats();
    VoxCPMDecodeResult no_host = runtime.decode(runtime.benchmark_clone_state(seed), z, 4, 1.5f, no_host_options);
    const BackendTransferStats no_host_stats = backend.transfer_stats();

    REQUIRE_FALSE(eager.output_0.empty());
    REQUIRE(no_host.output_0.empty());
    REQUIRE(no_host.output_1.output_pool != nullptr);
    REQUIRE(no_host.output_1.output_pool->is_initialized());
    REQUIRE(no_host.output_1.audio_frame_count == eager.output_1.audio_frame_count);
    REQUIRE(no_host.output_1.current_position == eager.output_1.current_position);
    REQUIRE(no_host.output_1.output_pool->export_patch_to_host(backend) == eager.output_0);
    REQUIRE(no_host.output_1.output_pool->export_latent_seq_to_host(backend, no_host.output_1.audio_frame_count) ==
            eager.output_1.output_pool->export_latent_seq_to_host(backend, eager.output_1.audio_frame_count));
    REQUIRE(no_host_stats.device_to_host_bytes < eager_stats.device_to_host_bytes);
    REQUIRE(eager_stats.device_to_host_bytes - no_host_stats.device_to_host_bytes >=
            eager.output_0.size() * sizeof(float));
}

TEST_CASE("decode can skip publishing stop logits to output pool to reduce d2d traffic",
          "[runtime][decode-state][no-stop-output-publish]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    VoxCPMDecodeState seed = runtime.create_decode_state();
    seed.current_position = 4;
    seed.audio_frame_count = 1;
    seed.streaming_prefix_len = 3;
    seed.lm_hidden.assign(static_cast<size_t>(runtime.base_lm().config().hidden_size), 0.125f);
    seed.residual_hidden.assign(static_cast<size_t>(runtime.residual_lm().config().hidden_size), -0.25f);
    seed.prefix_feat_cond.assign(static_cast<size_t>(runtime.config().feat_dim * runtime.config().patch_size), 0.5f);
    REQUIRE(seed.output_pool->write_patch_to_latent_seq_from_host(backend,
                                                                  seed.prefix_feat_cond.data(),
                                                                  seed.prefix_feat_cond.size(),
                                                                  0));

    const std::vector<float> z(static_cast<size_t>(runtime.config().feat_dim * runtime.config().patch_size), 0.1f);

    VoxCPMDecodeOptions publish_stop_options;
    publish_stop_options.export_patch_to_host = false;
    publish_stop_options.publish_stop_logits_to_output = true;
    backend.reset_transfer_stats();
    VoxCPMDecodeResult publish_stop =
        runtime.decode(runtime.benchmark_clone_state(seed), z, 4, 1.5f, publish_stop_options);
    const BackendTransferStats publish_stats = backend.transfer_stats();

    VoxCPMDecodeOptions skip_stop_options;
    skip_stop_options.export_patch_to_host = false;
    skip_stop_options.publish_stop_logits_to_output = false;
    backend.reset_transfer_stats();
    VoxCPMDecodeResult skip_stop =
        runtime.decode(runtime.benchmark_clone_state(seed), z, 4, 1.5f, skip_stop_options);
    const BackendTransferStats skip_stats = backend.transfer_stats();

    REQUIRE(publish_stop.output_0.empty());
    REQUIRE(skip_stop.output_0.empty());
    REQUIRE(skip_stop.output_2 == publish_stop.output_2);
    REQUIRE(skip_stop.output_1.output_pool != nullptr);
    REQUIRE(publish_stop.output_1.output_pool != nullptr);
    REQUIRE(skip_stop.output_1.output_pool->export_patch_to_host(backend) ==
            publish_stop.output_1.output_pool->export_patch_to_host(backend));
    REQUIRE(skip_stats.device_to_device_bytes < publish_stats.device_to_device_bytes);
    REQUIRE(publish_stats.device_to_device_bytes - skip_stats.device_to_device_bytes >=
            2 * sizeof(float));
}

TEST_CASE("decode can skip publishing patch output to output pool while keeping latent state correct",
          "[runtime][decode-state][no-patch-output-publish]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    VoxCPMDecodeState seed = runtime.create_decode_state();
    seed.current_position = 4;
    seed.audio_frame_count = 1;
    seed.streaming_prefix_len = 3;
    seed.lm_hidden.assign(static_cast<size_t>(runtime.base_lm().config().hidden_size), 0.125f);
    seed.residual_hidden.assign(static_cast<size_t>(runtime.residual_lm().config().hidden_size), -0.25f);
    seed.prefix_feat_cond.assign(static_cast<size_t>(runtime.config().feat_dim * runtime.config().patch_size), 0.5f);
    REQUIRE(seed.output_pool->write_patch_to_latent_seq_from_host(backend,
                                                                  seed.prefix_feat_cond.data(),
                                                                  seed.prefix_feat_cond.size(),
                                                                  0));

    const std::vector<float> z(static_cast<size_t>(runtime.config().feat_dim * runtime.config().patch_size), 0.1f);

    VoxCPMDecodeOptions publish_patch_options;
    publish_patch_options.export_patch_to_host = false;
    publish_patch_options.publish_stop_logits_to_output = false;
    publish_patch_options.publish_patch_to_output = true;
    backend.reset_transfer_stats();
    VoxCPMDecodeResult publish_patch =
        runtime.decode(runtime.benchmark_clone_state(seed), z, 4, 1.5f, publish_patch_options);
    const BackendTransferStats publish_stats = backend.transfer_stats();

    VoxCPMDecodeOptions skip_patch_options;
    skip_patch_options.export_patch_to_host = false;
    skip_patch_options.publish_stop_logits_to_output = false;
    skip_patch_options.publish_patch_to_output = false;
    backend.reset_transfer_stats();
    VoxCPMDecodeResult skip_patch =
        runtime.decode(runtime.benchmark_clone_state(seed), z, 4, 1.5f, skip_patch_options);
    const BackendTransferStats skip_stats = backend.transfer_stats();

    REQUIRE(publish_patch.output_0.empty());
    REQUIRE(skip_patch.output_0.empty());
    REQUIRE(skip_patch.output_2 == publish_patch.output_2);
    REQUIRE(skip_patch.output_1.audio_frame_count == publish_patch.output_1.audio_frame_count);
    REQUIRE(skip_patch.output_1.current_position == publish_patch.output_1.current_position);
    REQUIRE(skip_patch.output_1.output_pool->export_latent_seq_to_host(backend, skip_patch.output_1.audio_frame_count) ==
            publish_patch.output_1.output_pool->export_latent_seq_to_host(backend, publish_patch.output_1.audio_frame_count));

    std::vector<float> publish_prefix(runtime.config().feat_dim * runtime.config().patch_size, 0.0f);
    std::vector<float> skip_prefix(runtime.config().feat_dim * runtime.config().patch_size, 0.0f);
    REQUIRE(publish_patch.output_1.persistent_state->get_prefix_patch_to_host(backend,
                                                                              publish_prefix.data(),
                                                                              publish_prefix.size()));
    REQUIRE(skip_patch.output_1.persistent_state->get_prefix_patch_to_host(backend,
                                                                           skip_prefix.data(),
                                                                           skip_prefix.size()));
    REQUIRE(skip_prefix == publish_prefix);
    REQUIRE(skip_stats.device_to_device_bytes < publish_stats.device_to_device_bytes);
    REQUIRE(publish_stats.device_to_device_bytes - skip_stats.device_to_device_bytes >=
            publish_prefix.size() * sizeof(float));
}

TEST_CASE("decode can trust persistent state to skip redundant host-to-device state sync",
          "[runtime][decode-state][trust-persistent]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    VoxCPMDecodeState seed = runtime.create_decode_state();
    seed.current_position = 4;
    seed.audio_frame_count = 1;
    seed.streaming_prefix_len = 3;
    seed.lm_hidden.assign(static_cast<size_t>(runtime.base_lm().config().hidden_size), 0.125f);
    seed.residual_hidden.assign(static_cast<size_t>(runtime.residual_lm().config().hidden_size), -0.25f);
    seed.prefix_feat_cond.assign(static_cast<size_t>(runtime.config().feat_dim * runtime.config().patch_size), 0.5f);
    REQUIRE(seed.persistent_state->set_lm_hidden_from_host(backend, seed.lm_hidden.data(), seed.lm_hidden.size()));
    REQUIRE(seed.persistent_state->set_residual_hidden_from_host(backend,
                                                                 seed.residual_hidden.data(),
                                                                 seed.residual_hidden.size()));
    REQUIRE(seed.persistent_state->set_prefix_patch_from_host(backend,
                                                              seed.prefix_feat_cond.data(),
                                                              seed.prefix_feat_cond.size()));
    REQUIRE(seed.output_pool->write_patch_to_latent_seq_from_host(backend,
                                                                  seed.prefix_feat_cond.data(),
                                                                  seed.prefix_feat_cond.size(),
                                                                  0));

    const std::vector<float> z(static_cast<size_t>(runtime.config().feat_dim * runtime.config().patch_size), 0.1f);

    VoxCPMDecodeOptions auto_options;
    auto_options.export_patch_to_host = false;
    auto_options.publish_stop_logits_to_output = false;
    auto_options.publish_patch_to_output = false;

    backend.reset_transfer_stats();
    VoxCPMDecodeResult auto_sync =
        runtime.decode(runtime.benchmark_clone_state(seed), z, 4, 1.5f, auto_options);
    const BackendTransferStats auto_stats = backend.transfer_stats();

    VoxCPMDecodeOptions trusted_options = auto_options;
    trusted_options.trust_persistent_state = true;
    backend.reset_transfer_stats();
    VoxCPMDecodeResult trusted =
        runtime.decode(runtime.benchmark_clone_state(seed), z, 4, 1.5f, trusted_options);
    const BackendTransferStats trusted_stats = backend.transfer_stats();

    REQUIRE(auto_sync.output_0.empty());
    REQUIRE(trusted.output_0.empty());
    REQUIRE(trusted.output_2 == auto_sync.output_2);
    REQUIRE(trusted.output_1.current_position == auto_sync.output_1.current_position);
    REQUIRE(trusted.output_1.audio_frame_count == auto_sync.output_1.audio_frame_count);
    REQUIRE(trusted.output_1.output_pool->export_latent_seq_to_host(backend, trusted.output_1.audio_frame_count) ==
            auto_sync.output_1.output_pool->export_latent_seq_to_host(backend, auto_sync.output_1.audio_frame_count));
    REQUIRE(trusted_stats.host_to_device_bytes < auto_stats.host_to_device_bytes);
    REQUIRE(auto_stats.host_to_device_bytes - trusted_stats.host_to_device_bytes >=
            (seed.lm_hidden.size() + seed.residual_hidden.size() + seed.prefix_feat_cond.size()) * sizeof(float));
}

TEST_CASE("decode can skip eager prefix host shadow while keeping persistent state authoritative",
          "[runtime][decode-state][prefix-shadow]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    VoxCPMDecodeState seed = runtime.create_decode_state();
    seed.current_position = 4;
    seed.audio_frame_count = 1;
    seed.streaming_prefix_len = 3;
    seed.lm_hidden.assign(static_cast<size_t>(runtime.base_lm().config().hidden_size), 0.125f);
    seed.residual_hidden.assign(static_cast<size_t>(runtime.residual_lm().config().hidden_size), -0.25f);
    seed.prefix_feat_cond.assign(static_cast<size_t>(runtime.config().feat_dim * runtime.config().patch_size), 0.5f);
    REQUIRE(seed.persistent_state->set_lm_hidden_from_host(backend, seed.lm_hidden.data(), seed.lm_hidden.size()));
    REQUIRE(seed.persistent_state->set_residual_hidden_from_host(backend,
                                                                 seed.residual_hidden.data(),
                                                                 seed.residual_hidden.size()));
    REQUIRE(seed.persistent_state->set_prefix_patch_from_host(backend,
                                                              seed.prefix_feat_cond.data(),
                                                              seed.prefix_feat_cond.size()));
    REQUIRE(seed.output_pool->write_patch_to_latent_seq_from_host(backend,
                                                                  seed.prefix_feat_cond.data(),
                                                                  seed.prefix_feat_cond.size(),
                                                                  0));

    const std::vector<float> z(static_cast<size_t>(runtime.config().feat_dim * runtime.config().patch_size), 0.1f);

    ScopedEnvVar lazy_guard("VOXCPM_LAZY_HOST_STATE");
    ScopedEnvVar lazy_prefix_guard("VOXCPM_DECODE_LAZY_PREFIX_SHADOW");
    lazy_guard.set("0");

    lazy_prefix_guard.set("0");
    backend.reset_transfer_stats();
    VoxCPMDecodeResult eager = runtime.decode(runtime.benchmark_clone_state(seed), z, 4, 1.5f);
    const BackendTransferStats eager_stats = backend.transfer_stats();

    lazy_prefix_guard.set("1");
    backend.reset_transfer_stats();
    VoxCPMDecodeResult no_prefix_shadow = runtime.decode(runtime.benchmark_clone_state(seed), z, 4, 1.5f);
    const BackendTransferStats no_prefix_stats = backend.transfer_stats();

    REQUIRE(no_prefix_shadow.output_0 == eager.output_0);
    REQUIRE(no_prefix_shadow.output_2 == eager.output_2);
    REQUIRE(no_prefix_shadow.output_1.current_position == eager.output_1.current_position);
    REQUIRE(no_prefix_shadow.output_1.audio_frame_count == eager.output_1.audio_frame_count);
    REQUIRE(no_prefix_shadow.output_1.lm_hidden == eager.output_1.lm_hidden);
    REQUIRE(no_prefix_shadow.output_1.residual_hidden == eager.output_1.residual_hidden);
    REQUIRE(no_prefix_shadow.output_1.prefix_feat_cond.empty());
    REQUIRE(no_prefix_shadow.output_1.prefix_feat_cond.capacity() == 0);

    std::vector<float> persistent_prefix(runtime.config().feat_dim * runtime.config().patch_size, 0.0f);
    REQUIRE(no_prefix_shadow.output_1.persistent_state != nullptr);
    REQUIRE(no_prefix_shadow.output_1.persistent_state->get_prefix_patch_to_host(backend,
                                                                                 persistent_prefix.data(),
                                                                                 persistent_prefix.size()));
    REQUIRE(persistent_prefix == eager.output_1.prefix_feat_cond);
    REQUIRE(no_prefix_shadow.output_1.output_pool != nullptr);
    REQUIRE(no_prefix_shadow.output_1.output_pool->export_patch_to_host(backend) == eager.output_0);
    REQUIRE(no_prefix_stats.device_to_host_bytes < eager_stats.device_to_host_bytes);
    REQUIRE(eager_stats.device_to_host_bytes - no_prefix_stats.device_to_host_bytes >=
            persistent_prefix.size() * sizeof(float));
}

TEST_CASE("Legacy decode state now owns persistent state and output pool", "[runtime][decode-state]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    VoxCPMDecodeState state = runtime.create_decode_state();
    REQUIRE(state.base_lm_cache != nullptr);
    REQUIRE(state.residual_lm_cache != nullptr);
    REQUIRE(state.persistent_state != nullptr);
    REQUIRE(state.persistent_state->is_initialized());
    REQUIRE(state.output_pool != nullptr);
    REQUIRE(state.output_pool->is_initialized());
    REQUIRE(state.lm_hidden.empty());
    REQUIRE(state.residual_hidden.empty());
    REQUIRE(state.prefix_feat_cond.empty());
    REQUIRE(state.lm_hidden.capacity() == 0);
    REQUIRE(state.residual_hidden.capacity() == 0);
    REQUIRE(state.prefix_feat_cond.capacity() == 0);
}

TEST_CASE("create_decode_state zero-initializes persistent/output buffers without host bootstrap copies",
          "[runtime][decode-state][transfer]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    backend.reset_transfer_stats();
    VoxCPMDecodeState state = runtime.create_decode_state();
    const BackendTransferStats create_stats = backend.transfer_stats();

    REQUIRE(state.persistent_state != nullptr);
    REQUIRE(state.output_pool != nullptr);
    REQUIRE(create_stats.host_to_device_bytes == 0);

    backend.reset_transfer_stats();
    REQUIRE(state.output_pool->export_stop_logits_to_host(backend) == std::array<float, 2>{0.0f, 0.0f});
    REQUIRE(state.output_pool->export_patch_to_host(backend) ==
            std::vector<float>(static_cast<size_t>(runtime.config().feat_dim * runtime.config().patch_size), 0.0f));
    const BackendTransferStats export_stats = backend.transfer_stats();
    REQUIRE(export_stats.device_to_host_bytes == 0);
}

TEST_CASE("benchmark_clone_state preserves persistent-only decode state", "[runtime][decode-state][clone]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    VoxCPMDecodeState state = runtime.create_decode_state();
    const std::vector<float> lm_hidden(runtime.base_lm().config().hidden_size, 1.25f);
    const std::vector<float> residual_hidden(runtime.residual_lm().config().hidden_size, -0.5f);
    const std::vector<float> prefix_patch(runtime.config().feat_dim * runtime.config().patch_size, 0.75f);
    const std::vector<float> patch_output(runtime.config().feat_dim * runtime.config().patch_size, 0.33f);
    const std::array<float, 2> stop_logits = {0.2f, 0.8f};

    REQUIRE(state.persistent_state->set_lm_hidden_from_host(backend, lm_hidden.data(), lm_hidden.size()));
    REQUIRE(state.persistent_state->set_residual_hidden_from_host(backend,
                                                                  residual_hidden.data(),
                                                                  residual_hidden.size()));
    REQUIRE(state.persistent_state->set_prefix_patch_from_host(backend, prefix_patch.data(), prefix_patch.size()));
    REQUIRE(state.output_pool->publish_decode_outputs_from_host(backend,
                                                                patch_output.data(),
                                                                patch_output.size(),
                                                                stop_logits.data(),
                                                                stop_logits.size()));
    REQUIRE(state.output_pool->write_patch_to_latent_seq_from_host(backend, patch_output.data(), patch_output.size(), 0));
    state.audio_frame_count = 1;

    state.lm_hidden.clear();
    state.residual_hidden.clear();
    state.prefix_feat_cond.clear();

    VoxCPMDecodeState cloned = runtime.benchmark_clone_state(state);
    REQUIRE(cloned.persistent_state != nullptr);
    REQUIRE(cloned.output_pool != nullptr);
    REQUIRE(cloned.lm_hidden.empty());
    REQUIRE(cloned.residual_hidden.empty());
    REQUIRE(cloned.prefix_feat_cond.empty());
    REQUIRE(cloned.lm_hidden.capacity() == 0);
    REQUIRE(cloned.residual_hidden.capacity() == 0);
    REQUIRE(cloned.prefix_feat_cond.capacity() == 0);

    std::vector<float> cloned_lm(lm_hidden.size(), 0.0f);
    std::vector<float> cloned_residual(residual_hidden.size(), 0.0f);
    std::vector<float> cloned_prefix(prefix_patch.size(), 0.0f);
    REQUIRE(cloned.persistent_state->get_lm_hidden_to_host(backend, cloned_lm.data(), cloned_lm.size()));
    REQUIRE(cloned.persistent_state->get_residual_hidden_to_host(backend,
                                                                 cloned_residual.data(),
                                                                 cloned_residual.size()));
    REQUIRE(cloned.persistent_state->get_prefix_patch_to_host(backend,
                                                              cloned_prefix.data(),
                                                              cloned_prefix.size()));
    REQUIRE(cloned_lm == lm_hidden);
    REQUIRE(cloned_residual == residual_hidden);
    REQUIRE(cloned_prefix == prefix_patch);

    const HostDecodeOutput cloned_output = cloned.output_pool->export_decode_output_to_host(backend);
    REQUIRE(cloned_output.patch == patch_output);
    REQUIRE(cloned_output.stop_logits == stop_logits);
    REQUIRE(cloned.output_pool->export_latent_seq_to_host(backend, 1) == patch_output);
    REQUIRE(cloned.audio_frame_count == 1);
}

TEST_CASE("benchmark_clone_state preserves complete host shadow state without eager preallocation",
          "[runtime][decode-state][clone][host-shadow]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    VoxCPMDecodeState state = runtime.create_decode_state();
    state.current_position = 4;
    state.audio_frame_count = 1;
    state.streaming_prefix_len = 3;
    state.lm_hidden.assign(static_cast<size_t>(runtime.base_lm().config().hidden_size), 0.125f);
    state.residual_hidden.assign(static_cast<size_t>(runtime.residual_lm().config().hidden_size), -0.25f);
    state.prefix_feat_cond.assign(static_cast<size_t>(runtime.config().feat_dim * runtime.config().patch_size), 0.5f);

    VoxCPMDecodeState cloned = runtime.benchmark_clone_state(state);
    REQUIRE(cloned.current_position == state.current_position);
    REQUIRE(cloned.audio_frame_count == state.audio_frame_count);
    REQUIRE(cloned.streaming_prefix_len == state.streaming_prefix_len);
    REQUIRE(cloned.lm_hidden == state.lm_hidden);
    REQUIRE(cloned.residual_hidden == state.residual_hidden);
    REQUIRE(cloned.prefix_feat_cond == state.prefix_feat_cond);
}

TEST_CASE("benchmark_clone_state skips unpublished output-pool tensors to reduce d2d traffic",
          "[runtime][decode-state][clone][transfer]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    VoxCPMDecodeState latent_only = runtime.create_decode_state();
    const std::vector<float> patch(runtime.config().feat_dim * runtime.config().patch_size, 0.25f);
    latent_only.current_position = 4;
    latent_only.audio_frame_count = 1;
    REQUIRE(latent_only.output_pool->write_patch_to_latent_seq_from_host(backend, patch.data(), patch.size(), 0));

    VoxCPMDecodeState full_outputs = runtime.create_decode_state();
    full_outputs.current_position = 4;
    full_outputs.audio_frame_count = 1;
    REQUIRE(full_outputs.output_pool->write_patch_to_latent_seq_from_host(backend, patch.data(), patch.size(), 0));
    REQUIRE(full_outputs.output_pool->publish_patch_output_from_host(backend, patch.data(), patch.size()));
    const std::array<float, 2> stop_logits = {0.2f, 0.8f};
    REQUIRE(full_outputs.output_pool->publish_stop_logits_from_host(backend, stop_logits.data(), stop_logits.size()));

    backend.reset_transfer_stats();
    VoxCPMDecodeState cloned_latent_only = runtime.benchmark_clone_state(latent_only);
    const BackendTransferStats latent_only_stats = backend.transfer_stats();

    backend.reset_transfer_stats();
    VoxCPMDecodeState cloned_full_outputs = runtime.benchmark_clone_state(full_outputs);
    const BackendTransferStats full_outputs_stats = backend.transfer_stats();

    REQUIRE(cloned_latent_only.output_pool != nullptr);
    REQUIRE(cloned_full_outputs.output_pool != nullptr);
    REQUIRE(cloned_latent_only.output_pool->has_patch_output() == false);
    REQUIRE(cloned_latent_only.output_pool->has_stop_logits() == false);
    REQUIRE(cloned_full_outputs.output_pool->has_patch_output());
    REQUIRE(cloned_full_outputs.output_pool->has_stop_logits());
    REQUIRE(full_outputs_stats.device_to_device_bytes > latent_only_stats.device_to_device_bytes);
    REQUIRE(full_outputs_stats.device_to_device_bytes - latent_only_stats.device_to_device_bytes >=
            patch.size() * sizeof(float) + stop_logits.size() * sizeof(float));
}

TEST_CASE("benchmark_clone_state copies only the active latent_seq prefix",
          "[runtime][decode-state][clone][latent-prefix][transfer]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    const std::vector<float> patch(runtime.config().feat_dim * runtime.config().patch_size, 0.25f);

    VoxCPMDecodeState one_frame = runtime.create_decode_state();
    one_frame.audio_frame_count = 1;
    REQUIRE(one_frame.output_pool->write_patch_to_latent_seq_from_host(backend, patch.data(), patch.size(), 0));

    VoxCPMDecodeState three_frames = runtime.create_decode_state();
    three_frames.audio_frame_count = 3;
    REQUIRE(three_frames.output_pool->write_patch_to_latent_seq_from_host(backend, patch.data(), patch.size(), 0));
    REQUIRE(three_frames.output_pool->write_patch_to_latent_seq_from_host(backend, patch.data(), patch.size(), 1));
    REQUIRE(three_frames.output_pool->write_patch_to_latent_seq_from_host(backend, patch.data(), patch.size(), 2));

    backend.reset_transfer_stats();
    VoxCPMDecodeState cloned_one = runtime.benchmark_clone_state(one_frame);
    const BackendTransferStats one_stats = backend.transfer_stats();

    backend.reset_transfer_stats();
    VoxCPMDecodeState cloned_three = runtime.benchmark_clone_state(three_frames);
    const BackendTransferStats three_stats = backend.transfer_stats();

    REQUIRE(cloned_one.output_pool != nullptr);
    REQUIRE(cloned_three.output_pool != nullptr);
    REQUIRE(cloned_one.output_pool->export_latent_seq_to_host(backend, 1) == patch);
    REQUIRE(cloned_three.output_pool->export_latent_seq_to_host(backend, 3).size() == patch.size() * 3);
    REQUIRE(three_stats.device_to_device_bytes > one_stats.device_to_device_bytes);
    REQUIRE(three_stats.device_to_device_bytes - one_stats.device_to_device_bytes >=
            patch.size() * sizeof(float) * 2);
}

TEST_CASE("lazy host state keeps chained decode aligned with eager state", "[runtime][decode-state][lazy]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    VoxCPMDecodeState seed = runtime.create_decode_state();
    seed.current_position = 4;
    seed.audio_frame_count = 1;
    seed.streaming_prefix_len = 3;
    seed.lm_hidden.assign(static_cast<size_t>(runtime.base_lm().config().hidden_size), 0.125f);
    seed.residual_hidden.assign(static_cast<size_t>(runtime.residual_lm().config().hidden_size), -0.25f);
    seed.prefix_feat_cond.assign(static_cast<size_t>(runtime.config().feat_dim * runtime.config().patch_size), 0.5f);
    REQUIRE(seed.output_pool->write_patch_to_latent_seq_from_host(backend,
                                                                  seed.prefix_feat_cond.data(),
                                                                  seed.prefix_feat_cond.size(),
                                                                  0));

    const std::vector<float> z0(static_cast<size_t>(runtime.config().feat_dim * runtime.config().patch_size), 0.1f);
    const std::vector<float> z1(static_cast<size_t>(runtime.config().feat_dim * runtime.config().patch_size), -0.2f);

    ScopedEnvVar lazy_guard("VOXCPM_LAZY_HOST_STATE");

    lazy_guard.set("0");
    VoxCPMDecodeResult eager_first = runtime.decode(runtime.benchmark_clone_state(seed), z0, 4, 1.5f);
    VoxCPMDecodeResult eager_second = runtime.decode(runtime.benchmark_clone_state(eager_first.output_1), z1, 4, 1.5f);

    lazy_guard.set("1");
    VoxCPMDecodeResult lazy_first = runtime.decode(runtime.benchmark_clone_state(seed), z0, 4, 1.5f);
    VoxCPMDecodeResult lazy_second = runtime.decode(runtime.benchmark_clone_state(lazy_first.output_1), z1, 4, 1.5f);

    REQUIRE(lazy_first.output_1.lm_hidden.empty());
    REQUIRE(lazy_first.output_1.residual_hidden.empty());
    REQUIRE(lazy_first.output_1.prefix_feat_cond.empty());
    REQUIRE(lazy_second.output_1.lm_hidden.empty());
    REQUIRE(lazy_second.output_1.residual_hidden.empty());
    REQUIRE(lazy_second.output_1.prefix_feat_cond.empty());

    REQUIRE(lazy_first.output_0 == eager_first.output_0);
    REQUIRE(lazy_first.output_2 == eager_first.output_2);
    REQUIRE(lazy_second.output_0 == eager_second.output_0);
    REQUIRE(lazy_second.output_2 == eager_second.output_2);

    REQUIRE(lazy_first.output_1.current_position == eager_first.output_1.current_position);
    REQUIRE(lazy_second.output_1.current_position == eager_second.output_1.current_position);
    REQUIRE(lazy_first.output_1.audio_frame_count == eager_first.output_1.audio_frame_count);
    REQUIRE(lazy_second.output_1.audio_frame_count == eager_second.output_1.audio_frame_count);

    REQUIRE(lazy_first.output_1.persistent_state != nullptr);
    REQUIRE(lazy_second.output_1.persistent_state != nullptr);

    const std::vector<float> lazy_first_lm =
        read_persistent_tensor(backend, *lazy_first.output_1.persistent_state, lazy_first.output_1.persistent_state->lm_hidden());
    const std::vector<float> lazy_first_residual =
        read_persistent_tensor(backend, *lazy_first.output_1.persistent_state, lazy_first.output_1.persistent_state->residual_hidden());
    const std::vector<float> lazy_first_prefix =
        read_persistent_tensor(backend, *lazy_first.output_1.persistent_state, lazy_first.output_1.persistent_state->prefix_patch());
    REQUIRE(lazy_first_lm == eager_first.output_1.lm_hidden);
    REQUIRE(lazy_first_residual == eager_first.output_1.residual_hidden);
    REQUIRE(lazy_first_prefix == eager_first.output_1.prefix_feat_cond);

    const std::vector<float> lazy_second_lm =
        read_persistent_tensor(backend, *lazy_second.output_1.persistent_state, lazy_second.output_1.persistent_state->lm_hidden());
    const std::vector<float> lazy_second_residual =
        read_persistent_tensor(backend, *lazy_second.output_1.persistent_state, lazy_second.output_1.persistent_state->residual_hidden());
    const std::vector<float> lazy_second_prefix =
        read_persistent_tensor(backend, *lazy_second.output_1.persistent_state, lazy_second.output_1.persistent_state->prefix_patch());
    REQUIRE(lazy_second_lm == eager_second.output_1.lm_hidden);
    REQUIRE(lazy_second_residual == eager_second.output_1.residual_hidden);
    REQUIRE(lazy_second_prefix == eager_second.output_1.prefix_feat_cond);
}

TEST_CASE("prefill publishes persistent-first state in lazy mode", "[runtime][prefill][lazy]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    const int seq_len = 2;
    std::vector<int32_t> text(seq_len, 1);
    std::vector<int32_t> text_mask(seq_len, 1);
    std::vector<int32_t> feat_mask(seq_len, 1);
    std::vector<float> feat(static_cast<size_t>(seq_len * runtime.config().patch_size * runtime.config().feat_dim), 0.05f);

    ScopedEnvVar lazy_guard("VOXCPM_LAZY_HOST_STATE");
    lazy_guard.set("1");

    VoxCPMDecodeState state = runtime.prefill(text, text_mask, feat, feat_mask, seq_len, 3);
    REQUIRE(state.persistent_state != nullptr);
    REQUIRE(state.output_pool != nullptr);
    REQUIRE(state.audio_frame_count == seq_len);
    REQUIRE(state.lm_hidden.empty());
    REQUIRE(state.residual_hidden.empty());
    REQUIRE(state.prefix_feat_cond.empty());

    std::vector<float> persistent_lm(runtime.base_lm().config().hidden_size, 0.0f);
    std::vector<float> persistent_residual(runtime.residual_lm().config().hidden_size, 0.0f);
    std::vector<float> persistent_prefix(runtime.config().feat_dim * runtime.config().patch_size, 0.0f);
    REQUIRE(state.persistent_state->get_lm_hidden_to_host(backend, persistent_lm.data(), persistent_lm.size()));
    REQUIRE(state.persistent_state->get_residual_hidden_to_host(backend,
                                                                persistent_residual.data(),
                                                                persistent_residual.size()));
    REQUIRE(state.persistent_state->get_prefix_patch_to_host(backend,
                                                             persistent_prefix.data(),
                                                             persistent_prefix.size()));
    REQUIRE_FALSE(persistent_lm.empty());
    REQUIRE_FALSE(persistent_residual.empty());
    REQUIRE_FALSE(persistent_prefix.empty());

    const HostDecodeOutput staged_output = state.output_pool->export_decode_output_to_host(backend);
    REQUIRE(staged_output.patch == persistent_prefix);
    REQUIRE(staged_output.stop_logits == std::array<float, 2>{0.0f, 0.0f});
    REQUIRE(state.output_pool->export_latent_seq_to_host(backend, seq_len).size() ==
            static_cast<size_t>(seq_len * runtime.config().feat_dim * runtime.config().patch_size));
    const std::vector<float> latent = state.output_pool->export_latent_seq_to_host(backend, seq_len);
    REQUIRE(std::equal(persistent_prefix.begin(),
                       persistent_prefix.end(),
                       latent.begin() + static_cast<std::ptrdiff_t>((seq_len - 1) * persistent_prefix.size())));

    const std::vector<float> z(static_cast<size_t>(runtime.config().feat_dim * runtime.config().patch_size), -0.15f);
    VoxCPMDecodeResult result = runtime.decode(runtime.benchmark_clone_state(state), z, 4, 1.25f);
    REQUIRE(result.output_1.current_position == seq_len + 1);
    REQUIRE(result.output_1.audio_frame_count == seq_len + 1);
    REQUIRE(result.output_1.output_pool->export_latent_seq_to_host(backend, seq_len + 1).size() ==
            static_cast<size_t>((seq_len + 1) * runtime.config().feat_dim * runtime.config().patch_size));
}

TEST_CASE("prefill can skip eager prefix host shadow while keeping persistent state authoritative",
          "[runtime][prefill][prefix-shadow]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    const int seq_len = 2;
    std::vector<int32_t> text(seq_len, 1);
    std::vector<int32_t> text_mask(seq_len, 1);
    std::vector<int32_t> feat_mask(seq_len, 1);
    std::vector<float> feat(static_cast<size_t>(seq_len * runtime.config().patch_size * runtime.config().feat_dim), 0.05f);
    const std::vector<float> z(static_cast<size_t>(runtime.config().feat_dim * runtime.config().patch_size), -0.15f);

    ScopedEnvVar lazy_guard("VOXCPM_LAZY_HOST_STATE");
    ScopedEnvVar lazy_prefix_guard("VOXCPM_PREFILL_LAZY_PREFIX_SHADOW");

    lazy_guard.set("0");
    lazy_prefix_guard.set("0");
    backend.reset_transfer_stats();
    VoxCPMDecodeState eager = runtime.prefill(text, text_mask, feat, feat_mask, seq_len, 3);
    const BackendTransferStats eager_stats = backend.transfer_stats();

    lazy_prefix_guard.set("1");
    backend.reset_transfer_stats();
    VoxCPMDecodeState no_prefix_shadow = runtime.prefill(text, text_mask, feat, feat_mask, seq_len, 3);
    const BackendTransferStats no_prefix_stats = backend.transfer_stats();

    REQUIRE(no_prefix_shadow.current_position == eager.current_position);
    REQUIRE(no_prefix_shadow.audio_frame_count == eager.audio_frame_count);
    REQUIRE(no_prefix_shadow.lm_hidden == eager.lm_hidden);
    REQUIRE(no_prefix_shadow.residual_hidden == eager.residual_hidden);
    REQUIRE(no_prefix_shadow.prefix_feat_cond.empty());
    REQUIRE(no_prefix_shadow.prefix_feat_cond.capacity() == 0);
    REQUIRE(no_prefix_stats.host_to_device_bytes == eager_stats.host_to_device_bytes);
    REQUIRE(no_prefix_stats.device_to_host_bytes <= eager_stats.device_to_host_bytes);

    std::vector<float> persistent_prefix(runtime.config().feat_dim * runtime.config().patch_size, 0.0f);
    REQUIRE(no_prefix_shadow.persistent_state != nullptr);
    REQUIRE(no_prefix_shadow.persistent_state->get_prefix_patch_to_host(backend,
                                                                        persistent_prefix.data(),
                                                                        persistent_prefix.size()));
    REQUIRE(persistent_prefix == eager.prefix_feat_cond);
    REQUIRE(no_prefix_shadow.output_pool != nullptr);
    REQUIRE(no_prefix_shadow.output_pool->export_patch_to_host(backend) == eager.prefix_feat_cond);

    VoxCPMDecodeResult eager_result = runtime.decode(runtime.benchmark_clone_state(eager), z, 4, 1.25f);
    VoxCPMDecodeResult no_prefix_result =
        runtime.decode(runtime.benchmark_clone_state(no_prefix_shadow), z, 4, 1.25f);
    REQUIRE(no_prefix_result.output_0 == eager_result.output_0);
    REQUIRE(no_prefix_result.output_2 == eager_result.output_2);
    REQUIRE(no_prefix_result.output_1.current_position == eager_result.output_1.current_position);
    REQUIRE(no_prefix_result.output_1.audio_frame_count == eager_result.output_1.audio_frame_count);
}

TEST_CASE("prefill can consume backend-resident prompt patch tensors on the real main path",
          "[runtime][prefill][prompt-tensor][transfer]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    const int seq_len = 4;
    std::vector<int32_t> text = {1, 7, 3, 9};
    std::vector<int32_t> text_mask = {1, 0, 0, 1};
    std::vector<int32_t> feat_mask = {0, 1, 1, 0};
    const size_t patch_elem_count = static_cast<size_t>(runtime.config().feat_dim * runtime.config().patch_size);

    std::vector<float> feat(static_cast<size_t>(seq_len) * patch_elem_count, 0.0f);
    for (size_t i = 0; i < feat.size(); ++i) {
        feat[i] = static_cast<float>((static_cast<int>(i % 29) - 14)) * 0.015625f;
    }

    const std::vector<float> prompt_patches = extract_prompt_patch_range(feat, feat_mask, patch_elem_count);
    const int audio_frame_count = static_cast<int>(prompt_patches.size() / patch_elem_count);

    VoxCPMContext prompt_ctx(ContextType::Graph, 1, 1);
    ggml_tensor* prompt_patch_tensor =
        prompt_ctx.new_tensor_2d(GGML_TYPE_F32,
                                 runtime.config().feat_dim,
                                 audio_frame_count * runtime.config().patch_size);
    REQUIRE(prompt_patch_tensor != nullptr);
    ggml_backend_buffer_t prompt_buffer = backend.alloc_buffer(prompt_ctx.raw_context(), BufferUsage::Output);
    REQUIRE(prompt_buffer != nullptr);
    backend.tensor_set(prompt_patch_tensor, prompt_patches.data(), 0, prompt_patches.size() * sizeof(float));

    backend.reset_transfer_stats();
    VoxCPMDecodeState host_state = runtime.prefill(text, text_mask, feat, feat_mask, seq_len, 3);
    const BackendTransferStats host_stats = backend.transfer_stats();

    backend.reset_transfer_stats();
    VoxCPMDecodeState tensor_state =
        runtime.prefill_with_prompt_patch_tensor(text, text_mask, feat, prompt_patch_tensor, feat_mask, seq_len, 3);
    const BackendTransferStats tensor_stats = backend.transfer_stats();

    REQUIRE(tensor_state.current_position == host_state.current_position);
    REQUIRE(tensor_state.audio_frame_count == host_state.audio_frame_count);
    REQUIRE(tensor_state.audio_frame_count == audio_frame_count);
    REQUIRE(tensor_state.lm_hidden == host_state.lm_hidden);
    REQUIRE(tensor_state.residual_hidden == host_state.residual_hidden);
    REQUIRE(tensor_state.prefix_feat_cond == host_state.prefix_feat_cond);
    REQUIRE(tensor_state.output_pool->export_latent_seq_to_host(backend, audio_frame_count) ==
            host_state.output_pool->export_latent_seq_to_host(backend, audio_frame_count));
    REQUIRE(tensor_state.output_pool->export_patch_to_host(backend) ==
            host_state.output_pool->export_patch_to_host(backend));

    std::vector<float> tensor_prefix(patch_elem_count, 0.0f);
    std::vector<float> host_prefix(patch_elem_count, 0.0f);
    REQUIRE(tensor_state.persistent_state->get_prefix_patch_to_host(backend,
                                                                    tensor_prefix.data(),
                                                                    tensor_prefix.size()));
    REQUIRE(host_state.persistent_state->get_prefix_patch_to_host(backend,
                                                                  host_prefix.data(),
                                                                  host_prefix.size()));
    REQUIRE(tensor_prefix == host_prefix);

    REQUIRE(tensor_stats.host_to_device_bytes < host_stats.host_to_device_bytes);
    REQUIRE(host_stats.host_to_device_bytes - tensor_stats.host_to_device_bytes >=
            prompt_patches.size() * sizeof(float));

    backend.free_buffer(prompt_buffer);
}

TEST_CASE("prefill can consume backend-resident feature tensors on the real main path",
          "[runtime][prefill][feature-tensor][transfer]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    const int seq_len = 4;
    std::vector<int32_t> text = {2, 5, 7, 11};
    std::vector<int32_t> text_mask = {1, 0, 0, 1};
    std::vector<int32_t> feat_mask = {0, 1, 1, 0};
    const size_t feat_elem_count =
        static_cast<size_t>(seq_len * runtime.config().feat_dim * runtime.config().patch_size);
    const size_t patch_elem_count = static_cast<size_t>(runtime.config().feat_dim * runtime.config().patch_size);
    std::vector<float> feat(feat_elem_count, 0.0f);
    for (size_t i = 0; i < feat.size(); ++i) {
        feat[i] = static_cast<float>((static_cast<int>(i % 31) - 15)) * 0.0125f;
    }
    const std::vector<float> prompt_patches = extract_prompt_patch_range(feat, feat_mask, patch_elem_count);
    const int audio_frame_count = static_cast<int>(prompt_patches.size() / patch_elem_count);

    VoxCPMContext feat_ctx(ContextType::Graph, 1, 1);
    ggml_tensor* feat_tensor = feat_ctx.new_tensor_3d(GGML_TYPE_F32,
                                                      runtime.config().feat_dim,
                                                      runtime.config().patch_size,
                                                      seq_len);
    REQUIRE(feat_tensor != nullptr);
    ggml_backend_buffer_t feat_buffer = backend.alloc_buffer(feat_ctx.raw_context(), BufferUsage::Output);
    REQUIRE(feat_buffer != nullptr);
    backend.tensor_set(feat_tensor, feat.data(), 0, feat.size() * sizeof(float));

    backend.reset_transfer_stats();
    VoxCPMDecodeState host_state = runtime.prefill(text, text_mask, feat, feat_mask, seq_len, 3);
    const BackendTransferStats host_stats = backend.transfer_stats();

    backend.reset_transfer_stats();
    VoxCPMDecodeState tensor_state =
        runtime.prefill_with_feature_tensor(text, text_mask, feat_tensor, feat_mask, seq_len, 3);
    const BackendTransferStats tensor_stats = backend.transfer_stats();

    REQUIRE(tensor_state.current_position == host_state.current_position);
    REQUIRE(tensor_state.audio_frame_count == host_state.audio_frame_count);
    REQUIRE(tensor_state.audio_frame_count == audio_frame_count);
    REQUIRE(tensor_state.lm_hidden == host_state.lm_hidden);
    REQUIRE(tensor_state.residual_hidden == host_state.residual_hidden);
    REQUIRE(tensor_state.prefix_feat_cond == host_state.prefix_feat_cond);
    REQUIRE(tensor_state.output_pool->export_latent_seq_to_host(backend, audio_frame_count) ==
            host_state.output_pool->export_latent_seq_to_host(backend, audio_frame_count));
    REQUIRE(tensor_state.output_pool->export_patch_to_host(backend) ==
            host_state.output_pool->export_patch_to_host(backend));

    std::vector<float> tensor_lm(runtime.base_lm().config().hidden_size, 0.0f);
    std::vector<float> host_lm(runtime.base_lm().config().hidden_size, 0.0f);
    REQUIRE(tensor_state.persistent_state->get_lm_hidden_to_host(backend, tensor_lm.data(), tensor_lm.size()));
    REQUIRE(host_state.persistent_state->get_lm_hidden_to_host(backend, host_lm.data(), host_lm.size()));
    REQUIRE(tensor_lm == host_lm);

    std::vector<float> tensor_residual(runtime.residual_lm().config().hidden_size, 0.0f);
    std::vector<float> host_residual(runtime.residual_lm().config().hidden_size, 0.0f);
    REQUIRE(tensor_state.persistent_state->get_residual_hidden_to_host(backend,
                                                                       tensor_residual.data(),
                                                                       tensor_residual.size()));
    REQUIRE(host_state.persistent_state->get_residual_hidden_to_host(backend,
                                                                     host_residual.data(),
                                                                     host_residual.size()));
    REQUIRE(tensor_residual == host_residual);

    REQUIRE(tensor_stats.host_to_device_bytes < host_stats.host_to_device_bytes);
    REQUIRE(host_stats.host_to_device_bytes - tensor_stats.host_to_device_bytes >=
            (feat.size() + prompt_patches.size()) * sizeof(float));

    backend.free_buffer(feat_buffer);
}

TEST_CASE("prefill can consume backend-resident token and mask tensors on the real main path",
          "[runtime][prefill][input-tensors][transfer]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    const int seq_len = 4;
    std::vector<int32_t> text = {2, 5, 7, 11};
    std::vector<int32_t> text_mask_i = {1, 0, 0, 1};
    std::vector<int32_t> feat_mask_i = {0, 1, 1, 0};
    std::vector<float> text_mask_f(seq_len, 0.0f);
    std::vector<float> feat_mask_f(seq_len, 0.0f);
    for (int i = 0; i < seq_len; ++i) {
        text_mask_f[static_cast<size_t>(i)] = text_mask_i[static_cast<size_t>(i)] != 0 ? 1.0f : 0.0f;
        feat_mask_f[static_cast<size_t>(i)] = feat_mask_i[static_cast<size_t>(i)] != 0 ? 1.0f : 0.0f;
    }

    const size_t feat_elem_count =
        static_cast<size_t>(seq_len * runtime.config().feat_dim * runtime.config().patch_size);

    std::vector<float> feat(feat_elem_count, 0.0f);
    for (size_t i = 0; i < feat.size(); ++i) {
        feat[i] = static_cast<float>((static_cast<int>(i % 31) - 15)) * 0.0125f;
    }

    VoxCPMContext input_ctx(ContextType::Graph, 4, 4);
    ggml_tensor* text_tensor = input_ctx.new_tensor_1d(GGML_TYPE_I32, seq_len);
    ggml_tensor* text_mask_tensor = input_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    ggml_tensor* feat_tensor = input_ctx.new_tensor_3d(GGML_TYPE_F32,
                                                       runtime.config().feat_dim,
                                                       runtime.config().patch_size,
                                                       seq_len);
    ggml_tensor* feat_mask_tensor = input_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    REQUIRE(text_tensor != nullptr);
    REQUIRE(text_mask_tensor != nullptr);
    REQUIRE(feat_tensor != nullptr);
    REQUIRE(feat_mask_tensor != nullptr);
    ggml_backend_buffer_t input_buffer = backend.alloc_buffer(input_ctx.raw_context(), BufferUsage::Output);
    REQUIRE(input_buffer != nullptr);
    backend.tensor_set(text_tensor, text.data(), 0, text.size() * sizeof(int32_t));
    backend.tensor_set(text_mask_tensor, text_mask_f.data(), 0, text_mask_f.size() * sizeof(float));
    backend.tensor_set(feat_tensor, feat.data(), 0, feat.size() * sizeof(float));
    backend.tensor_set(feat_mask_tensor, feat_mask_f.data(), 0, feat_mask_f.size() * sizeof(float));

    backend.reset_transfer_stats();
    VoxCPMDecodeState feature_state =
        runtime.prefill_with_feature_tensor(text, text_mask_i, feat_tensor, feat_mask_i, seq_len, 3);
    const BackendTransferStats feature_stats = backend.transfer_stats();

    backend.reset_transfer_stats();
    VoxCPMDecodeState input_state =
        runtime.prefill_with_input_tensors(text_tensor, text_mask_tensor, feat_tensor, feat_mask_tensor, seq_len, 3);
    const BackendTransferStats input_stats = backend.transfer_stats();

    REQUIRE(input_state.current_position == feature_state.current_position);
    REQUIRE(input_state.audio_frame_count == feature_state.audio_frame_count);
    REQUIRE(input_state.lm_hidden == feature_state.lm_hidden);
    REQUIRE(input_state.residual_hidden == feature_state.residual_hidden);
    REQUIRE(input_state.prefix_feat_cond == feature_state.prefix_feat_cond);
    REQUIRE(input_state.output_pool->export_latent_seq_to_host(backend, input_state.audio_frame_count) ==
            feature_state.output_pool->export_latent_seq_to_host(backend, feature_state.audio_frame_count));
    REQUIRE(input_state.output_pool->export_patch_to_host(backend) ==
            feature_state.output_pool->export_patch_to_host(backend));

    std::vector<float> input_lm(runtime.base_lm().config().hidden_size, 0.0f);
    std::vector<float> feature_lm(runtime.base_lm().config().hidden_size, 0.0f);
    REQUIRE(input_state.persistent_state->get_lm_hidden_to_host(backend, input_lm.data(), input_lm.size()));
    REQUIRE(feature_state.persistent_state->get_lm_hidden_to_host(backend, feature_lm.data(), feature_lm.size()));
    REQUIRE(input_lm == feature_lm);

    REQUIRE(input_stats.host_to_device_bytes < feature_stats.host_to_device_bytes);
    REQUIRE(feature_stats.host_to_device_bytes - input_stats.host_to_device_bytes >=
            text.size() * sizeof(int32_t) + text_mask_f.size() * sizeof(float) + feat_mask_f.size() * sizeof(float));
    REQUIRE(input_stats.device_to_host_bytes >= feature_stats.device_to_host_bytes);
    REQUIRE(input_stats.device_to_host_bytes <=
            feature_stats.device_to_host_bytes + feat_mask_f.size() * sizeof(float));

    backend.free_buffer(input_buffer);
}

TEST_CASE("prefill can avoid feat_mask host control mirror with explicit prompt positions",
          "[runtime][prefill][prompt-positions][transfer]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    const int seq_len = 4;
    std::vector<int32_t> text = {2, 5, 7, 11};
    std::vector<int32_t> text_mask_i = {1, 0, 0, 1};
    std::vector<int32_t> feat_mask_i = {0, 1, 1, 0};
    std::vector<int32_t> prompt_positions = {1, 2};
    std::vector<float> text_mask_f(seq_len, 0.0f);
    std::vector<float> feat_mask_f(seq_len, 0.0f);
    for (int i = 0; i < seq_len; ++i) {
        text_mask_f[static_cast<size_t>(i)] = text_mask_i[static_cast<size_t>(i)] != 0 ? 1.0f : 0.0f;
        feat_mask_f[static_cast<size_t>(i)] = feat_mask_i[static_cast<size_t>(i)] != 0 ? 1.0f : 0.0f;
    }

    const size_t feat_elem_count =
        static_cast<size_t>(seq_len * runtime.config().feat_dim * runtime.config().patch_size);
    std::vector<float> feat(feat_elem_count, 0.0f);
    for (size_t i = 0; i < feat.size(); ++i) {
        feat[i] = static_cast<float>((static_cast<int>(i % 31) - 15)) * 0.0125f;
    }

    VoxCPMContext input_ctx(ContextType::Graph, 4, 4);
    ggml_tensor* text_tensor = input_ctx.new_tensor_1d(GGML_TYPE_I32, seq_len);
    ggml_tensor* text_mask_tensor = input_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    ggml_tensor* feat_tensor = input_ctx.new_tensor_3d(GGML_TYPE_F32,
                                                       runtime.config().feat_dim,
                                                       runtime.config().patch_size,
                                                       seq_len);
    ggml_tensor* feat_mask_tensor = input_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    REQUIRE(text_tensor != nullptr);
    REQUIRE(text_mask_tensor != nullptr);
    REQUIRE(feat_tensor != nullptr);
    REQUIRE(feat_mask_tensor != nullptr);
    ggml_backend_buffer_t input_buffer = backend.alloc_buffer(input_ctx.raw_context(), BufferUsage::Output);
    REQUIRE(input_buffer != nullptr);
    backend.tensor_set(text_tensor, text.data(), 0, text.size() * sizeof(int32_t));
    backend.tensor_set(text_mask_tensor, text_mask_f.data(), 0, text_mask_f.size() * sizeof(float));
    backend.tensor_set(feat_tensor, feat.data(), 0, feat.size() * sizeof(float));
    backend.tensor_set(feat_mask_tensor, feat_mask_f.data(), 0, feat_mask_f.size() * sizeof(float));

    backend.reset_transfer_stats();
    VoxCPMDecodeState mirrored_state =
        runtime.prefill_with_input_tensors(text_tensor, text_mask_tensor, feat_tensor, feat_mask_tensor, seq_len, 3);
    const BackendTransferStats mirrored_stats = backend.transfer_stats();

    backend.reset_transfer_stats();
    VoxCPMDecodeState explicit_state = runtime.prefill_with_input_tensors_and_prompt_positions(text_tensor,
                                                                                               text_mask_tensor,
                                                                                               feat_tensor,
                                                                                               feat_mask_tensor,
                                                                                               prompt_positions,
                                                                                               seq_len,
                                                                                               3);
    const BackendTransferStats explicit_stats = backend.transfer_stats();

    REQUIRE(explicit_state.current_position == mirrored_state.current_position);
    REQUIRE(explicit_state.audio_frame_count == mirrored_state.audio_frame_count);
    REQUIRE(explicit_state.lm_hidden == mirrored_state.lm_hidden);
    REQUIRE(explicit_state.residual_hidden == mirrored_state.residual_hidden);
    REQUIRE(explicit_state.prefix_feat_cond == mirrored_state.prefix_feat_cond);
    REQUIRE(explicit_state.output_pool->export_latent_seq_to_host(backend, explicit_state.audio_frame_count) ==
            mirrored_state.output_pool->export_latent_seq_to_host(backend, mirrored_state.audio_frame_count));
    REQUIRE(explicit_state.output_pool->export_patch_to_host(backend) ==
            mirrored_state.output_pool->export_patch_to_host(backend));

    REQUIRE(explicit_stats.host_to_device_bytes == mirrored_stats.host_to_device_bytes);
    REQUIRE(explicit_stats.device_to_host_bytes <= mirrored_stats.device_to_host_bytes);
    REQUIRE(mirrored_stats.device_to_host_bytes - explicit_stats.device_to_host_bytes >=
            feat_mask_f.size() * sizeof(float));

    backend.free_buffer(input_buffer);
}

TEST_CASE("prefill tensor-input module entry matches the explicit prompt-positions wrapper",
          "[runtime][prefill][module-entry]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    const int seq_len = 4;
    std::vector<int32_t> text = {2, 5, 7, 11};
    std::vector<float> text_mask_f = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> feat_mask_f = {0.0f, 1.0f, 1.0f, 0.0f};
    std::vector<int32_t> prompt_positions = {1, 2};

    const size_t feat_elem_count =
        static_cast<size_t>(seq_len * runtime.config().feat_dim * runtime.config().patch_size);
    std::vector<float> feat(feat_elem_count, 0.0f);
    for (size_t i = 0; i < feat.size(); ++i) {
        feat[i] = static_cast<float>((static_cast<int>(i % 29) - 14)) * 0.01f;
    }

    VoxCPMContext input_ctx(ContextType::Graph, 4, 4);
    ggml_tensor* text_tensor = input_ctx.new_tensor_1d(GGML_TYPE_I32, seq_len);
    ggml_tensor* text_mask_tensor = input_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    ggml_tensor* feat_tensor = input_ctx.new_tensor_3d(GGML_TYPE_F32,
                                                       runtime.config().feat_dim,
                                                       runtime.config().patch_size,
                                                       seq_len);
    ggml_tensor* feat_mask_tensor = input_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    REQUIRE(text_tensor != nullptr);
    REQUIRE(text_mask_tensor != nullptr);
    REQUIRE(feat_tensor != nullptr);
    REQUIRE(feat_mask_tensor != nullptr);
    ggml_backend_buffer_t input_buffer = backend.alloc_buffer(input_ctx.raw_context(), BufferUsage::Output);
    REQUIRE(input_buffer != nullptr);
    backend.tensor_set(text_tensor, text.data(), 0, text.size() * sizeof(int32_t));
    backend.tensor_set(text_mask_tensor, text_mask_f.data(), 0, text_mask_f.size() * sizeof(float));
    backend.tensor_set(feat_tensor, feat.data(), 0, feat.size() * sizeof(float));
    backend.tensor_set(feat_mask_tensor, feat_mask_f.data(), 0, feat_mask_f.size() * sizeof(float));

    VoxCPMDecodeState wrapper_state = runtime.prefill_with_input_tensors_and_prompt_positions(text_tensor,
                                                                                              text_mask_tensor,
                                                                                              feat_tensor,
                                                                                              feat_mask_tensor,
                                                                                              prompt_positions,
                                                                                              seq_len,
                                                                                              3);

    VoxCPMPrefillTensorInputs inputs;
    inputs.text_src = text_tensor;
    inputs.text_mask_src = text_mask_tensor;
    inputs.feat_src = feat_tensor;
    inputs.feat_mask_src = feat_mask_tensor;
    inputs.prompt_positions = prompt_positions;
    inputs.seq_len = seq_len;
    inputs.streaming_prefix_len = 3;

    VoxCPMDecodeState module_state = runtime.prefill_from_tensor_inputs(inputs);

    REQUIRE(module_state.current_position == wrapper_state.current_position);
    REQUIRE(module_state.audio_frame_count == wrapper_state.audio_frame_count);
    REQUIRE(module_state.lm_hidden == wrapper_state.lm_hidden);
    REQUIRE(module_state.residual_hidden == wrapper_state.residual_hidden);
    REQUIRE(module_state.prefix_feat_cond == wrapper_state.prefix_feat_cond);
    REQUIRE(module_state.output_pool->export_latent_seq_to_host(backend, module_state.audio_frame_count) ==
            wrapper_state.output_pool->export_latent_seq_to_host(backend, wrapper_state.audio_frame_count));
    REQUIRE(module_state.output_pool->export_patch_to_host(backend) ==
            wrapper_state.output_pool->export_patch_to_host(backend));

    backend.free_buffer(input_buffer);
}

TEST_CASE("prefill tensor-input module entry can derive prompt positions from feat_mask",
          "[runtime][prefill][module-entry][compat]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    const int seq_len = 4;
    std::vector<int32_t> text = {2, 5, 7, 11};
    std::vector<float> text_mask_f = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> feat_mask_f = {0.0f, 1.0f, 1.0f, 0.0f};

    const size_t feat_elem_count =
        static_cast<size_t>(seq_len * runtime.config().feat_dim * runtime.config().patch_size);
    std::vector<float> feat(feat_elem_count, 0.0f);
    for (size_t i = 0; i < feat.size(); ++i) {
        feat[i] = static_cast<float>((static_cast<int>(i % 17) - 8)) * 0.02f;
    }

    VoxCPMContext input_ctx(ContextType::Graph, 4, 4);
    ggml_tensor* text_tensor = input_ctx.new_tensor_1d(GGML_TYPE_I32, seq_len);
    ggml_tensor* text_mask_tensor = input_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    ggml_tensor* feat_tensor = input_ctx.new_tensor_3d(GGML_TYPE_F32,
                                                       runtime.config().feat_dim,
                                                       runtime.config().patch_size,
                                                       seq_len);
    ggml_tensor* feat_mask_tensor = input_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    REQUIRE(text_tensor != nullptr);
    REQUIRE(text_mask_tensor != nullptr);
    REQUIRE(feat_tensor != nullptr);
    REQUIRE(feat_mask_tensor != nullptr);
    ggml_backend_buffer_t input_buffer = backend.alloc_buffer(input_ctx.raw_context(), BufferUsage::Output);
    REQUIRE(input_buffer != nullptr);
    backend.tensor_set(text_tensor, text.data(), 0, text.size() * sizeof(int32_t));
    backend.tensor_set(text_mask_tensor, text_mask_f.data(), 0, text_mask_f.size() * sizeof(float));
    backend.tensor_set(feat_tensor, feat.data(), 0, feat.size() * sizeof(float));
    backend.tensor_set(feat_mask_tensor, feat_mask_f.data(), 0, feat_mask_f.size() * sizeof(float));

    VoxCPMDecodeState wrapper_state =
        runtime.prefill_with_input_tensors(text_tensor, text_mask_tensor, feat_tensor, feat_mask_tensor, seq_len, 3);

    VoxCPMPrefillTensorInputs inputs;
    inputs.text_src = text_tensor;
    inputs.text_mask_src = text_mask_tensor;
    inputs.feat_src = feat_tensor;
    inputs.feat_mask_src = feat_mask_tensor;
    inputs.seq_len = seq_len;
    inputs.streaming_prefix_len = 3;

    VoxCPMDecodeState module_state = runtime.prefill_from_tensor_inputs(inputs);

    REQUIRE(module_state.current_position == wrapper_state.current_position);
    REQUIRE(module_state.audio_frame_count == wrapper_state.audio_frame_count);
    REQUIRE(module_state.lm_hidden == wrapper_state.lm_hidden);
    REQUIRE(module_state.residual_hidden == wrapper_state.residual_hidden);
    REQUIRE(module_state.prefix_feat_cond == wrapper_state.prefix_feat_cond);
    REQUIRE(module_state.output_pool->export_latent_seq_to_host(backend, module_state.audio_frame_count) ==
            wrapper_state.output_pool->export_latent_seq_to_host(backend, wrapper_state.audio_frame_count));
    REQUIRE(module_state.output_pool->export_patch_to_host(backend) ==
            wrapper_state.output_pool->export_patch_to_host(backend));

    backend.free_buffer(input_buffer);
}

TEST_CASE("real CLI prompt sample prefill matches legacy hidden-state and stop-predictor path",
          "[runtime][prefill][cli][stop][diagnostic]") {
    const std::string model_path = voxcpm::test::get_model_path();
    const std::string prompt_audio_path =
        (std::filesystem::current_path() / "examples" / "tai_yi_xian_ren.wav").string();
    REQUIRE(std::filesystem::exists(model_path));
    REQUIRE(std::filesystem::exists(prompt_audio_path));

    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(model_path, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    AudioVAE audio_vae;
    REQUIRE(audio_vae.load_from_store(store));

    const RealPromptInputs prepared = make_real_prompt_inputs(runtime,
                                                              audio_vae,
                                                              backend,
                                                              *store,
                                                              prompt_audio_path,
                                                              "对，这就是我，万人敬仰的太乙真人。",
                                                              "大家好，我现在正在大可奇奇体验AI科技。");

    VoxCPMDecodeState current_state = runtime.prefill(prepared.full_text_tokens,
                                                     prepared.text_mask,
                                                     prepared.feat,
                                                     prepared.feat_mask,
                                                     prepared.seq_len,
                                                     3);

    const int seq_len = prepared.seq_len;
    const int hidden_size = runtime.base_lm().config().hidden_size;
    std::vector<float> text_mask_f(static_cast<size_t>(seq_len), 0.0f);
    std::vector<float> feat_mask_f(static_cast<size_t>(seq_len), 0.0f);
    for (int i = 0; i < seq_len; ++i) {
        text_mask_f[static_cast<size_t>(i)] = prepared.text_mask[static_cast<size_t>(i)] != 0 ? 1.0f : 0.0f;
        feat_mask_f[static_cast<size_t>(i)] = prepared.feat_mask[static_cast<size_t>(i)] != 0 ? 1.0f : 0.0f;
    }

    const std::vector<float> text_embed = runtime.benchmark_run_embedding(prepared.full_text_tokens);
    const std::vector<float> feat_embed =
        runtime.benchmark_run_locenc_sequence_to_lm_projection(prepared.feat, seq_len);
    REQUIRE(text_embed.size() == feat_embed.size());

    std::vector<float> combined_embed(text_embed.size(), 0.0f);
    for (int t = 0; t < seq_len; ++t) {
        const float text_scale = text_mask_f[static_cast<size_t>(t)];
        const float feat_scale = feat_mask_f[static_cast<size_t>(t)];
        for (int h = 0; h < hidden_size; ++h) {
            const size_t idx = static_cast<size_t>(t) * static_cast<size_t>(hidden_size) + static_cast<size_t>(h);
            combined_embed[idx] = text_scale * text_embed[idx] + feat_scale * feat_embed[idx];
        }
    }

    MiniCPMKVCache base_cache(runtime.base_lm().config().n_layer,
                              runtime.base_lm().config().n_kv_heads,
                              runtime.config().max_length,
                              runtime.base_lm().config().head_dim());
    MiniCPMKVCache residual_cache(runtime.residual_lm().config().n_layer,
                                  runtime.residual_lm().config().n_kv_heads,
                                  runtime.config().max_length,
                                  runtime.residual_lm().config().head_dim());
    base_cache.init(backend);
    residual_cache.init(backend);

    const std::vector<float> base_output =
        runtime.benchmark_run_base_lm_forward(combined_embed, seq_len, base_cache, true);
    const std::vector<float> blended_output =
        runtime.benchmark_run_masked_fsq_blend(base_output, text_mask_f, feat_mask_f, seq_len);
    const std::vector<float> legacy_lm_hidden =
        slice_last_column_major_2d(blended_output, hidden_size, seq_len - 1);

    std::vector<float> residual_inputs =
        build_expected_residual_inputs(runtime, backend, combined_embed, blended_output, feat_mask_f, seq_len);
    const std::vector<float> legacy_residual_hidden =
        runtime.benchmark_run_residual_lm_forward_last_hidden(residual_inputs, seq_len, residual_cache, true);

    const std::vector<float> current_stop = runtime.benchmark_run_stop_predictor(current_state.lm_hidden);
    const std::vector<float> legacy_stop = runtime.benchmark_run_stop_predictor(legacy_lm_hidden);

    REQUIRE(current_state.lm_hidden.size() == legacy_lm_hidden.size());
    for (size_t i = 0; i < legacy_lm_hidden.size(); ++i) {
        REQUIRE(current_state.lm_hidden[i] == Catch::Approx(legacy_lm_hidden[i]));
    }

    REQUIRE(current_state.residual_hidden.size() == legacy_residual_hidden.size());
    for (size_t i = 0; i < legacy_residual_hidden.size(); ++i) {
        REQUIRE(current_state.residual_hidden[i] == Catch::Approx(legacy_residual_hidden[i]));
    }

    REQUIRE(current_stop.size() == legacy_stop.size());
    for (size_t i = 0; i < legacy_stop.size(); ++i) {
        REQUIRE(current_stop[i] == Catch::Approx(legacy_stop[i]));
    }

    REQUIRE(current_state.persistent_state != nullptr);
    std::vector<float> persistent_lm(current_state.lm_hidden.size(), 0.0f);
    std::vector<float> persistent_residual(current_state.residual_hidden.size(), 0.0f);
    std::vector<float> persistent_prefix(current_state.prefix_feat_cond.size(), 0.0f);
    REQUIRE(current_state.persistent_state->get_lm_hidden_to_host(backend,
                                                                  persistent_lm.data(),
                                                                  persistent_lm.size()));
    REQUIRE(current_state.persistent_state->get_residual_hidden_to_host(backend,
                                                                        persistent_residual.data(),
                                                                        persistent_residual.size()));
    REQUIRE(current_state.persistent_state->get_prefix_patch_to_host(backend,
                                                                     persistent_prefix.data(),
                                                                     persistent_prefix.size()));
    for (size_t i = 0; i < current_state.lm_hidden.size(); ++i) {
        REQUIRE(persistent_lm[i] == Catch::Approx(current_state.lm_hidden[i]).margin(1e-5));
    }
    for (size_t i = 0; i < current_state.residual_hidden.size(); ++i) {
        REQUIRE(persistent_residual[i] == Catch::Approx(current_state.residual_hidden[i]).margin(1e-5));
    }
    for (size_t i = 0; i < current_state.prefix_feat_cond.size(); ++i) {
        REQUIRE(persistent_prefix[i] == Catch::Approx(current_state.prefix_feat_cond[i]).margin(1e-5));
    }
}

TEST_CASE("real CLI prompt sample decode matches decomposed single-step path",
          "[.][runtime][decode][cli][stop][diagnostic]") {
    const std::string model_path = voxcpm::test::get_model_path();
    const std::string prompt_audio_path =
        (std::filesystem::current_path() / "examples" / "tai_yi_xian_ren.wav").string();
    REQUIRE(std::filesystem::exists(model_path));
    REQUIRE(std::filesystem::exists(prompt_audio_path));

    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(model_path, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    AudioVAE audio_vae;
    REQUIRE(audio_vae.load_from_store(store));

    const RealPromptInputs prepared = make_real_prompt_inputs(runtime,
                                                              audio_vae,
                                                              backend,
                                                              *store,
                                                              prompt_audio_path,
                                                              "对，这就是我，万人敬仰的太乙真人。",
                                                              "大家好，我现在正在大可奇奇体验AI科技。");

    VoxCPMDecodeState state = runtime.prefill(prepared.full_text_tokens,
                                             prepared.text_mask,
                                             prepared.feat,
                                             prepared.feat_mask,
                                             prepared.seq_len,
                                             3);
    VoxCPMDecodeState cloned_state = runtime.benchmark_clone_state(state);
    VoxCPMDecodeState manual_state = runtime.benchmark_clone_state(state);

    const std::vector<float> z =
        make_deterministic_noise_patch(runtime.config().feat_dim, runtime.config().patch_size, 0);

    const auto front_half_pair = runtime.benchmark_run_decode_front_half_with_curr_embed(z,
                                                                                          manual_state.lm_hidden,
                                                                                          manual_state.residual_hidden,
                                                                                          manual_state.prefix_feat_cond,
                                                                                          10,
                                                                                          2.0f);
    const auto state_front_half = runtime.benchmark_run_decode_front_half_from_state(z, manual_state, 10, 2.0f);
    REQUIRE(state_front_half.first.size() == front_half_pair.first.size());
    for (size_t i = 0; i < state_front_half.first.size(); ++i) {
        REQUIRE(state_front_half.first[i] == Catch::Approx(front_half_pair.first[i]));
    }
    REQUIRE(state_front_half.second.size() == front_half_pair.second.size());
    for (size_t i = 0; i < state_front_half.second.size(); ++i) {
        REQUIRE(state_front_half.second[i] == Catch::Approx(front_half_pair.second[i]));
    }
    const std::vector<float>& patch = front_half_pair.first;
    const std::vector<float>& curr_embed = front_half_pair.second;
    const std::vector<float> stop_logits = runtime.benchmark_run_stop_predictor(manual_state.lm_hidden);
    const std::vector<float> next_lm_hidden =
        runtime.benchmark_run_base_lm_decode_step(curr_embed, manual_state.current_position, *manual_state.base_lm_cache);

    const std::vector<float> residual_input =
        apply_residual_bridge(runtime, backend, next_lm_hidden, curr_embed, 1);
    const std::vector<float> next_residual_hidden =
        runtime.benchmark_run_residual_lm_decode_step(residual_input,
                                                      manual_state.current_position,
                                                      *manual_state.residual_lm_cache,
                                                      true);

    VoxCPMDecodeResult current = runtime.decode(std::move(state), z, 10, 2.0f);
    VoxCPMDecodeResult cloned = runtime.decode(std::move(cloned_state), z, 10, 2.0f);

    REQUIRE(current.output_0.size() == cloned.output_0.size());
    for (size_t i = 0; i < current.output_0.size(); ++i) {
        REQUIRE(current.output_0[i] == Catch::Approx(cloned.output_0[i]));
    }
    REQUIRE(current.output_2 == cloned.output_2);
    REQUIRE(current.output_1.lm_hidden.size() == cloned.output_1.lm_hidden.size());
    for (size_t i = 0; i < current.output_1.lm_hidden.size(); ++i) {
        REQUIRE(current.output_1.lm_hidden[i] == Catch::Approx(cloned.output_1.lm_hidden[i]));
    }
    REQUIRE(current.output_1.residual_hidden.size() == cloned.output_1.residual_hidden.size());
    for (size_t i = 0; i < current.output_1.residual_hidden.size(); ++i) {
        REQUIRE(current.output_1.residual_hidden[i] == Catch::Approx(cloned.output_1.residual_hidden[i]));
    }

    REQUIRE(current.output_0.size() == patch.size());
    for (size_t i = 0; i < patch.size(); ++i) {
        REQUIRE(current.output_0[i] == Catch::Approx(patch[i]));
    }

    REQUIRE(current.output_2 == (stop_logits[1] > stop_logits[0]));
    REQUIRE(current.output_1.lm_hidden.size() == next_lm_hidden.size());
    for (size_t i = 0; i < next_lm_hidden.size(); ++i) {
        REQUIRE(current.output_1.lm_hidden[i] == Catch::Approx(next_lm_hidden[i]));
    }

    REQUIRE(current.output_1.residual_hidden.size() == next_residual_hidden.size());
    for (size_t i = 0; i < next_residual_hidden.size(); ++i) {
        REQUIRE(current.output_1.residual_hidden[i] == Catch::Approx(next_residual_hidden[i]));
    }
}

TEST_CASE("real CLI reference+prompt sample prefill matches legacy hidden-state and stop-predictor path",
          "[runtime][prefill][cli][reference][stop][diagnostic]") {
    const std::string model_path = voxcpm::test::get_model_path();
    const std::string prompt_audio_path =
        (std::filesystem::current_path() / "examples" / "tai_yi_xian_ren.wav").string();
    REQUIRE(std::filesystem::exists(model_path));
    REQUIRE(std::filesystem::exists(prompt_audio_path));

    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(model_path, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    AudioVAE audio_vae;
    REQUIRE(audio_vae.load_from_store(store));

    const RealPromptInputs prepared = make_real_reference_prompt_inputs(runtime,
                                                                        audio_vae,
                                                                        backend,
                                                                        *store,
                                                                        prompt_audio_path,
                                                                        prompt_audio_path,
                                                                        "对，这就是我，万人敬仰的太乙真人。",
                                                                        "大家好，我现在正在大可奇奇体验AI科技。");

    VoxCPMDecodeState current_state = runtime.prefill(prepared.full_text_tokens,
                                                     prepared.text_mask,
                                                     prepared.feat,
                                                     prepared.feat_mask,
                                                     prepared.seq_len,
                                                     4);

    const int seq_len = prepared.seq_len;
    const int hidden_size = runtime.base_lm().config().hidden_size;
    std::vector<float> text_mask_f(static_cast<size_t>(seq_len), 0.0f);
    std::vector<float> feat_mask_f(static_cast<size_t>(seq_len), 0.0f);
    for (int i = 0; i < seq_len; ++i) {
        text_mask_f[static_cast<size_t>(i)] = prepared.text_mask[static_cast<size_t>(i)] != 0 ? 1.0f : 0.0f;
        feat_mask_f[static_cast<size_t>(i)] = prepared.feat_mask[static_cast<size_t>(i)] != 0 ? 1.0f : 0.0f;
    }

    const std::vector<float> text_embed = runtime.benchmark_run_embedding(prepared.full_text_tokens);
    const std::vector<float> feat_embed =
        runtime.benchmark_run_locenc_sequence_to_lm_projection(prepared.feat, seq_len);
    REQUIRE(text_embed.size() == feat_embed.size());

    std::vector<float> combined_embed(text_embed.size(), 0.0f);
    for (int t = 0; t < seq_len; ++t) {
        const float text_scale = text_mask_f[static_cast<size_t>(t)];
        const float feat_scale = feat_mask_f[static_cast<size_t>(t)];
        for (int h = 0; h < hidden_size; ++h) {
            const size_t idx = static_cast<size_t>(t) * static_cast<size_t>(hidden_size) + static_cast<size_t>(h);
            combined_embed[idx] = text_scale * text_embed[idx] + feat_scale * feat_embed[idx];
        }
    }

    MiniCPMKVCache base_cache(runtime.base_lm().config().n_layer,
                              runtime.base_lm().config().n_kv_heads,
                              runtime.config().max_length,
                              runtime.base_lm().config().head_dim());
    MiniCPMKVCache residual_cache(runtime.residual_lm().config().n_layer,
                                  runtime.residual_lm().config().n_kv_heads,
                                  runtime.config().max_length,
                                  runtime.residual_lm().config().head_dim());
    base_cache.init(backend);
    residual_cache.init(backend);

    const std::vector<float> base_output =
        runtime.benchmark_run_base_lm_forward(combined_embed, seq_len, base_cache, true);
    const std::vector<float> blended_output =
        runtime.benchmark_run_masked_fsq_blend(base_output, text_mask_f, feat_mask_f, seq_len);
    const std::vector<float> legacy_lm_hidden =
        slice_last_column_major_2d(blended_output, hidden_size, seq_len - 1);

    std::vector<float> residual_inputs =
        build_expected_residual_inputs(runtime, backend, combined_embed, blended_output, feat_mask_f, seq_len);
    const std::vector<float> legacy_residual_hidden =
        runtime.benchmark_run_residual_lm_forward_last_hidden(residual_inputs, seq_len, residual_cache, true);

    const std::vector<float> current_stop = runtime.benchmark_run_stop_predictor(current_state.lm_hidden);
    const std::vector<float> legacy_stop = runtime.benchmark_run_stop_predictor(legacy_lm_hidden);

    REQUIRE(current_state.lm_hidden.size() == legacy_lm_hidden.size());
    for (size_t i = 0; i < legacy_lm_hidden.size(); ++i) {
        REQUIRE(current_state.lm_hidden[i] == Catch::Approx(legacy_lm_hidden[i]));
    }

    REQUIRE(current_state.residual_hidden.size() == legacy_residual_hidden.size());
    for (size_t i = 0; i < legacy_residual_hidden.size(); ++i) {
        REQUIRE(current_state.residual_hidden[i] == Catch::Approx(legacy_residual_hidden[i]));
    }

    REQUIRE(current_stop.size() == legacy_stop.size());
    for (size_t i = 0; i < legacy_stop.size(); ++i) {
        REQUIRE(current_stop[i] == Catch::Approx(legacy_stop[i]));
    }
}

TEST_CASE("real CLI reference+prompt sample decode matches decomposed single-step path",
          "[.][runtime][decode][cli][reference][diagnostic]") {
    const std::string model_path = voxcpm::test::get_model_path();
    const std::string prompt_audio_path =
        (std::filesystem::current_path() / "examples" / "tai_yi_xian_ren.wav").string();
    REQUIRE(std::filesystem::exists(model_path));
    REQUIRE(std::filesystem::exists(prompt_audio_path));

    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(model_path, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    AudioVAE audio_vae;
    REQUIRE(audio_vae.load_from_store(store));

    const RealPromptInputs prepared = make_real_reference_prompt_inputs(runtime,
                                                                        audio_vae,
                                                                        backend,
                                                                        *store,
                                                                        prompt_audio_path,
                                                                        prompt_audio_path,
                                                                        "对，这就是我，万人敬仰的太乙真人。",
                                                                        "大家好，我现在正在大可奇奇体验AI科技。");

    VoxCPMDecodeState state = runtime.prefill(prepared.full_text_tokens,
                                             prepared.text_mask,
                                             prepared.feat,
                                             prepared.feat_mask,
                                             prepared.seq_len,
                                             4);
    VoxCPMDecodeState cloned_state = runtime.benchmark_clone_state(state);
    VoxCPMDecodeState manual_state = runtime.benchmark_clone_state(state);

    const std::vector<float> z =
        make_deterministic_noise_patch(runtime.config().feat_dim, runtime.config().patch_size, 0);

    const auto front_half_pair = runtime.benchmark_run_decode_front_half_with_curr_embed(z,
                                                                                          manual_state.lm_hidden,
                                                                                          manual_state.residual_hidden,
                                                                                          manual_state.prefix_feat_cond,
                                                                                          10,
                                                                                          2.0f);
    const auto state_front_half = runtime.benchmark_run_decode_front_half_from_state(z, manual_state, 10, 2.0f);
    REQUIRE(state_front_half.first.size() == front_half_pair.first.size());
    for (size_t i = 0; i < state_front_half.first.size(); ++i) {
        REQUIRE(state_front_half.first[i] == Catch::Approx(front_half_pair.first[i]));
    }
    REQUIRE(state_front_half.second.size() == front_half_pair.second.size());
    for (size_t i = 0; i < state_front_half.second.size(); ++i) {
        REQUIRE(state_front_half.second[i] == Catch::Approx(front_half_pair.second[i]));
    }
    const std::vector<float>& patch = front_half_pair.first;
    const std::vector<float>& curr_embed = front_half_pair.second;
    const std::vector<float> stop_logits = runtime.benchmark_run_stop_predictor(manual_state.lm_hidden);
    const std::vector<float> next_lm_hidden =
        runtime.benchmark_run_base_lm_decode_step(curr_embed, manual_state.current_position, *manual_state.base_lm_cache);

    const std::vector<float> residual_input =
        apply_residual_bridge(runtime, backend, next_lm_hidden, curr_embed, 1);
    const std::vector<float> next_residual_hidden =
        runtime.benchmark_run_residual_lm_decode_step(residual_input,
                                                      manual_state.current_position,
                                                      *manual_state.residual_lm_cache,
                                                      true);

    VoxCPMDecodeResult current = runtime.decode(std::move(state), z, 10, 2.0f);
    VoxCPMDecodeResult cloned = runtime.decode(std::move(cloned_state), z, 10, 2.0f);

    REQUIRE(current.output_0.size() == cloned.output_0.size());
    for (size_t i = 0; i < current.output_0.size(); ++i) {
        REQUIRE(current.output_0[i] == Catch::Approx(cloned.output_0[i]));
    }
    REQUIRE(current.output_2 == cloned.output_2);
    REQUIRE(current.output_1.lm_hidden.size() == cloned.output_1.lm_hidden.size());
    for (size_t i = 0; i < current.output_1.lm_hidden.size(); ++i) {
        REQUIRE(current.output_1.lm_hidden[i] == Catch::Approx(cloned.output_1.lm_hidden[i]));
    }
    REQUIRE(current.output_1.residual_hidden.size() == cloned.output_1.residual_hidden.size());
    for (size_t i = 0; i < current.output_1.residual_hidden.size(); ++i) {
        REQUIRE(current.output_1.residual_hidden[i] == Catch::Approx(cloned.output_1.residual_hidden[i]));
    }

    REQUIRE(current.output_0.size() == patch.size());
    for (size_t i = 0; i < patch.size(); ++i) {
        REQUIRE(current.output_0[i] == Catch::Approx(patch[i]));
    }

    REQUIRE(current.output_2 == (stop_logits[1] > stop_logits[0]));
    REQUIRE(current.output_1.lm_hidden.size() == next_lm_hidden.size());
    for (size_t i = 0; i < next_lm_hidden.size(); ++i) {
        REQUIRE(current.output_1.lm_hidden[i] == Catch::Approx(next_lm_hidden[i]));
    }

    REQUIRE(current.output_1.residual_hidden.size() == next_residual_hidden.size());
    for (size_t i = 0; i < next_residual_hidden.size(); ++i) {
        REQUIRE(current.output_1.residual_hidden[i] == Catch::Approx(next_residual_hidden[i]));
    }
}

TEST_CASE("real CLI reference+prompt sample decode remains aligned across multiple steps",
          "[.][runtime][decode][cli][reference][multistep][diagnostic]") {
    const std::string model_path = voxcpm::test::get_model_path();
    const std::string prompt_audio_path =
        (std::filesystem::current_path() / "examples" / "tai_yi_xian_ren.wav").string();
    REQUIRE(std::filesystem::exists(model_path));
    REQUIRE(std::filesystem::exists(prompt_audio_path));

    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(model_path, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    AudioVAE audio_vae;
    REQUIRE(audio_vae.load_from_store(store));

    const RealPromptInputs prepared = make_real_reference_prompt_inputs(runtime,
                                                                        audio_vae,
                                                                        backend,
                                                                        *store,
                                                                        prompt_audio_path,
                                                                        prompt_audio_path,
                                                                        "对，这就是我，万人敬仰的太乙真人。",
                                                                        "大家好，我现在正在大可奇奇体验AI科技。");

    VoxCPMDecodeState state = runtime.prefill(prepared.full_text_tokens,
                                             prepared.text_mask,
                                             prepared.feat,
                                             prepared.feat_mask,
                                             prepared.seq_len,
                                             4);
    VoxCPMDecodeState cloned_state = runtime.benchmark_clone_state(state);
    VoxCPMDecodeState manual_state = runtime.benchmark_clone_state(state);

    constexpr int kSteps = 12;
    for (int step = 0; step < kSteps; ++step) {
        INFO("step=" << step);
        const std::vector<float> z =
            make_deterministic_noise_patch(runtime.config().feat_dim, runtime.config().patch_size, step);

        const auto front_half_pair = runtime.benchmark_run_decode_front_half_with_curr_embed(z,
                                                                                              manual_state.lm_hidden,
                                                                                              manual_state.residual_hidden,
                                                                                              manual_state.prefix_feat_cond,
                                                                                              10,
                                                                                              2.0f);
        const std::vector<float>& patch = front_half_pair.first;
        const std::vector<float>& curr_embed = front_half_pair.second;
        const std::vector<float> stop_logits = runtime.benchmark_run_stop_predictor(manual_state.lm_hidden);
        const std::vector<float> next_lm_hidden =
            runtime.benchmark_run_base_lm_decode_step(curr_embed,
                                                      manual_state.current_position,
                                                      *manual_state.base_lm_cache);
        const std::vector<float> residual_input =
            apply_residual_bridge(runtime, backend, next_lm_hidden, curr_embed, 1);
        const std::vector<float> next_residual_hidden =
            runtime.benchmark_run_residual_lm_decode_step(residual_input,
                                                          manual_state.current_position,
                                                          *manual_state.residual_lm_cache,
                                                          true);

        VoxCPMDecodeResult current = runtime.decode(std::move(state), z, 10, 2.0f);
        VoxCPMDecodeResult cloned = runtime.decode(std::move(cloned_state), z, 10, 2.0f);

        REQUIRE(current.output_0.size() == patch.size());
        REQUIRE(cloned.output_0.size() == patch.size());
        for (size_t i = 0; i < patch.size(); ++i) {
            REQUIRE(current.output_0[i] == Catch::Approx(patch[i]));
            REQUIRE(cloned.output_0[i] == Catch::Approx(patch[i]));
        }

        REQUIRE(current.output_2 == (stop_logits[1] > stop_logits[0]));
        REQUIRE(cloned.output_2 == current.output_2);

        REQUIRE(current.output_1.lm_hidden.size() == next_lm_hidden.size());
        REQUIRE(cloned.output_1.lm_hidden.size() == next_lm_hidden.size());
        for (size_t i = 0; i < next_lm_hidden.size(); ++i) {
            REQUIRE(current.output_1.lm_hidden[i] == Catch::Approx(next_lm_hidden[i]));
            REQUIRE(cloned.output_1.lm_hidden[i] == Catch::Approx(next_lm_hidden[i]));
        }

        REQUIRE(current.output_1.residual_hidden.size() == next_residual_hidden.size());
        REQUIRE(cloned.output_1.residual_hidden.size() == next_residual_hidden.size());
        for (size_t i = 0; i < next_residual_hidden.size(); ++i) {
            REQUIRE(current.output_1.residual_hidden[i] == Catch::Approx(next_residual_hidden[i]));
            REQUIRE(cloned.output_1.residual_hidden[i] == Catch::Approx(next_residual_hidden[i]));
        }

        manual_state.lm_hidden = next_lm_hidden;
        manual_state.residual_hidden = next_residual_hidden;
        manual_state.prefix_feat_cond = patch;
        manual_state.current_position += 1;
        manual_state.audio_frame_count += 1;

        state = std::move(current.output_1);
        cloned_state = std::move(cloned.output_1);
    }
}

TEST_CASE("reference+prompt latent assembly matches output pool timeline",
          "[runtime][decode][cli][reference][latent][diagnostic]") {
    const std::string model_path = voxcpm::test::get_model_path();
    const std::string prompt_audio_path =
        (std::filesystem::current_path() / "examples" / "tai_yi_xian_ren.wav").string();
    REQUIRE(std::filesystem::exists(model_path));
    REQUIRE(std::filesystem::exists(prompt_audio_path));

    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(model_path, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    AudioVAE audio_vae;
    REQUIRE(audio_vae.load_from_store(store));

    const RealPromptInputs prepared = make_real_reference_prompt_inputs(runtime,
                                                                        audio_vae,
                                                                        backend,
                                                                        *store,
                                                                        prompt_audio_path,
                                                                        prompt_audio_path,
                                                                        "对，这就是我，万人敬仰的太乙真人。",
                                                                        "大家好，我现在正在大可奇奇体验AI科技。");

    const int patch_size = runtime.config().patch_size;
    const int feat_dim = runtime.config().feat_dim;
    const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;

    const std::vector<float> prompt_feat =
        encode_test_audio_features(runtime, audio_vae, backend, prompt_audio_path, TestPaddingMode::Left);
    const int prompt_audio_length = static_cast<int>(prompt_feat.size() / frame_stride);

    VoxCPMDecodeState state = runtime.prefill(prepared.full_text_tokens,
                                             prepared.text_mask,
                                             prepared.feat,
                                             prepared.feat_mask,
                                             prepared.seq_len,
                                             4);

    std::vector<float> generated_steps;
    constexpr int kSteps = 8;
    for (int step = 0; step < kSteps; ++step) {
        const std::vector<float> z =
            make_deterministic_noise_patch(runtime.config().feat_dim, runtime.config().patch_size, step);
        VoxCPMDecodeResult current = runtime.decode(std::move(state), z, 10, 2.0f);
        generated_steps.insert(generated_steps.end(), current.output_0.begin(), current.output_0.end());
        state = std::move(current.output_1);
    }

    REQUIRE(state.output_pool != nullptr);
    REQUIRE(state.output_pool->is_initialized());

    const int prepended_context_frames = std::min(4 - 1, prompt_audio_length);
    const int total_frames = prepended_context_frames + kSteps;

    const std::vector<float> assembled_frames =
        build_decode_frames_for_test(prompt_feat, prompt_audio_length, generated_steps, 4, patch_size, feat_dim);
    REQUIRE(static_cast<int>(assembled_frames.size() / frame_stride) == total_frames);
    const std::vector<float> assembled_latent =
        patch_major_frames_to_latent_for_test(assembled_frames, patch_size, feat_dim);
    const std::vector<float> output_pool_latent =
        state.output_pool->export_audio_vae_latent_to_host(backend,
                                                           prompt_audio_length - prepended_context_frames,
                                                           total_frames);

    REQUIRE(output_pool_latent.size() == assembled_latent.size());
    for (size_t i = 0; i < assembled_latent.size(); ++i) {
        REQUIRE(output_pool_latent[i] == Catch::Approx(assembled_latent[i]));
    }
}

TEST_CASE("residual_lm last-hidden path avoids full sequence d2h in prefill-style usage",
          "[runtime][prefill][residual][transfer]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    const int seq_len = 3;
    const int hidden_size = runtime.residual_lm().config().hidden_size;
    std::vector<float> input(static_cast<size_t>(seq_len * hidden_size), 0.0f);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>((static_cast<int>(i % 17) - 8)) * 0.03125f;
    }

    MiniCPMKVCache full_cache(runtime.residual_lm().config().n_layer,
                              runtime.residual_lm().config().n_kv_heads,
                              runtime.config().max_length,
                              runtime.residual_lm().config().head_dim());
    MiniCPMKVCache last_hidden_cache(runtime.residual_lm().config().n_layer,
                                     runtime.residual_lm().config().n_kv_heads,
                                     runtime.config().max_length,
                                     runtime.residual_lm().config().head_dim());
    full_cache.init(backend);
    last_hidden_cache.init(backend);

    backend.reset_transfer_stats();
    const std::vector<float> full_output = runtime.benchmark_run_residual_lm_forward(input, seq_len, full_cache, true);
    const BackendTransferStats full_stats = backend.transfer_stats();

    backend.reset_transfer_stats();
    const std::vector<float> last_hidden =
        runtime.benchmark_run_residual_lm_forward_last_hidden(input, seq_len, last_hidden_cache, true);
    const BackendTransferStats last_hidden_stats = backend.transfer_stats();

    REQUIRE(last_hidden == slice_last_column_major_2d(full_output, hidden_size, seq_len - 1));
    REQUIRE(last_hidden_stats.device_to_host_bytes < full_stats.device_to_host_bytes);
    REQUIRE(full_stats.device_to_host_bytes - last_hidden_stats.device_to_host_bytes >=
            static_cast<size_t>(hidden_size * (seq_len - 1)) * sizeof(float));
}

TEST_CASE("prefill no longer pays extra eager bootstrap h2d before lazy state handoff",
          "[runtime][prefill][lazy][transfer]") {
    auto store = std::make_shared<VoxCPMWeightStore>();
    VoxCPMBackend backend(BackendType::CPU, 2);
    REQUIRE(store->load_from_file(VOXCPM_DEFAULT_MODEL_PATH, backend));

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_store(store, backend));

    const int seq_len = 2;
    std::vector<int32_t> text(seq_len, 1);
    std::vector<int32_t> text_mask(seq_len, 1);
    std::vector<int32_t> feat_mask(seq_len, 1);
    std::vector<float> feat(static_cast<size_t>(seq_len * runtime.config().patch_size * runtime.config().feat_dim), 0.05f);

    ScopedEnvVar lazy_guard("VOXCPM_LAZY_HOST_STATE");

    lazy_guard.set("0");
    backend.reset_transfer_stats();
    VoxCPMDecodeState eager = runtime.prefill(text, text_mask, feat, feat_mask, seq_len, 3);
    const BackendTransferStats eager_stats = backend.transfer_stats();

    lazy_guard.set("1");
    backend.reset_transfer_stats();
    VoxCPMDecodeState lazy = runtime.prefill(text, text_mask, feat, feat_mask, seq_len, 3);
    const BackendTransferStats lazy_stats = backend.transfer_stats();
    const size_t lm_hidden_bytes = static_cast<size_t>(runtime.base_lm().config().hidden_size) * sizeof(float);
    const size_t residual_hidden_bytes = static_cast<size_t>(runtime.residual_lm().config().hidden_size) * sizeof(float);

    REQUIRE(eager.current_position == lazy.current_position);
    REQUIRE(eager.audio_frame_count == lazy.audio_frame_count);
    REQUIRE(lazy.lm_hidden.empty());
    REQUIRE(lazy.residual_hidden.empty());
    REQUIRE(lazy.prefix_feat_cond.empty());
    REQUIRE(eager_stats.host_to_device_bytes == lazy_stats.host_to_device_bytes);
    REQUIRE(lazy_stats.device_to_host_bytes < eager_stats.device_to_host_bytes);
    REQUIRE(eager_stats.device_to_host_bytes - lazy_stats.device_to_host_bytes >=
            lm_hidden_bytes + residual_hidden_bytes);
}

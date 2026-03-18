#include "voxcpm/server_common.h"

#include "voxcpm/audio_io.h"
#include "voxcpm/context.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <stdexcept>

namespace voxcpm {

namespace {

using json = nlohmann::json;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

std::filesystem::path manifest_path_for(const std::string& root, const std::string& id) {
    return std::filesystem::path(root) / id / "manifest.json";
}

std::filesystem::path prompt_path_for(const std::string& root, const std::string& id) {
    return std::filesystem::path(root) / id / "prompt_feat.bin";
}

void write_binary_file(const std::filesystem::path& path, const std::vector<float>& values) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        fail("Failed to open file for writing: " + path.string());
    }
    out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
}

std::vector<float> read_binary_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
        fail("Failed to open file for reading: " + path.string());
    }
    const std::streamsize size = in.tellg();
    if (size < 0 || (size % static_cast<std::streamsize>(sizeof(float))) != 0) {
        fail("Invalid prompt feature blob: " + path.string());
    }
    in.seekg(0, std::ios::beg);
    std::vector<float> values(static_cast<size_t>(size) / sizeof(float), 0.0f);
    in.read(reinterpret_cast<char*>(values.data()), size);
    return values;
}

std::vector<float> extract_prompt_features(AudioVAE& audio_vae,
                                           VoxCPMBackend& backend,
                                           std::vector<float> audio,
                                           int sample_rate,
                                           int patch_size,
                                           int feat_dim) {
    VoxCPMContext graph_ctx(ContextType::Graph, 32768, 262144);
    ggml_tensor* latent = audio_vae.encode(graph_ctx, backend, audio, sample_rate);
    if (!latent) {
        fail("Failed to build AudioVAE encode graph");
    }

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, latent);
    backend.reserve_compute_memory(graph, "server.audio_vae.encode");
    backend.alloc_graph(graph, "server.audio_vae.encode");
    const auto& preprocessed = audio_vae.last_preprocessed_audio();
    backend.tensor_set(audio_vae.last_input_tensor(), preprocessed.data(), 0, preprocessed.size() * sizeof(float));
    if (backend.compute(graph) != GGML_STATUS_SUCCESS) {
        fail("AudioVAE encode failed");
    }

    const int total_patches = static_cast<int>(latent->ne[0]);
    const int latent_dim = static_cast<int>(latent->ne[1]);
    if (latent_dim != feat_dim) {
        fail("Prompt latent dim mismatch");
    }
    if (total_patches % patch_size != 0) {
        fail("Prompt latent patches are not divisible by patch size");
    }

    std::vector<float> encoded(static_cast<size_t>(total_patches) * latent_dim);
    backend.tensor_get(latent, encoded.data(), 0, encoded.size() * sizeof(float));

    const int audio_length = total_patches / patch_size;
    std::vector<float> features(static_cast<size_t>(audio_length) * patch_size * feat_dim, 0.0f);
    for (int t = 0; t < audio_length; ++t) {
        for (int p = 0; p < patch_size; ++p) {
            const int patch_index = t * patch_size + p;
            for (int d = 0; d < feat_dim; ++d) {
                const size_t src = static_cast<size_t>(d) * total_patches + patch_index;
                const size_t dst = (static_cast<size_t>(t) * patch_size + p) * feat_dim + d;
                features[dst] = encoded[src];
            }
        }
    }
    return features;
}

std::vector<float> decode_audio(AudioVAE& audio_vae,
                                VoxCPMBackend& backend,
                                const std::vector<float>& features,
                                int total_patches,
                                int feat_dim) {
    VoxCPMContext graph_ctx(ContextType::Graph, 32768, 262144);
    ggml_tensor* latent = graph_ctx.new_tensor_2d(GGML_TYPE_F32, total_patches, feat_dim);
    ggml_set_input(latent);
    ggml_tensor* audio = audio_vae.decode(graph_ctx, backend, latent);
    if (!audio) {
        fail("Failed to build AudioVAE decode graph");
    }

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, audio);
    backend.reserve_compute_memory(graph, "server.audio_vae.decode");
    backend.alloc_graph(graph, "server.audio_vae.decode");
    backend.tensor_set(latent, features.data(), 0, features.size() * sizeof(float));
    if (backend.compute(graph) != GGML_STATUS_SUCCESS) {
        fail("AudioVAE decode failed");
    }

    std::vector<float> waveform(static_cast<size_t>(ggml_nelements(audio)));
    backend.tensor_get(audio, waveform.data(), 0, waveform.size() * sizeof(float));
    return waveform;
}

void fill_noise(std::vector<float>& noise, int patch_size, int feat_dim, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    noise.resize(static_cast<size_t>(patch_size) * feat_dim);
    for (float& value : noise) {
        value = dist(rng);
    }
}

std::vector<float> build_decode_feature_sequence(const std::vector<float>& prompt_feat,
                                                 int prompt_audio_length,
                                                 const std::vector<float>& generated_steps,
                                                 int streaming_prefix_len,
                                                 int patch_size,
                                                 int feat_dim,
                                                 int* prepended_context_frames) {
    const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;
    int context_frames = 0;
    if (!prompt_feat.empty() && prompt_audio_length > 0 && streaming_prefix_len > 1) {
        context_frames = std::min(streaming_prefix_len - 1, prompt_audio_length);
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
    if (prepended_context_frames != nullptr) {
        *prepended_context_frames = context_frames;
    }
    return decode_frames;
}

void patch_major_to_latent(const std::vector<float>& frames,
                           int patch_size,
                           int feat_dim,
                           std::vector<float>& latent) {
    const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;
    const int total_frames = static_cast<int>(frames.size() / frame_stride);
    const int total_patches = total_frames * patch_size;
    latent.assign(static_cast<size_t>(total_patches) * feat_dim, 0.0f);
    for (int frame = 0; frame < total_frames; ++frame) {
        for (int patch = 0; patch < patch_size; ++patch) {
            const int time_index = frame * patch_size + patch;
            for (int d = 0; d < feat_dim; ++d) {
                const size_t src = (static_cast<size_t>(frame) * patch_size + patch) * feat_dim + d;
                const size_t dst = static_cast<size_t>(d) * total_patches + time_index;
                latent[dst] = frames[src];
            }
        }
    }
}

std::vector<float> patch_major_to_latent(const std::vector<float>& frames,
                                         int patch_size,
                                         int feat_dim) {
    std::vector<float> latent;
    patch_major_to_latent(frames, patch_size, feat_dim, latent);
    return latent;
}

void append_stream_frame(std::vector<float>& recent_frames,
                         const std::vector<float>& patch,
                         int max_frames,
                         int patch_size,
                         int feat_dim) {
    const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;
    recent_frames.insert(recent_frames.end(), patch.begin(), patch.end());
    const size_t max_elems = static_cast<size_t>(max_frames) * frame_stride;
    if (recent_frames.size() > max_elems) {
        recent_frames.erase(recent_frames.begin(),
                            recent_frames.begin() + static_cast<std::ptrdiff_t>(recent_frames.size() - max_elems));
    }
}

}  // namespace

VoiceStore::VoiceStore(std::string root_dir)
    : root_dir_(std::move(root_dir)) {
    std::filesystem::create_directories(root_dir_);
}

bool VoiceStore::has_voice(const std::string& id) const {
    return std::filesystem::exists(manifest_path_for(root_dir_, id)) &&
           std::filesystem::exists(prompt_path_for(root_dir_, id));
}

void VoiceStore::save_voice(const PromptFeatures& features) {
    if (!is_valid_voice_id(features.id)) {
        fail("Invalid voice id");
    }
    const auto dir = std::filesystem::path(root_dir_) / features.id;
    std::filesystem::create_directories(dir);

    json manifest = {
        {"id", features.id},
        {"prompt_text", features.prompt_text},
        {"prompt_audio_length", features.prompt_audio_length},
        {"sample_rate", features.sample_rate},
        {"patch_size", features.patch_size},
        {"feat_dim", features.feat_dim},
        {"created_at", features.created_at},
        {"updated_at", features.updated_at},
    };

    std::ofstream out(manifest_path_for(root_dir_, features.id));
    if (!out.is_open()) {
        fail("Failed to write voice manifest");
    }
    out << manifest.dump(2);
    write_binary_file(prompt_path_for(root_dir_, features.id), features.prompt_feat);
}

PromptFeatures VoiceStore::load_voice(const std::string& id) const {
    const auto manifest_path = manifest_path_for(root_dir_, id);
    const auto prompt_path = prompt_path_for(root_dir_, id);
    if (!std::filesystem::exists(manifest_path) || !std::filesystem::exists(prompt_path)) {
        fail("Voice not found: " + id);
    }

    std::ifstream in(manifest_path);
    if (!in.is_open()) {
        fail("Failed to read voice manifest");
    }
    json manifest = json::parse(in);
    PromptFeatures features;
    features.id = manifest.at("id").get<std::string>();
    features.prompt_text = manifest.at("prompt_text").get<std::string>();
    features.prompt_audio_length = manifest.at("prompt_audio_length").get<int>();
    features.sample_rate = manifest.at("sample_rate").get<int>();
    features.patch_size = manifest.at("patch_size").get<int>();
    features.feat_dim = manifest.at("feat_dim").get<int>();
    features.created_at = manifest.value("created_at", "");
    features.updated_at = manifest.value("updated_at", "");
    features.prompt_feat = read_binary_file(prompt_path);
    return features;
}

VoiceMetadata VoiceStore::load_metadata(const std::string& id) const {
    const PromptFeatures features = load_voice(id);
    return VoiceMetadata{
        features.id,
        features.prompt_text,
        features.prompt_audio_length,
        features.sample_rate,
        features.patch_size,
        features.feat_dim,
        features.created_at,
        features.updated_at,
    };
}

void VoiceStore::delete_voice(const std::string& id) {
    const auto dir = std::filesystem::path(root_dir_) / id;
    if (!std::filesystem::exists(dir)) {
        fail("Voice not found: " + id);
    }
    std::filesystem::remove_all(dir);
}

VoxCPMServiceCore::VoxCPMServiceCore(std::string model_path, BackendType backend_type, int threads)
    : model_path_(std::move(model_path)),
      backend_type_(backend_type),
      threads_(threads) {}

void VoxCPMServiceCore::load() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (loaded_) {
        return;
    }

    if (!std::filesystem::exists(model_path_) || !std::filesystem::is_regular_file(model_path_)) {
        fail("Model path must point to an existing GGUF file");
    }

    backend_ = std::make_unique<VoxCPMBackend>(backend_type_, threads_);
    store_ = std::make_shared<VoxCPMWeightStore>();
    if (!store_->load_from_file(model_path_, *backend_)) {
        fail("Failed to load GGUF: " + model_path_);
    }
    if (!runtime_.load_from_store(store_, *backend_)) {
        fail("Failed to initialize VoxCPM runtime from GGUF");
    }
    if (!audio_vae_.load_from_store(store_)) {
        fail("Failed to initialize AudioVAE from GGUF");
    }

    tokenizer_ = std::make_unique<VoxCPMTokenizer>();
    if (!tokenizer_->load_from_store(*store_)) {
        fail("Failed to load tokenizer metadata from GGUF");
    }
    split_tokenizer_ = std::make_unique<ChineseCharSplitTokenizer>(*tokenizer_);
    loaded_ = true;
}

PromptFeatures VoxCPMServiceCore::encode_prompt_audio(const std::string& id,
                                                      const std::string& prompt_text,
                                                      const std::vector<float>& mono_audio,
                                                      int sample_rate) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_) {
        fail("Model core is not loaded");
    }
    return encode_prompt_audio_locked(id, prompt_text, mono_audio, sample_rate);
}

PromptFeatures VoxCPMServiceCore::encode_prompt_audio_locked(const std::string& id,
                                                             const std::string& prompt_text,
                                                             const std::vector<float>& mono_audio,
                                                             int sample_rate) {
    const int patch_size_value = runtime_.config().patch_size;
    const int feat_dim_value = runtime_.config().feat_dim;
    const int patch_len = patch_size_value * audio_vae_.config().hop_length();
    std::vector<float> resampled = resample_audio_to_rate(mono_audio, sample_rate, audio_vae_.config().sample_rate);
    if (resampled.size() % static_cast<size_t>(patch_len) != 0) {
        const size_t padding = static_cast<size_t>(patch_len) - (resampled.size() % static_cast<size_t>(patch_len));
        resampled.insert(resampled.begin(), padding, 0.0f);
    }

    PromptFeatures features;
    features.id = id;
    features.prompt_text = prompt_text;
    features.prompt_feat = extract_prompt_features(audio_vae_,
                                                   *backend_,
                                                   resampled,
                                                   audio_vae_.config().sample_rate,
                                                   patch_size_value,
                                                   feat_dim_value);
    features.prompt_audio_length =
        static_cast<int>(features.prompt_feat.size() / static_cast<size_t>(patch_size_value * feat_dim_value));
    features.sample_rate = audio_vae_.config().sample_rate;
    features.patch_size = patch_size_value;
    features.feat_dim = feat_dim_value;
    const std::string now = make_timestamp_utc();
    features.created_at = now;
    features.updated_at = now;
    std::cerr << "[voice] encoded id=" << id
              << " prompt_audio_length=" << features.prompt_audio_length
              << " patch_size=" << features.patch_size
              << " feat_dim=" << features.feat_dim
              << " sample_rate=" << features.sample_rate
              << "\n";
    return features;
}

SynthesisResult VoxCPMServiceCore::synthesize(const SynthesisRequest& request) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_) {
        fail("Model core is not loaded");
    }
    return synthesize_locked(request);
}

SynthesisResult VoxCPMServiceCore::synthesize_locked(const SynthesisRequest& request) {
    if (request.text.empty()) {
        fail("Text input must not be empty");
    }
    const int patch_size_value = runtime_.config().patch_size;
    const int feat_dim_value = runtime_.config().feat_dim;
    const int patch_len = patch_size_value * audio_vae_.config().hop_length();
    const size_t expected_prompt_feat_size =
        static_cast<size_t>(request.prompt.prompt_audio_length) *
        static_cast<size_t>(patch_size_value) *
        static_cast<size_t>(feat_dim_value);

    if (request.prompt.prompt_audio_length < 0) {
        fail("Voice metadata is invalid: prompt_audio_length must be >= 0");
    }
    if (request.prompt.patch_size != patch_size_value) {
        fail("Voice metadata patch_size does not match the loaded model");
    }
    if (request.prompt.feat_dim != feat_dim_value) {
        fail("Voice metadata feat_dim does not match the loaded model");
    }
    if (request.prompt.prompt_feat.size() != expected_prompt_feat_size) {
        fail("Voice metadata is inconsistent with stored prompt features");
    }

    std::vector<int32_t> text_tokens = split_tokenizer_->encode(request.prompt.prompt_text + request.text, false);
    text_tokens.push_back(101);
    std::cerr << "[tts] synth start text_tokens=" << text_tokens.size()
              << " prompt_audio_length=" << request.prompt.prompt_audio_length
              << " prompt_feat_size=" << request.prompt.prompt_feat.size()
              << "\n";

    const int prompt_audio_length = request.prompt.prompt_audio_length;
    std::vector<int32_t> full_text_tokens = text_tokens;
    full_text_tokens.resize(text_tokens.size() + static_cast<size_t>(prompt_audio_length), 0);
    const int seq_len = static_cast<int>(full_text_tokens.size());

    std::vector<float> feat(static_cast<size_t>(seq_len) * patch_size_value * feat_dim_value, 0.0f);
    std::copy(request.prompt.prompt_feat.begin(),
              request.prompt.prompt_feat.end(),
              feat.begin() + static_cast<std::ptrdiff_t>(text_tokens.size()) * patch_size_value * feat_dim_value);

    std::vector<int32_t> text_mask(text_tokens.size(), 1);
    text_mask.resize(seq_len, 0);
    std::vector<int32_t> feat_mask(text_tokens.size(), 0);
    feat_mask.resize(seq_len, 1);

    VoxCPMDecodeState state = runtime_.prefill(full_text_tokens,
                                               text_mask,
                                               feat,
                                               feat_mask,
                                               seq_len,
                                               request.streaming_prefix_len);
    std::cerr << "[tts] prefill done seq_len=" << seq_len << "\n";

    const int target_text_token_count =
        std::max<int>(1, static_cast<int>(split_tokenizer_->tokenize(request.text).size()));
    const int max_len = std::min(target_text_token_count * 6 + 10, 2000);
    constexpr int kMinLen = 2;

    std::mt19937 rng(std::random_device{}());
    std::vector<float> generated_steps;
    generated_steps.reserve(static_cast<size_t>(max_len) * patch_size_value * feat_dim_value);
    std::vector<float> noise;
    std::vector<float> stream_recent_frames;
    std::vector<float> stream_latent;
    const size_t frame_stride = static_cast<size_t>(patch_size_value) * feat_dim_value;
    const int context_frames =
        (!request.prompt.prompt_feat.empty() && prompt_audio_length > 0 && request.streaming_prefix_len > 1)
            ? std::min(request.streaming_prefix_len - 1, prompt_audio_length)
            : 0;
    if (context_frames > 0) {
        const size_t context_offset = static_cast<size_t>(prompt_audio_length - context_frames) * frame_stride;
        stream_recent_frames.insert(stream_recent_frames.end(),
                                    request.prompt.prompt_feat.begin() + static_cast<std::ptrdiff_t>(context_offset),
                                    request.prompt.prompt_feat.end());
    }

    for (int step = 0; step < max_len; ++step) {
        fill_noise(noise, patch_size_value, feat_dim_value, rng);
        VoxCPMDecodeResult result = runtime_.decode(std::move(state),
                                                    noise,
                                                    request.inference_timesteps,
                                                    request.cfg_value);
        generated_steps.insert(generated_steps.end(), result.output_0.begin(), result.output_0.end());
        state = std::move(result.output_1);

        if (request.chunk_callback) {
            append_stream_frame(stream_recent_frames,
                                result.output_0,
                                request.streaming_prefix_len,
                                patch_size_value,
                                feat_dim_value);
            const int recent_frame_count =
                static_cast<int>(stream_recent_frames.size() / static_cast<size_t>(patch_size_value * feat_dim_value));
            const int recent_patches = recent_frame_count * patch_size_value;
            if (recent_patches > 0) {
                patch_major_to_latent(stream_recent_frames, patch_size_value, feat_dim_value, stream_latent);
                std::vector<float> chunk_waveform = decode_audio(audio_vae_, *backend_, stream_latent, recent_patches, feat_dim_value);
                if (chunk_waveform.size() > static_cast<size_t>(patch_len)) {
                    chunk_waveform.erase(chunk_waveform.begin(),
                                         chunk_waveform.end() - static_cast<std::ptrdiff_t>(patch_len));
                }
                request.chunk_callback(chunk_waveform);
            }
        }

        if (step > kMinLen && result.output_2) {
            break;
        }
    }
    std::cerr << "[tts] decode loop done generated_steps=" << generated_steps.size() << "\n";

    const int generated_frames = static_cast<int>(generated_steps.size() / static_cast<size_t>(patch_size_value * feat_dim_value));
    int prepended_context_frames = 0;
    const std::vector<float> decode_frames = build_decode_feature_sequence(request.prompt.prompt_feat,
                                                                           prompt_audio_length,
                                                                           generated_steps,
                                                                           request.streaming_prefix_len,
                                                                           patch_size_value,
                                                                           feat_dim_value,
                                                                           &prepended_context_frames);
    const int total_frames = static_cast<int>(decode_frames.size() / static_cast<size_t>(patch_size_value * feat_dim_value));
    const int total_patches = total_frames * patch_size_value;
    if (generated_frames == 0 || total_patches == 0) {
        fail("Model generated no audio patches");
    }

    const std::vector<float> latent = patch_major_to_latent(decode_frames, patch_size_value, feat_dim_value);
    std::vector<float> waveform = decode_audio(audio_vae_, *backend_, latent, total_patches, feat_dim_value);
    if (prompt_audio_length > 0) {
        const size_t trim = static_cast<size_t>(patch_len) * static_cast<size_t>(prepended_context_frames);
        if (waveform.size() > trim) {
            waveform.erase(waveform.begin(), waveform.begin() + static_cast<std::ptrdiff_t>(trim));
        }
    }

    return SynthesisResult{
        std::move(waveform),
        audio_vae_.config().sample_rate,
        generated_frames,
    };
}

int VoxCPMServiceCore::sample_rate() const {
    return audio_vae_.config().sample_rate;
}

int VoxCPMServiceCore::patch_size() const {
    return runtime_.config().patch_size;
}

int VoxCPMServiceCore::feat_dim() const {
    return runtime_.config().feat_dim;
}

std::string make_timestamp_utc() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &now_time);
#else
    gmtime_r(&now_time, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

bool is_valid_voice_id(const std::string& id) {
    if (id.empty()) {
        return false;
    }
    return std::all_of(id.begin(), id.end(), [](unsigned char c) {
        return std::isalnum(c) || c == '-' || c == '_' || c == '.';
    });
}

}  // namespace voxcpm

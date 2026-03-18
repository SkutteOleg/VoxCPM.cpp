#include <catch2/catch_test_macros.hpp>

#include "voxcpm/audio_io.h"
#include "voxcpm/server_common.h"

#include <filesystem>

namespace voxcpm {
namespace test {

TEST_CASE("Audio response format parser supports the documented formats", "[server]") {
    REQUIRE(parse_audio_response_format("mp3") == AudioResponseFormat::Mp3);
    REQUIRE(parse_audio_response_format("flac") == AudioResponseFormat::Flac);
    REQUIRE(parse_audio_response_format("wav") == AudioResponseFormat::Wav);
    REQUIRE(parse_audio_response_format("pcm") == AudioResponseFormat::Pcm);
    REQUIRE_THROWS(parse_audio_response_format("opus"));
    REQUIRE_THROWS(parse_audio_response_format("aac"));
    REQUIRE_THROWS(parse_audio_response_format("bogus"));
}

TEST_CASE("Voice ids are restricted to filesystem-safe characters", "[server]") {
    REQUIRE(is_valid_voice_id("voice_123"));
    REQUIRE(is_valid_voice_id("voice.demo-1"));
    REQUIRE_FALSE(is_valid_voice_id(""));
    REQUIRE_FALSE(is_valid_voice_id("../escape"));
    REQUIRE_FALSE(is_valid_voice_id("voice with space"));
}

TEST_CASE("VoiceStore persists manifest and prompt features round-trip", "[server]") {
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "voxcpm_server_voice_store_test";
    std::filesystem::remove_all(root);

    VoiceStore store(root.string());
    PromptFeatures features;
    features.id = "voice_abc";
    features.prompt_text = "hello";
    features.prompt_feat = {1.0f, 2.5f, -3.0f, 4.0f};
    features.prompt_audio_length = 2;
    features.sample_rate = 16000;
    features.patch_size = 2;
    features.feat_dim = 2;
    features.created_at = "2026-03-18T00:00:00Z";
    features.updated_at = "2026-03-18T00:00:01Z";

    store.save_voice(features);
    REQUIRE(store.has_voice(features.id));

    const PromptFeatures loaded = store.load_voice(features.id);
    REQUIRE(loaded.id == features.id);
    REQUIRE(loaded.prompt_text == features.prompt_text);
    REQUIRE(loaded.prompt_feat == features.prompt_feat);
    REQUIRE(loaded.prompt_audio_length == features.prompt_audio_length);
    REQUIRE(loaded.sample_rate == features.sample_rate);
    REQUIRE(loaded.patch_size == features.patch_size);
    REQUIRE(loaded.feat_dim == features.feat_dim);

    const VoiceMetadata metadata = store.load_metadata(features.id);
    REQUIRE(metadata.id == features.id);
    REQUIRE(metadata.prompt_text == features.prompt_text);

    store.delete_voice(features.id);
    REQUIRE_FALSE(store.has_voice(features.id));
    std::filesystem::remove_all(root);
}

}  // namespace test
}  // namespace voxcpm

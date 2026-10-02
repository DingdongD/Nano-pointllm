#include "Vgtsu_fused_fps_knn_controller.h"
#include "verilated.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
constexpr int kPoints = 8192;
constexpr int kCenters = 512;
constexpr int kNeighbors = 32;
constexpr int kLanes = 64;
constexpr int kReductionLanes = 512;

std::vector<uint32_t> read_words(const std::string& path, size_t count) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open " + path);
    const size_t bytes = static_cast<size_t>(stream.tellg());
    if (bytes != count * sizeof(uint32_t))
        throw std::runtime_error("unexpected size for " + path);
    stream.seekg(0);
    std::vector<uint32_t> words(count);
    stream.read(reinterpret_cast<char*>(words.data()), bytes);
    if (!stream) throw std::runtime_error("cannot read " + path);
    return words;
}

std::vector<uint8_t> read_bytes(const std::string& path, size_t count) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open " + path);
    const size_t bytes = static_cast<size_t>(stream.tellg());
    if (bytes != count) throw std::runtime_error("unexpected size for " + path);
    stream.seekg(0);
    std::vector<uint8_t> values(count);
    stream.read(reinterpret_cast<char*>(values.data()), bytes);
    if (!stream) throw std::runtime_error("cannot read " + path);
    return values;
}

uint32_t float_key(uint32_t bits) {
    return bits & 0x80000000U ? ~bits : bits ^ 0x80000000U;
}

bool less_distance(uint32_t left, uint32_t right) {
    return float_key(left) < float_key(right);
}

bool greater_distance(uint32_t left, uint32_t right) {
    return float_key(left) > float_key(right);
}

bool fps_better(uint32_t candidate_distance, int candidate_index,
                uint32_t current_distance, int current_index) {
    if (candidate_distance != current_distance)
        return greater_distance(candidate_distance, current_distance);
    const int candidate_lane = candidate_index % kReductionLanes;
    const int current_lane = current_index % kReductionLanes;
    if (candidate_lane != current_lane) return candidate_lane < current_lane;
    return candidate_index / kReductionLanes < current_index / kReductionLanes;
}

uint32_t extract_bits(const WData* words, int offset, int width) {
    const int word = offset / 32;
    const int shift = offset % 32;
    uint64_t value = static_cast<uint64_t>(words[word]) >> shift;
    if (shift + width > 32)
        value |= static_cast<uint64_t>(words[word + 1]) << (32 - shift);
    return static_cast<uint32_t>(value & ((1ULL << width) - 1));
}

void tick(Vgtsu_fused_fps_knn_controller& top) {
    top.clk = 0;
    top.eval();
    top.clk = 1;
    top.eval();
}

void hash_value(uint64_t& hash, uint32_t value) {
    hash ^= value;
    hash *= 1099511628211ULL;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "usage: driver fps.bin knn.bin centers.bin knn_ref.bin valid.bin max_cycles\n";
        return 2;
    }
    Verilated::commandArgs(argc, argv);
    const auto fps = read_words(argv[1], static_cast<size_t>(kCenters) * kPoints);
    const auto knn = read_words(argv[2], static_cast<size_t>(kCenters) * kPoints);
    const auto reference_centers = read_words(argv[3], kCenters);
    const auto reference_knn = read_words(
        argv[4], static_cast<size_t>(kCenters) * kNeighbors);
    const auto fps_valid = read_bytes(argv[5], kPoints);
    const int max_cycles = std::atoi(argv[6]);

    Vgtsu_fused_fps_knn_controller top;
    top.rst_n = 0;
    top.in_valid = 0;
    top.in_fps_mask = 0;
    top.in_knn_mask = 0;
    top.out_ready = 0;
    tick(top);
    tick(top);
    top.rst_n = 1;

    std::vector<uint32_t> nearest(kPoints, 0x7f800000U);
    int center = 0;
    int round = 0;
    int tile = 0;
    int input_beats = 0;
    int output_rounds = 0;
    int neighbor_values = 0;
    int mismatches = 0;
    int center_mismatches = 0;
    int next_center_mismatches = 0;
    int knn_set_mismatches = 0;
    int neighbor_index_mismatches = 0;
    int neighbor_distance_mismatches = 0;
    uint64_t center_hash = 1469598103934665603ULL;
    uint64_t neighbor_hash = 1469598103934665603ULL;
    std::vector<std::pair<uint32_t, int>> candidates(kPoints);

    for (int cycle = 0; cycle < max_cycles; ++cycle) {
        top.in_valid = round < kCenters;
        uint64_t fps_mask = 0;
        for (int lane = 0; lane < kLanes; ++lane)
            if (fps_valid[tile * kLanes + lane]) fps_mask |= 1ULL << lane;
        top.in_fps_mask = fps_mask;
        top.in_knn_mask = std::numeric_limits<uint64_t>::max();
        top.out_ready = 1;
        if (top.in_valid) {
            for (int lane = 0; lane < kLanes; ++lane) {
                const int index = tile * kLanes + lane;
                top.in_fps_distances[lane] = fps[round * kPoints + index];
                top.in_knn_distances[lane] = knn[round * kPoints + index];
            }
        }
        top.clk = 0;
        top.eval();
        const bool input_fire = top.in_valid && top.in_ready;
        const bool output_fire = top.out_valid && top.out_ready;

        if (input_fire) {
            for (int lane = 0; lane < kLanes; ++lane) {
                const int index = tile * kLanes + lane;
                const uint32_t fps_value = fps[round * kPoints + index];
                if (fps_valid[index] && less_distance(fps_value, nearest[index]))
                    nearest[index] = fps_value;
                candidates[index] = {knn[round * kPoints + index], index};
            }
            ++input_beats;
            ++tile;
        }

        if (output_fire) {
            int expected_next = 0;
            uint32_t expected_distance = 0;
            for (int index = 0; index < kPoints; ++index)
                if (fps_valid[index] && fps_better(
                        nearest[index], index, expected_distance, expected_next)) {
                    expected_next = index;
                    expected_distance = nearest[index];
                }
            std::partial_sort(
                candidates.begin(), candidates.begin() + kNeighbors,
                candidates.end(), [](const auto& left, const auto& right) {
                    if (left.first != right.first)
                        return less_distance(left.first, right.first);
                    return left.second < right.second;
                });

            const int actual_center = static_cast<int>(top.out_center);
            const int actual_next = static_cast<int>(top.out_next_center);
            if (actual_center != static_cast<int>(reference_centers[round])) {
                ++mismatches;
                ++center_mismatches;
            }
            if (round + 1 < kCenters &&
                actual_next != static_cast<int>(reference_centers[round + 1])) {
                ++mismatches;
                ++next_center_mismatches;
            }
            if (actual_next != expected_next) ++mismatches;
            hash_value(center_hash, static_cast<uint32_t>(actual_center));
            hash_value(center_hash, static_cast<uint32_t>(actual_next));

            std::unordered_set<int> expected_set;
            for (int slot = 0; slot < kNeighbors; ++slot)
                expected_set.insert(static_cast<int>(
                    reference_knn[round * kNeighbors + slot]));
            bool set_mismatch = false;
            for (int slot = 0; slot < kNeighbors; ++slot) {
                const int actual_index = static_cast<int>(extract_bits(
                    top.out_neighbor_indices, slot * 13, 13));
                const uint32_t actual_distance = top.out_neighbor_distances[slot];
                if (actual_index != candidates[slot].second) {
                    if (neighbor_index_mismatches < 4)
                        std::cerr << "INDEX_MISMATCH round=" << round
                                  << " slot=" << slot << " actual=" << actual_index
                                  << " expected=" << candidates[slot].second << "\n";
                    ++mismatches;
                    ++neighbor_index_mismatches;
                }
                if (actual_distance != candidates[slot].first) {
                    if (neighbor_distance_mismatches < 4)
                        std::cerr << "DISTANCE_MISMATCH round=" << round
                                  << " slot=" << slot << " actual=" << actual_distance
                                  << " expected=" << candidates[slot].first << "\n";
                    ++mismatches;
                    ++neighbor_distance_mismatches;
                }
                if (!expected_set.erase(actual_index)) set_mismatch = true;
                hash_value(neighbor_hash, static_cast<uint32_t>(actual_index));
                hash_value(neighbor_hash, actual_distance);
                ++neighbor_values;
            }
            if (set_mismatch || !expected_set.empty()) {
                ++mismatches;
                ++knn_set_mismatches;
            }
            center = expected_next;
            tile = 0;
            ++round;
            ++output_rounds;
        }

        top.clk = 1;
        top.eval();
        if (top.done) {
            std::cout << "SUMMARY " << cycle + 1 << " " << input_beats << " "
                      << output_rounds << " " << neighbor_values << " "
                      << mismatches << " " << center_mismatches << " "
                      << next_center_mismatches << " " << knn_set_mismatches << " "
                      << neighbor_index_mismatches << " "
                      << neighbor_distance_mismatches << " "
                      << center_hash << " " << neighbor_hash << "\n";
            top.final();
            return mismatches == 0 ? 0 : 1;
        }
    }
    std::cerr << "real FPS/KNN driver timed out\n";
    top.final();
    return 1;
}

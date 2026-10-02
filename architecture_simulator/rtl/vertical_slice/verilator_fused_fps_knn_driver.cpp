#include "Vgtsu_fused_fps_knn_controller.h"
#include "verilated.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace {
constexpr int kPoints = 8192;
constexpr int kCenters = 512;
constexpr int kNeighbors = 32;
constexpr int kLanes = 64;
constexpr int kReductionLanes = 512;

struct Point { int x; int y; int z; };

Point point(int index) {
    return {
        (index * 17 + 3) % 257 - 128,
        (index * index * 5 + 7) % 251 - 125,
        (index * 11 + 5) % 241 - 120,
    };
}

uint32_t fps_distance(int center, int index) {
    const Point a = point(center);
    const Point b = point(index);
    const int dx = a.x - b.x;
    const int dy = a.y - b.y;
    const int dz = a.z - b.z;
    return static_cast<uint32_t>(dx * dx + dy * dy + dz * dz);
}

uint32_t knn_distance(int center, int index) {
    return fps_distance(center, index) * 8U
        + static_cast<uint32_t>((index * 7 + center * 3) % 7);
}

bool fps_better(uint32_t candidate_distance, int candidate_index,
                uint32_t current_distance, int current_index) {
    if (candidate_distance != current_distance)
        return candidate_distance > current_distance;
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
    Verilated::commandArgs(argc, argv);
    const int max_cycles = argc > 1 ? std::atoi(argv[1]) : 100000;
    Vgtsu_fused_fps_knn_controller top;
    top.rst_n = 0;
    top.in_valid = 0;
    top.in_fps_mask = 0;
    top.in_knn_mask = 0;
    top.out_ready = 0;
    tick(top);
    tick(top);
    top.rst_n = 1;

    std::vector<uint32_t> nearest(kPoints, std::numeric_limits<uint32_t>::max());
    int center = 0;
    int round = 0;
    int tile = 0;
    int input_beats = 0;
    int output_rounds = 0;
    int neighbor_values = 0;
    int mismatches = 0;
    uint64_t center_hash = 1469598103934665603ULL;
    uint64_t neighbor_hash = 1469598103934665603ULL;
    std::vector<std::pair<uint32_t, int>> candidates(kPoints);

    for (int cycle = 0; cycle < max_cycles; ++cycle) {
        top.in_valid = round < kCenters;
        top.in_fps_mask = std::numeric_limits<uint64_t>::max();
        top.in_knn_mask = std::numeric_limits<uint64_t>::max();
        top.out_ready = 1;
        if (top.in_valid) {
            for (int lane = 0; lane < kLanes; ++lane) {
                const int index = tile * kLanes + lane;
                top.in_fps_distances[lane] = fps_distance(center, index);
                top.in_knn_distances[lane] = knn_distance(center, index);
            }
        }
        top.clk = 0;
        top.eval();
        const bool input_fire = top.in_valid && top.in_ready;
        const bool output_fire = top.out_valid && top.out_ready;

        if (input_fire) {
            for (int lane = 0; lane < kLanes; ++lane) {
                const int index = tile * kLanes + lane;
                const uint32_t fps = fps_distance(center, index);
                nearest[index] = std::min(nearest[index], fps);
                candidates[index] = {knn_distance(center, index), index};
            }
            ++input_beats;
            ++tile;
        }

        int expected_next = 0;
        if (output_fire) {
            for (int index = 1; index < kPoints; ++index)
                if (fps_better(nearest[index], index,
                               nearest[expected_next], expected_next))
                    expected_next = index;
            std::partial_sort(
                candidates.begin(), candidates.begin() + kNeighbors,
                candidates.end(), [](const auto& left, const auto& right) {
                    return left < right;
                }
            );
            if (static_cast<int>(top.out_center) != center) ++mismatches;
            if (static_cast<int>(top.out_next_center) != expected_next) ++mismatches;
            hash_value(center_hash, static_cast<uint32_t>(top.out_center));
            hash_value(center_hash, static_cast<uint32_t>(top.out_next_center));
            for (int slot = 0; slot < kNeighbors; ++slot) {
                const int actual_index = static_cast<int>(extract_bits(
                    top.out_neighbor_indices, slot * 13, 13));
                const uint32_t actual_distance = top.out_neighbor_distances[slot];
                if (actual_index != candidates[slot].second) ++mismatches;
                if (actual_distance != candidates[slot].first) ++mismatches;
                hash_value(neighbor_hash, static_cast<uint32_t>(actual_index));
                hash_value(neighbor_hash, actual_distance);
                ++neighbor_values;
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
                      << mismatches << " " << center_hash << " "
                      << neighbor_hash << "\n";
            top.final();
            return mismatches == 0 ? 0 : 1;
        }
    }
    std::cerr << "production FPS/KNN driver timed out\n";
    top.final();
    return 1;
}

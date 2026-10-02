#include "Vgtsu_dense64_weight_reuse_fabric.h"
#include "verilated.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace {
int wrap_i8(int value) {
    return ((value + 128) % 256 + 256) % 256 - 128;
}

uint32_t activation(int row, int chunk) {
    uint32_t packed = 0;
    for (int lane = 0; lane < 4; ++lane) {
        const int value = wrap_i8(row * 17 + (chunk * 4 + lane) * 7 - 33);
        packed |= static_cast<uint32_t>(static_cast<uint8_t>(value)) << (lane * 8);
    }
    return packed;
}

void weights(Vgtsu_dense64_weight_reuse_fabric& top, int n_tile,
             int k_block, int first_line, int n, int block_chunks,
             int memory_lanes) {
    for (int memory_lane = 0; memory_lane < memory_lanes; ++memory_lane) {
        const int line = first_line + memory_lane;
        const int chunk = k_block * block_chunks + line / 4;
        const int quarter = line % 4;
        for (int local = 0; local < 16; ++local) {
            const int column = n_tile * 64 + quarter * 16 + local;
            uint32_t packed = 0;
            for (int lane = 0; lane < 4; ++lane) {
                const int value = column < n
                    ? wrap_i8(column * 13 - (chunk * 4 + lane) * 5 + 29) : 0;
                packed |= static_cast<uint32_t>(static_cast<uint8_t>(value))
                    << (lane * 8);
            }
            top.weight_lines[memory_lane * 16 + local] = packed;
        }
    }
}

int32_t expected_output(int row, int column, int k) {
    int32_t result = 0;
    for (int index = 0; index < k; ++index) {
        result += wrap_i8(row * 17 + index * 7 - 33)
            * wrap_i8(column * 13 - index * 5 + 29);
    }
    return result;
}

void tick(Vgtsu_dense64_weight_reuse_fabric& top) {
    top.clk = 0;
    top.eval();
    top.clk = 1;
    top.eval();
}
}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    if (argc != 9) {
        std::cerr << "expected M N K M_TILE K_BLOCK memory_lanes max_cycles output_tiles\n";
        return 2;
    }
    const int m = std::atoi(argv[1]);
    const int n = std::atoi(argv[2]);
    const int k = std::atoi(argv[3]);
    const int m_tile = std::atoi(argv[4]);
    const int k_block = std::atoi(argv[5]);
    const int memory_lanes = std::atoi(argv[6]);
    const int max_cycles = std::atoi(argv[7]);
    const int output_tiles = std::atoi(argv[8]);
    (void)m;
    (void)m_tile;
    const int n_tiles = (n + 63) / 64;
    const int block_chunks = k_block / 4;

    Vgtsu_dense64_weight_reuse_fabric top;
    top.rst_n = 0;
    top.weight_valid = 0;
    top.activation_valid = 0;
    top.output_ready = 0;
    tick(top);
    tick(top);
    top.rst_n = 1;

    int output_values = 0;
    int mismatches = 0;
    uint64_t output_hash = 1469598103934665603ULL;
    for (int cycle = 0; cycle < max_cycles; ++cycle) {
        top.weight_valid = 1;
        top.activation_valid = 1;
        top.output_ready = 1;
        top.clk = 0;
        top.eval();
        weights(top, top.request_weight_n_tile, top.request_weight_k_block,
                top.request_weight_line, n, block_chunks, memory_lanes);
        top.activation_data = activation(
            top.request_activation_row, top.request_activation_chunk
        );
        top.eval();

        const bool output_fire = top.output_valid && top.output_ready;
        if (output_fire) {
            const int row = top.output_tag / n_tiles;
            const int base_column = (top.output_tag % n_tiles) * 64;
            for (int local = 0; local < 64; ++local) {
                if ((top.output_column_mask >> local) & 1ULL) {
                    const int column = base_column + local;
                    const int32_t actual = static_cast<int32_t>(
                        top.output_accumulators[local]
                    );
                    const int32_t expected = expected_output(row, column, k);
                    if (actual != expected) ++mismatches;
                    output_hash ^= static_cast<uint32_t>(actual);
                    output_hash *= 1099511628211ULL;
                    ++output_values;
                }
            }
        }

        top.clk = 1;
        top.eval();
        if (output_fire && static_cast<int>(top.count_output_tiles) == output_tiles) {
            std::cout << "SUMMARY " << cycle + 1 << " "
                      << top.count_weight_lines << " "
                      << top.count_wbuf_bank_writes << " "
                      << top.count_wbuf_load_tiles << " "
                      << top.count_activation_chunks << " "
                      << top.count_wbuf_read_issues << " "
                      << top.count_wbuf_responses << " "
                      << top.count_dot4_chunks << " "
                      << top.count_partial_tiles << " "
                      << top.count_output_tiles << " " << output_values << " "
                      << top.count_fifo_peak << " " << mismatches << " "
                      << output_hash << " " << static_cast<int>(top.done) << "\n";
            top.final();
            return mismatches == 0 ? 0 : 1;
        }
    }
    std::cerr << "weight-reuse Verilator driver timed out\n";
    top.final();
    return 1;
}

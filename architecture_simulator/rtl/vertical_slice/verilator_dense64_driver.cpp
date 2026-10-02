#include "Vgtsu_dense64_abuf_wbuf_fabric.h"
#include "verilated.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace {
int wrap_i8(int value) {
    int wrapped = ((value + 128) % 256 + 256) % 256 - 128;
    return wrapped;
}

uint32_t activation(int sequence, int m, int n, int k) {
    (void)m;
    const int n_tiles = (n + 63) / 64;
    const int chunks = k / 4;
    const int row = sequence / (n_tiles * chunks);
    const int chunk = sequence % chunks;
    uint32_t result = 0;
    for (int lane = 0; lane < 4; ++lane) {
        result |= static_cast<uint32_t>(static_cast<uint8_t>(wrap_i8(
            row * 17 + (chunk * 4 + lane) * 7 - 33))) << (lane * 8);
    }
    return result;
}

void weights(Vgtsu_dense64_abuf_wbuf_fabric& top, int sequence,
             int quarter, int n, int k) {
    const int n_tiles = (n + 63) / 64;
    const int chunks = k / 4;
    const int within_row = sequence % (n_tiles * chunks);
    const int n_tile = within_row / chunks;
    const int chunk = within_row % chunks;
    for (int local = 0; local < 16; ++local) {
        const int column = n_tile * 64 + quarter * 16 + local;
        uint32_t packed = 0;
        for (int lane = 0; lane < 4; ++lane) {
            const int value = column < n
                ? wrap_i8(column * 13 - (chunk * 4 + lane) * 5 + 29) : 0;
            packed |= static_cast<uint32_t>(static_cast<uint8_t>(value))
                << (lane * 8);
        }
        top.line_weights[local] = packed;
    }
}

void tick(Vgtsu_dense64_abuf_wbuf_fabric& top) {
    top.clk = 0;
    top.eval();
    top.clk = 1;
    top.eval();
}
}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    if (argc != 10) {
        std::cerr << "expected M N K source_mod source_phase output_mod "
                     "output_phase max_cycles output_tiles\n";
        return 2;
    }
    const int m = std::atoi(argv[1]);
    const int n = std::atoi(argv[2]);
    const int k = std::atoi(argv[3]);
    const int source_mod = std::atoi(argv[4]);
    const int source_phase = std::atoi(argv[5]);
    const int output_mod = std::atoi(argv[6]);
    const int output_phase = std::atoi(argv[7]);
    const int max_cycles = std::atoi(argv[8]);
    const int output_tiles = std::atoi(argv[9]);
    const int packets = m * ((n + 63) / 64) * (k / 4);
    const int lines = packets * 4;

    Vgtsu_dense64_abuf_wbuf_fabric top;
    top.rst_n = 0;
    top.line_valid = 0;
    top.output_ready = 0;
    tick(top);
    tick(top);
    top.rst_n = 1;

    int sent = 0;
    int output_values = 0;
    int source_backpressure = 0;
    int output_backpressure = 0;
    for (int cycle = 0; cycle < max_cycles; ++cycle) {
        const bool source_gate = source_mod == 0 || cycle % source_mod != source_phase;
        const bool output_gate = output_mod == 0 || cycle % output_mod != output_phase;
        const int sequence = sent / 4;
        const int quarter = sent % 4;
        top.line_valid = sent < lines && source_gate;
        top.line_sequence = sequence;
        top.line_quarter = quarter;
        top.line_activation = activation(sequence, m, n, k);
        weights(top, sequence, quarter, n, k);
        top.output_ready = output_gate;
        top.clk = 0;
        top.eval();

        const bool line_fire = top.line_valid && top.line_ready;
        const bool output_fire = top.output_valid && top.output_ready;
        if (top.line_valid && !top.line_ready) ++source_backpressure;
        if (top.output_valid && !top.output_ready) ++output_backpressure;
        if (output_fire) {
            const int base_column = (top.output_tag % ((n + 63) / 64)) * 64;
            for (int column = 0; column < 64; ++column) {
                if ((top.output_column_mask >> column) & 1ULL) {
                    const int32_t value = static_cast<int32_t>(
                        top.output_accumulators[column]);
                    std::cout << "TRACE " << cycle << " OUTPUT_ACCEPT "
                              << base_column + column << " " << value << "\n";
                    ++output_values;
                }
            }
        }

        top.clk = 1;
        top.eval();
        if (line_fire) ++sent;
        if (output_fire && static_cast<int>(top.count_output_tiles) == output_tiles) {
            std::cout << "SUMMARY " << cycle + 1 << " "
                      << top.count_ingress_lines << " "
                      << top.count_wbuf_bank_writes << " "
                      << top.count_abuf_writes << " "
                      << top.count_wbuf_read_issues << " "
                      << top.count_wbuf_responses << " "
                      << top.count_dot4_chunks << " "
                      << top.count_output_tiles << " " << output_values << " "
                      << top.count_fill_compute_overlap << " "
                      << source_backpressure << " " << output_backpressure << " "
                      << top.count_fifo_peak << " "
                      << static_cast<int>(top.overflow_error) << " "
                      << static_cast<int>(top.done) << "\n";
            top.final();
            return 0;
        }
    }
    std::cerr << "Dense64 Verilator driver timed out\n";
    top.final();
    return 1;
}

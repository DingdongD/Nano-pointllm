// Structural composition of weight-stream Dense64 with the correlated dual
// A8/BF16 post-processing adapter. Scale metadata is selected by dense tag.
module gtsu_dense64_stream_dual_output #(
    parameter integer M = 1,
    parameter integer N = 4096,
    parameter integer K = 4096,
    parameter integer N_MEM_LANES = 4,
    parameter integer BF16_LANES = 16,
    parameter integer READ_LATENCY = 3,
    parameter integer FIFO_DEPTH = 8
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         line_valid,
    output wire                         line_ready,
    input  wire [31:0]                  line_sequence,
    input  wire [1:0]                   line_quarter,
    input  wire [31:0]                  line_activation,
    input  wire [N_MEM_LANES*512-1:0]   line_weights,
    output wire [31:0]                  scale_request_tag,
    input  wire [2047:0]                scale_biases,
    input  wire [2047:0]                scale_multipliers,
    input  wire [383:0]                 scale_right_shifts,
    input  wire [15:0]                  activation_scale_fp16,
    input  wire [1023:0]                weight_scales_fp16,
    output wire                         output_valid,
    input  wire                         output_ready,
    output wire [31:0]                  output_tag,
    output wire [63:0]                  output_mask,
    output wire [511:0]                 output_a8,
    output wire [1023:0]                output_bf16,
    output wire                         done,
    output wire                         overflow_error
);
    wire dense_valid;
    wire dense_ready;
    wire [31:0] dense_tag;
    wire [63:0] dense_mask;
    wire [2047:0] dense_accumulators;
    wire dense_done;
    wire post_idle;
    assign scale_request_tag = dense_tag;
    assign done = dense_done && post_idle;

    wire [31:0] unused_ingress_lines;
    wire [31:0] unused_bank_writes;
    wire [31:0] unused_abuf_writes;
    wire [31:0] unused_read_issues;
    wire [31:0] unused_responses;
    wire [31:0] unused_chunks;
    wire [31:0] unused_dense_tiles;
    wire [31:0] unused_overlap;
    wire [31:0] unused_fifo_peak;
    gtsu_dense64_abuf_wbuf_fabric #(
        .M(M), .N(N), .K(K), .N_MEM_LANES(N_MEM_LANES),
        .READ_LATENCY(READ_LATENCY), .FIFO_DEPTH(FIFO_DEPTH)
    ) dense (
        .clk(clk), .rst_n(rst_n), .line_valid(line_valid),
        .line_ready(line_ready), .line_sequence(line_sequence),
        .line_quarter(line_quarter), .line_activation(line_activation),
        .line_weights(line_weights), .output_valid(dense_valid),
        .output_ready(dense_ready), .output_tag(dense_tag),
        .output_column_mask(dense_mask),
        .output_accumulators(dense_accumulators), .done(dense_done),
        .overflow_error(overflow_error),
        .count_ingress_lines(unused_ingress_lines),
        .count_wbuf_bank_writes(unused_bank_writes),
        .count_abuf_writes(unused_abuf_writes),
        .count_wbuf_read_issues(unused_read_issues),
        .count_wbuf_responses(unused_responses),
        .count_dot4_chunks(unused_chunks),
        .count_output_tiles(unused_dense_tiles),
        .count_fill_compute_overlap(unused_overlap),
        .count_fifo_peak(unused_fifo_peak)
    );

    wire [31:0] unused_post_input_tiles;
    wire [31:0] unused_a8_values;
    wire [31:0] unused_bf16_groups;
    wire [31:0] unused_bf16_values;
    wire [31:0] unused_post_output_tiles;
    gtsu_dense64_dual_output #(.BF16_LANES(BF16_LANES)) post (
        .clk(clk), .rst_n(rst_n), .in_valid(dense_valid),
        .in_ready(dense_ready), .in_tag(dense_tag), .in_mask(dense_mask),
        .in_accumulators(dense_accumulators), .in_biases(scale_biases),
        .in_multipliers(scale_multipliers),
        .in_right_shifts(scale_right_shifts),
        .in_activation_scale_fp16(activation_scale_fp16),
        .in_weight_scales_fp16(weight_scales_fp16),
        .out_valid(output_valid), .out_ready(output_ready),
        .out_tag(output_tag), .out_mask(output_mask),
        .out_a8(output_a8), .out_bf16(output_bf16), .idle(post_idle),
        .count_input_tiles(unused_post_input_tiles),
        .count_a8_values(unused_a8_values),
        .count_bf16_groups(unused_bf16_groups),
        .count_bf16_values(unused_bf16_values),
        .count_output_tiles(unused_post_output_tiles)
    );
endmodule

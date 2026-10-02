// Completion-trace boundary: DRAMsim3 supplies tagged 64-byte lines, then the
// synthesizable DMA ROB unpacks into the physical Dense SRAM pipeline.
module gtsu_dense_dramsim_dma_pipeline #(
    parameter integer M = 2,
    parameter integer N = 5,
    parameter integer K = 16,
    parameter integer N_TILE = 3,
    parameter integer K_BLOCK = 8,
    parameter integer ROB_DEPTH = 4,
    parameter integer READ_LATENCY = 3,
    parameter integer FIFO_DEPTH = 8,
    parameter [31:0] REQUANT_MULTIPLIER = 32'd1,
    parameter [5:0] REQUANT_RIGHT_SHIFT = 6'd5,
    parameter integer N_TILES = (N+N_TILE-1)/N_TILE,
    parameter integer K_BLOCKS = K/K_BLOCK,
    parameter integer BLOCK_CHUNKS = K_BLOCK/4,
    parameter integer PAYLOADS = M*N_TILES*K_BLOCKS*BLOCK_CHUNKS,
    parameter integer LINES = PAYLOADS/4
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         completion_valid,
    output wire                         completion_ready,
    input  wire [31:0]                  completion_tag,
    input  wire [511:0]                 completion_data,
    output wire                         output_valid,
    input  wire                         output_ready,
    output wire [31:0]                  output_tag,
    output wire [N_TILE-1:0]            output_column_mask,
    output wire [N_TILE*8-1:0]          output_data,
    output wire                         done,
    output wire                         overflow_error,
    output wire                         dma_protocol_error,
    output wire [31:0]                  count_dma_completions,
    output wire [31:0]                  count_dma_words,
    output wire [31:0]                  count_dma_rob_backpressure,
    output wire [31:0]                  count_dma_rob_peak,
    output wire [31:0]                  count_payload_writes,
    output wire [31:0]                  count_sram_read_issues,
    output wire [31:0]                  count_sram_responses,
    output wire [31:0]                  count_dot4_inputs,
    output wire [31:0]                  count_output_tiles,
    output wire [31:0]                  count_read_write_overlap_cycles
);
    wire payload_valid;
    wire payload_ready;
    wire [31:0] payload_sequence;
    wire [7:0] payload_chunk;
    wire [127:0] payload_data;

    gtsu_dram_dma_unpack #(
        .LINES(LINES), .ROB_DEPTH(ROB_DEPTH), .BLOCK_CHUNKS(BLOCK_CHUNKS)
    ) dma (
        .clk(clk), .rst_n(rst_n),
        .completion_valid(completion_valid),
        .completion_ready(completion_ready), .completion_tag(completion_tag),
        .completion_data(completion_data), .payload_valid(payload_valid),
        .payload_ready(payload_ready), .payload_sequence(payload_sequence),
        .payload_chunk(payload_chunk), .payload_data(payload_data),
        .protocol_error(dma_protocol_error),
        .count_completion_accepts(count_dma_completions),
        .count_payload_words(count_dma_words),
        .count_rob_backpressure_cycles(count_dma_rob_backpressure),
        .count_rob_peak(count_dma_rob_peak)
    );

    gtsu_dense_sram_requant_pipeline #(
        .M(M), .N(N), .K(K), .N_TILE(N_TILE), .K_BLOCK(K_BLOCK),
        .READ_LATENCY(READ_LATENCY), .FIFO_DEPTH(FIFO_DEPTH),
        .REQUANT_MULTIPLIER(REQUANT_MULTIPLIER),
        .REQUANT_RIGHT_SHIFT(REQUANT_RIGHT_SHIFT)
    ) dense (
        .clk(clk), .rst_n(rst_n), .payload_valid(payload_valid),
        .payload_ready(payload_ready), .payload_sequence(payload_sequence),
        .payload_chunk(payload_chunk), .payload_data(payload_data),
        .output_valid(output_valid), .output_ready(output_ready),
        .output_tag(output_tag), .output_column_mask(output_column_mask),
        .output_data(output_data), .done(done),
        .overflow_error(overflow_error),
        .count_payload_writes(count_payload_writes),
        .count_sram_read_issues(count_sram_read_issues),
        .count_sram_responses(count_sram_responses),
        .count_dot4_inputs(count_dot4_inputs),
        .count_output_tiles(count_output_tiles),
        .count_read_write_overlap_cycles(count_read_write_overlap_cycles)
    );

    initial begin
        if ((PAYLOADS % 4) != 0)
            $error("strict DMA wrapper requires full four-word DRAM lines");
    end
endmodule

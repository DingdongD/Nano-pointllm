module gtsu_geometry_dram_sram_loader #(
    parameter integer POINTS = 128,
    parameter integer LANES = 64,
    parameter integer ROB_DEPTH = 8,
    parameter integer READ_LATENCY = 3,
    parameter integer LINES = POINTS / 4,
    parameter integer ROWS = (POINTS + LANES - 1) / LANES,
    parameter integer ROW_W = (ROWS <= 2) ? 1 : $clog2(ROWS),
    parameter integer INDEX_W = (POINTS <= 2) ? 1 : $clog2(POINTS)
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         completion_valid,
    output wire                         completion_ready,
    input  wire [31:0]                  completion_tag,
    input  wire [511:0]                 completion_data,
    input  wire                         min_load_valid,
    output wire                         min_load_ready,
    input  wire [ROW_W-1:0]             min_load_row,
    input  wire [LANES*32-1:0]          min_load_values,
    input  wire                         tile_req_valid,
    output wire                         tile_req_ready,
    input  wire [ROW_W-1:0]             tile_req_row,
    output wire                         tile_rsp_valid,
    output wire [LANES*128-1:0]         tile_rsp_point_words,
    output wire [LANES*32-1:0]          tile_rsp_min_values,
    output wire                         loader_done,
    output wire                         dma_protocol_error,
    output wire                         sram_collision,
    output wire [31:0]                  count_dma_completions,
    output wire [31:0]                  count_point_words
);
    wire payload_valid, payload_ready;
    wire [31:0] payload_sequence;
    wire [7:0] payload_chunk;
    wire [127:0] payload_data;
    wire [31:0] unused_dma_backpressure, unused_dma_peak;

    gtsu_dram_dma_unpack #(
        .LINES(LINES), .ROB_DEPTH(ROB_DEPTH), .BLOCK_CHUNKS(1)
    ) dma (
        .clk(clk), .rst_n(rst_n),
        .completion_valid(completion_valid),
        .completion_ready(completion_ready), .completion_tag(completion_tag),
        .completion_data(completion_data), .payload_valid(payload_valid),
        .payload_ready(payload_ready), .payload_sequence(payload_sequence),
        .payload_chunk(payload_chunk), .payload_data(payload_data),
        .protocol_error(dma_protocol_error),
        .count_completion_accepts(count_dma_completions),
        .count_payload_words(count_point_words),
        .count_rob_backpressure_cycles(unused_dma_backpressure),
        .count_rob_peak(unused_dma_peak)
    );

    assign loader_done = count_point_words == POINTS;

    gtsu_geometry_sram_fabric #(
        .POINTS(POINTS), .LANES(LANES), .READ_LATENCY(READ_LATENCY)
    ) sram (
        .clk(clk), .rst_n(rst_n),
        .point_load_valid(payload_valid), .point_load_ready(payload_ready),
        .point_load_index(payload_sequence[INDEX_W-1:0]),
        .point_load_word(payload_data),
        .min_load_valid(min_load_valid), .min_load_ready(min_load_ready),
        .min_load_row(min_load_row), .min_load_values(min_load_values),
        .tile_req_valid(tile_req_valid), .tile_req_ready(tile_req_ready),
        .tile_req_row(tile_req_row), .tile_rsp_valid(tile_rsp_valid),
        .tile_rsp_point_words(tile_rsp_point_words),
        .tile_rsp_min_values(tile_rsp_min_values),
        .min_update_valid(1'b0), .min_update_ready(), .min_update_row('0),
        .min_update_mask('0), .min_update_values('0),
        .same_address_rw_collision(sram_collision)
    );

    initial begin
        if (POINTS % 4 != 0)
            $error("geometry DMA loader requires full 64-byte lines");
    end
endmodule

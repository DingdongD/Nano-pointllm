// Physically composed Dense vertical slice:
// payload DMA -> 16-bank 128-bit 1R1W SRAM -> response FIFO -> shared Dot4
// -> parallel INT32-to-A8 requant. N_TILE<=3 keeps one A and all B vectors in
// one SRAM word; production N_TILE=64 requires a later multiword gather stage.
module gtsu_dense_sram_requant_pipeline #(
    parameter integer M = 2,
    parameter integer N = 5,
    parameter integer K = 16,
    parameter integer N_TILE = 3,
    parameter integer K_BLOCK = 8,
    parameter integer BANKS = 16,
    // Compact lock capacity prevents Yosys from expanding a production SRAM
    // macro into flops; bank count, word width, and latency remain production-like.
    parameter integer ROWS_PER_BANK = 4,
    parameter integer READ_LATENCY = 3,
    parameter integer FIFO_DEPTH = 8,
    parameter [31:0] REQUANT_MULTIPLIER = 32'd1,
    parameter [5:0] REQUANT_RIGHT_SHIFT = 6'd5,
    parameter integer ROW_W = (M <= 1) ? 1 : $clog2(M),
    parameter integer NT_W = (((N+N_TILE-1)/N_TILE) <= 1) ? 1 : $clog2((N+N_TILE-1)/N_TILE),
    parameter integer KB_W = (((K+K_BLOCK-1)/K_BLOCK) <= 1) ? 1 : $clog2((K+K_BLOCK-1)/K_BLOCK),
    parameter integer CHUNK_W = ((K_BLOCK/4) <= 1) ? 1 : $clog2(K_BLOCK/4),
    parameter integer BANK_W = (BANKS <= 1) ? 1 : $clog2(BANKS),
    parameter integer ROW_ADDR_W = (ROWS_PER_BANK <= 1) ? 1 : $clog2(ROWS_PER_BANK),
    parameter integer SRAM_ADDR_W = BANK_W + ROW_ADDR_W,
    parameter integer FIFO_COUNT_W = $clog2(FIFO_DEPTH + 1),
    parameter integer META_W = 32 + N_TILE + 2,
    parameter integer FIFO_W = 128 + META_W
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         payload_valid,
    output wire                         payload_ready,
    input  wire [31:0]                  payload_sequence,
    input  wire [7:0]                   payload_chunk,
    input  wire [127:0]                 payload_data,
    output wire                         output_valid,
    input  wire                         output_ready,
    output wire [31:0]                  output_tag,
    output wire [N_TILE-1:0]            output_column_mask,
    output wire [N_TILE*8-1:0]          output_data,
    output wire                         done,
    output wire                         overflow_error,
    output reg  [31:0]                  count_payload_writes,
    output reg  [31:0]                  count_sram_read_issues,
    output reg  [31:0]                  count_sram_responses,
    output reg  [31:0]                  count_dot4_inputs,
    output reg  [31:0]                  count_output_tiles,
    output reg  [31:0]                  count_read_write_overlap_cycles
);
    localparam integer N_TILES = (N + N_TILE - 1) / N_TILE;
    localparam integer TOTAL_CHUNKS = K / 4;
    localparam integer BLOCK_CHUNKS = K_BLOCK / 4;
    localparam integer K_BLOCKS = (TOTAL_CHUNKS + BLOCK_CHUNKS - 1) / BLOCK_CHUNKS;
    localparam integer TOTAL_BLOCKS = M * N_TILES * K_BLOCKS;
    localparam integer TAG_W = 4;

    reg [31:0] next_load_sequence;
    reg [7:0] next_load_chunk;
    wire payload_protocol_match = payload_sequence == next_load_sequence
        && payload_chunk == next_load_chunk;
    wire payload_last_chunk = next_load_chunk == BLOCK_CHUNKS - 1;

    wire ctrl_load_ready;
    wire ctrl_read_valid;
    wire ctrl_read_ready;
    wire [31:0] ctrl_read_sequence;
    wire [ROW_W-1:0] ctrl_read_row;
    wire [NT_W-1:0] ctrl_read_n_tile;
    wire [KB_W-1:0] ctrl_read_k_block;
    wire [CHUNK_W-1:0] ctrl_read_chunk;
    wire [N_TILE-1:0] ctrl_read_mask;
    wire ctrl_read_first;
    wire ctrl_read_last;
    wire ctrl_block_release;
    wire ctrl_done;

    wire sram_c0_ready;
    wire sram_c1_ready;
    wire payload_request = payload_valid && payload_protocol_match
        && next_load_sequence < TOTAL_BLOCKS && ctrl_load_ready;
    assign payload_ready = payload_protocol_match && next_load_sequence < TOTAL_BLOCKS
        && ctrl_load_ready && sram_c1_ready;
    wire payload_fire = payload_valid && payload_ready;
    wire ctrl_load_valid = payload_fire && payload_last_chunk;

    gtsu_dense_mnk_controller #(
        .M(M), .N(N), .K(K), .N_TILE(N_TILE), .K_BLOCK(K_BLOCK)
    ) controller (
        .clk(clk), .rst_n(rst_n),
        .load_valid(ctrl_load_valid), .load_ready(ctrl_load_ready),
        .load_sequence(next_load_sequence),
        .read_valid(ctrl_read_valid), .read_ready(ctrl_read_ready),
        .read_sequence(ctrl_read_sequence), .read_row(ctrl_read_row),
        .read_n_tile(ctrl_read_n_tile), .read_k_block(ctrl_read_k_block),
        .read_chunk(ctrl_read_chunk), .read_column_mask(ctrl_read_mask),
        .read_first(ctrl_read_first), .read_last(ctrl_read_last),
        .block_release(ctrl_block_release), .done(ctrl_done)
    );

    wire [SRAM_ADDR_W-1:0] write_word_addr =
        (next_load_sequence[0] * BLOCK_CHUNKS) + next_load_chunk;
    wire [SRAM_ADDR_W-1:0] read_word_addr =
        (ctrl_read_sequence[0] * BLOCK_CHUNKS) + ctrl_read_chunk;

    reg [TAG_W-1:0] issue_tag;
    reg [31:0] metadata_sequence [0:(1<<TAG_W)-1];
    reg [N_TILE-1:0] metadata_mask [0:(1<<TAG_W)-1];
    reg metadata_first [0:(1<<TAG_W)-1];
    reg metadata_last [0:(1<<TAG_W)-1];
    reg [FIFO_COUNT_W:0] outstanding_reads;

    wire [FIFO_COUNT_W-1:0] fifo_occupancy;
    wire response_fifo_valid;
    wire response_fifo_ready;
    wire [FIFO_W-1:0] response_fifo_data;
    wire response_fifo_in_ready;
    wire read_credit = outstanding_reads + fifo_occupancy < FIFO_DEPTH;
    assign ctrl_read_ready = sram_c0_ready && read_credit;
    wire read_fire = ctrl_read_valid && ctrl_read_ready;

    wire sram_c0_rsp_valid;
    wire [TAG_W-1:0] sram_c0_rsp_tag;
    wire [127:0] sram_c0_rsp_data;
    wire sram_c1_rsp_valid;
    wire [TAG_W-1:0] unused_c1_rsp_tag;
    wire [127:0] unused_c1_rsp_data;
    wire [BANK_W-1:0] unused_c0_rsp_bank;
    wire [BANK_W-1:0] unused_c1_rsp_bank;
    wire [ROW_ADDR_W-1:0] unused_c0_rsp_row;
    wire [ROW_ADDR_W-1:0] unused_c1_rsp_row;

    gtsu_banked_sram_2client #(
        .BANKS(BANKS), .ROWS_PER_BANK(ROWS_PER_BANK), .DATA_W(128),
        .READ_LATENCY(READ_LATENCY), .TAG_W(TAG_W)
    ) payload_sram (
        .clk(clk), .rst_n(rst_n),
        .c0_req_valid(ctrl_read_valid && read_credit),
        .c0_req_ready(sram_c0_ready), .c0_req_write(1'b0),
        .c0_req_word_addr(read_word_addr), .c0_req_tag(issue_tag),
        .c0_req_wdata(128'd0), .c0_req_bwen(16'd0),
        .c1_req_valid(payload_request), .c1_req_ready(sram_c1_ready),
        .c1_req_write(1'b1), .c1_req_word_addr(write_word_addr),
        .c1_req_tag({TAG_W{1'b0}}), .c1_req_wdata(payload_data),
        .c1_req_bwen(16'hffff),
        .c0_rsp_valid(sram_c0_rsp_valid), .c0_rsp_tag(sram_c0_rsp_tag),
        .c0_rsp_bank(unused_c0_rsp_bank), .c0_rsp_row(unused_c0_rsp_row),
        .c0_rsp_rdata(sram_c0_rsp_data),
        .c1_rsp_valid(sram_c1_rsp_valid), .c1_rsp_tag(unused_c1_rsp_tag),
        .c1_rsp_bank(unused_c1_rsp_bank), .c1_rsp_row(unused_c1_rsp_row),
        .c1_rsp_rdata(unused_c1_rsp_data),
        .event_c0_accept(), .event_c1_accept()
    );

    wire [META_W-1:0] response_metadata = {
        metadata_last[sram_c0_rsp_tag], metadata_first[sram_c0_rsp_tag],
        metadata_mask[sram_c0_rsp_tag], metadata_sequence[sram_c0_rsp_tag]
    };
    assign overflow_error = sram_c0_rsp_valid && !response_fifo_in_ready;

    gtsu_rv_fifo #(.WIDTH(FIFO_W), .DEPTH(FIFO_DEPTH)) response_fifo (
        .clk(clk), .rst_n(rst_n),
        .in_valid(sram_c0_rsp_valid), .in_ready(response_fifo_in_ready),
        .in_data({response_metadata, sram_c0_rsp_data}),
        .out_valid(response_fifo_valid), .out_ready(response_fifo_ready),
        .out_data(response_fifo_data), .occupancy(fifo_occupancy)
    );

    wire [127:0] fifo_payload = response_fifo_data[127:0];
    wire [31:0] fifo_sequence = response_fifo_data[128 +: 32];
    wire [N_TILE-1:0] fifo_mask = response_fifo_data[160 +: N_TILE];
    wire fifo_first = response_fifo_data[160+N_TILE];
    wire fifo_last = response_fifo_data[161+N_TILE];

    wire dense_in_ready;
    wire dense_out_valid;
    wire dense_out_ready;
    wire [31:0] dense_out_tag;
    wire [N_TILE-1:0] dense_out_mask;
    wire [N_TILE*32-1:0] dense_out_accumulator;
    assign response_fifo_ready = dense_in_ready;

    gtsu_dense_dot4_tile #(.N_TILE(N_TILE), .TAG_WIDTH(32)) dense_tile (
        .clk(clk), .rst_n(rst_n),
        .in_valid(response_fifo_valid), .in_ready(dense_in_ready),
        .in_tag(fifo_sequence), .in_first(fifo_first), .in_last(fifo_last),
        .in_column_mask(fifo_mask), .in_a(fifo_payload[31:0]),
        .in_b(fifo_payload[32 +: N_TILE*32]),
        .out_valid(dense_out_valid), .out_ready(dense_out_ready),
        .out_tag(dense_out_tag), .out_column_mask(dense_out_mask),
        .out_accumulator(dense_out_accumulator)
    );

    wire [N_TILE-1:0] requant_in_ready;
    wire [N_TILE-1:0] requant_out_valid;
    wire [N_TILE*32-1:0] requant_out_tag;
    wire [N_TILE*8-1:0] requant_out_data;
    wire all_requant_ready = &requant_in_ready;
    wire all_requant_valid = &requant_out_valid;
    wire any_requant_valid = |requant_out_valid;
    wire requant_fire = dense_out_valid && all_requant_ready;
    assign dense_out_ready = all_requant_ready;
    reg [N_TILE-1:0] output_mask_reg;

    genvar column;
    generate
        for (column = 0; column < N_TILE; column = column + 1) begin : requant_lane
            gtsu_requantize_int32 #(.TAG_WIDTH(32)) requant (
                .clk(clk), .rst_n(rst_n), .in_valid(requant_fire),
                .in_ready(requant_in_ready[column]), .in_tag(dense_out_tag),
                .in_accumulator(dense_out_accumulator[column*32 +: 32]),
                .in_bias(32'sd0), .in_multiplier(REQUANT_MULTIPLIER),
                .in_right_shift(REQUANT_RIGHT_SHIFT),
                .out_valid(requant_out_valid[column]),
                .out_ready(output_ready && all_requant_valid),
                .out_tag(requant_out_tag[column*32 +: 32]),
                .out_data(requant_out_data[column*8 +: 8])
            );
        end
    endgenerate

    assign output_valid = all_requant_valid;
    assign output_tag = requant_out_tag[31:0];
    assign output_column_mask = output_mask_reg;
    assign output_data = requant_out_data;
    wire output_fire = output_valid && output_ready;
    assign done = ctrl_done && next_load_sequence == TOTAL_BLOCKS
        && outstanding_reads == 0 && fifo_occupancy == 0
        && !dense_out_valid && !any_requant_valid;

    integer index;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            next_load_sequence <= 0;
            next_load_chunk <= 0;
            issue_tag <= 0;
            outstanding_reads <= 0;
            output_mask_reg <= 0;
            count_payload_writes <= 0;
            count_sram_read_issues <= 0;
            count_sram_responses <= 0;
            count_dot4_inputs <= 0;
            count_output_tiles <= 0;
            count_read_write_overlap_cycles <= 0;
            for (index = 0; index < (1<<TAG_W); index = index + 1) begin
                metadata_sequence[index] <= 0;
                metadata_mask[index] <= 0;
                metadata_first[index] <= 0;
                metadata_last[index] <= 0;
            end
        end else begin
            if (payload_fire) begin
                count_payload_writes <= count_payload_writes + 1;
                if (payload_last_chunk) begin
                    next_load_sequence <= next_load_sequence + 1;
                    next_load_chunk <= 0;
                end else begin
                    next_load_chunk <= next_load_chunk + 1'b1;
                end
            end
            if (read_fire) begin
                metadata_sequence[issue_tag] <= ctrl_read_sequence;
                metadata_mask[issue_tag] <= ctrl_read_mask;
                metadata_first[issue_tag] <= ctrl_read_first;
                metadata_last[issue_tag] <= ctrl_read_last;
                issue_tag <= issue_tag + 1'b1;
                count_sram_read_issues <= count_sram_read_issues + 1;
            end
            case ({read_fire, sram_c0_rsp_valid})
                2'b10: outstanding_reads <= outstanding_reads + 1'b1;
                2'b01: outstanding_reads <= outstanding_reads - 1'b1;
                default: outstanding_reads <= outstanding_reads;
            endcase
            if (sram_c0_rsp_valid)
                count_sram_responses <= count_sram_responses + 1;
            if (response_fifo_valid && dense_in_ready)
                count_dot4_inputs <= count_dot4_inputs + 1;
            if (requant_fire)
                output_mask_reg <= dense_out_mask;
            if (output_fire)
                count_output_tiles <= count_output_tiles + 1;
            if (payload_fire && read_fire)
                count_read_write_overlap_cycles <= count_read_write_overlap_cycles + 1;
        end
    end

    initial begin
        if (N_TILE <= 0 || N_TILE > 3)
            $error("one-word physical slice requires N_TILE in [1,3]");
        if ((K % K_BLOCK) != 0 || (K_BLOCK % 4) != 0)
            $error("physical slice currently requires full K blocks");
        if (FIFO_DEPTH <= READ_LATENCY)
            $error("response FIFO must exceed fixed SRAM read latency");
    end
endmodule

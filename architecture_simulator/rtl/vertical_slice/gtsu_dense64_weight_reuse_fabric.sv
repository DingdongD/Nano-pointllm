// High-M mode for the shared Dense64 tile. A K-block of weights remains in
// WBUF while M_TILE activation rows consume it; tagged partial sums preserve
// output state across K blocks.
module gtsu_dense64_weight_reuse_fabric #(
    parameter integer M = 5,
    parameter integer N = 70,
    parameter integer K = 24,
    parameter integer M_TILE = 3,
    parameter integer K_BLOCK = 16,
    parameter integer N_MEM_LANES = 4,
    parameter integer READ_LATENCY = 3,
    parameter integer FIFO_DEPTH = 8,
    parameter integer N_TILE = 64,
    parameter integer BANKS = 16,
    parameter integer N_TILES = (N+N_TILE-1)/N_TILE,
    parameter integer M_TILES = (M+M_TILE-1)/M_TILE,
    parameter integer CHUNKS = K/4,
    parameter integer BLOCK_CHUNKS = K_BLOCK/4,
    parameter integer K_BLOCKS = (CHUNKS+BLOCK_CHUNKS-1)/BLOCK_CHUNKS,
    parameter integer FIFO_W = 32+32+2048+64+2,
    parameter integer FIFO_COUNT_W = $clog2(FIFO_DEPTH+1)
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         weight_valid,
    output wire                         weight_ready,
    input  wire [N_MEM_LANES*512-1:0]   weight_lines,
    input  wire                         activation_valid,
    output wire                         activation_ready,
    input  wire [31:0]                  activation_data,
    output wire [31:0]                  request_weight_n_tile,
    output wire [31:0]                  request_weight_k_block,
    output wire [31:0]                  request_weight_line,
    output wire [31:0]                  request_activation_row,
    output wire [31:0]                  request_activation_chunk,
    output wire                         output_valid,
    input  wire                         output_ready,
    output wire [31:0]                  output_tag,
    output wire [63:0]                  output_column_mask,
    output wire [2047:0]                output_accumulators,
    output wire                         done,
    output wire                         overflow_error,
    output reg  [31:0]                  count_weight_lines,
    output reg  [31:0]                  count_wbuf_bank_writes,
    output reg  [31:0]                  count_wbuf_load_tiles,
    output reg  [31:0]                  count_activation_chunks,
    output reg  [31:0]                  count_wbuf_read_issues,
    output reg  [31:0]                  count_wbuf_responses,
    output reg  [31:0]                  count_dot4_chunks,
    output reg  [31:0]                  count_partial_tiles,
    output reg  [31:0]                  count_output_tiles,
    output reg  [31:0]                  count_fifo_peak
);
    localparam [1:0] STATE_FILL = 0;
    localparam [1:0] STATE_COMPUTE = 1;
    localparam [1:0] STATE_WAIT = 2;

    reg [1:0] state;
    reg [31:0] m_tile_index;
    reg [31:0] n_tile_index;
    reg [31:0] k_block_index;
    reg [31:0] fill_line;
    reg [31:0] local_row;
    reg [31:0] chunk_in_block;
    reg all_tiles_complete;

    reg [127:0] wbuf [0:BLOCK_CHUNKS-1][0:BANKS-1];
    reg signed [31:0] partial_sums [0:M_TILE-1][0:N_TILE-1];

    wire [31:0] rows_this_tile =
        (m_tile_index == M_TILES-1) ? M-(M_TILES-1)*M_TILE : M_TILE;
    wire [31:0] chunks_this_block =
        (k_block_index == K_BLOCKS-1)
        ? CHUNKS-(K_BLOCKS-1)*BLOCK_CHUNKS : BLOCK_CHUNKS;
    wire [31:0] global_row = m_tile_index*M_TILE + local_row;
    wire [31:0] global_chunk = k_block_index*BLOCK_CHUNKS + chunk_in_block;

    assign request_weight_n_tile = n_tile_index;
    assign request_weight_k_block = k_block_index;
    assign request_weight_line = fill_line;
    assign request_activation_row = global_row;
    assign request_activation_chunk = global_chunk;

    assign weight_ready = state == STATE_FILL
        && fill_line + N_MEM_LANES <= chunks_this_block*4;
    wire weight_fire = weight_valid && weight_ready;

    wire [FIFO_COUNT_W-1:0] fifo_occupancy;
    reg [READ_LATENCY-1:0] read_valid_pipe;
    reg [FIFO_W-1:0] read_data_pipe [0:READ_LATENCY-1];
    integer pipe_occupancy;
    integer pipe_index;
    always @(*) begin
        pipe_occupancy = 0;
        for (pipe_index = 0; pipe_index < READ_LATENCY; pipe_index = pipe_index + 1)
            pipe_occupancy = pipe_occupancy + read_valid_pipe[pipe_index];
    end
    wire read_credit = fifo_occupancy + pipe_occupancy < FIFO_DEPTH;
    assign activation_ready = state == STATE_COMPUTE && read_credit;
    wire read_fire = activation_valid && activation_ready;

    wire [2047:0] read_weights;
    genvar bank;
    generate
        for (bank = 0; bank < BANKS; bank = bank + 1) begin : read_bank
            assign read_weights[bank*128 +: 128] = wbuf[chunk_in_block][bank];
        end
    endgenerate
    wire [31:0] valid_columns = (n_tile_index == N_TILES-1)
        ? N-(N_TILES-1)*N_TILE : N_TILE;
    wire [63:0] read_mask = valid_columns >= 64
        ? 64'hffffffffffffffff : (64'hffffffffffffffff >> (64-valid_columns));
    wire [31:0] read_tag = {k_block_index[7:0], n_tile_index[7:0], global_row[15:0]};
    wire read_first = chunk_in_block == 0;
    wire read_last = chunk_in_block == chunks_this_block-1;
    wire [FIFO_W-1:0] read_payload = {
        read_last, read_first, read_mask, read_weights,
        activation_data, read_tag
    };

    wire response_fifo_in_ready;
    wire response_fifo_valid;
    wire response_fifo_ready;
    wire [FIFO_W-1:0] response_fifo_data;
    assign overflow_error = read_valid_pipe[READ_LATENCY-1]
        && !response_fifo_in_ready;
    gtsu_rv_fifo #(.WIDTH(FIFO_W), .DEPTH(FIFO_DEPTH)) response_fifo (
        .clk(clk), .rst_n(rst_n),
        .in_valid(read_valid_pipe[READ_LATENCY-1]),
        .in_ready(response_fifo_in_ready),
        .in_data(read_data_pipe[READ_LATENCY-1]),
        .out_valid(response_fifo_valid), .out_ready(response_fifo_ready),
        .out_data(response_fifo_data), .occupancy(fifo_occupancy)
    );

    wire [31:0] fifo_tag = response_fifo_data[31:0];
    wire [31:0] fifo_activation = response_fifo_data[63:32];
    wire [2047:0] fifo_weights = response_fifo_data[2111:64];
    wire [63:0] fifo_mask = response_fifo_data[2175:2112];
    wire fifo_first = response_fifo_data[2176];
    wire fifo_last = response_fifo_data[2177];

    wire dense_in_ready;
    wire dense_output_valid;
    wire dense_output_ready;
    wire [31:0] dense_output_tag;
    wire [63:0] dense_output_mask;
    wire [2047:0] dense_output_accumulators;
    wire [15:0] partial_row = dense_output_tag[15:0];
    wire [7:0] partial_n_tile = dense_output_tag[23:16];
    wire [7:0] partial_k_block = dense_output_tag[31:24];
    wire partial_is_final = partial_k_block == K_BLOCKS-1;

    reg output_valid_reg;
    reg [31:0] output_tag_reg;
    reg [63:0] output_mask_reg;
    reg [2047:0] output_accumulator_reg;
    assign output_valid = output_valid_reg;
    assign output_tag = output_tag_reg;
    assign output_column_mask = output_mask_reg;
    assign output_accumulators = output_accumulator_reg;
    assign dense_output_ready = !partial_is_final
        || !output_valid_reg || output_ready;
    assign response_fifo_ready = dense_in_ready;

    gtsu_dense_dot4_tile #(.N_TILE(64), .TAG_WIDTH(32)) dense_tile (
        .clk(clk), .rst_n(rst_n), .in_valid(response_fifo_valid),
        .in_ready(dense_in_ready), .in_tag(fifo_tag),
        .in_first(fifo_first), .in_last(fifo_last),
        .in_column_mask(fifo_mask), .in_a(fifo_activation),
        .in_b(fifo_weights), .out_valid(dense_output_valid),
        .out_ready(dense_output_ready), .out_tag(dense_output_tag),
        .out_column_mask(dense_output_mask),
        .out_accumulator(dense_output_accumulators)
    );

    assign done = all_tiles_complete && read_valid_pipe == 0
        && fifo_occupancy == 0 && !dense_output_valid && !output_valid_reg;

    integer index;
    integer row_index;
    integer write_bank;
    integer memory_lane;
    integer write_line;
    integer write_chunk;
    integer write_quarter;
    reg signed [31:0] partial_value;
    reg signed [31:0] combined_value;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= STATE_FILL;
            m_tile_index <= 0;
            n_tile_index <= 0;
            k_block_index <= 0;
            fill_line <= 0;
            local_row <= 0;
            chunk_in_block <= 0;
            all_tiles_complete <= 0;
            read_valid_pipe <= 0;
            output_valid_reg <= 0;
            output_tag_reg <= 0;
            output_mask_reg <= 0;
            output_accumulator_reg <= 0;
            for (index = 0; index < READ_LATENCY; index = index + 1)
                read_data_pipe[index] <= 0;
            // WBUF is fully written before use; block zero initializes every
            // live partial-sum row before a later K block can read it.
            count_weight_lines <= 0;
            count_wbuf_bank_writes <= 0;
            count_wbuf_load_tiles <= 0;
            count_activation_chunks <= 0;
            count_wbuf_read_issues <= 0;
            count_wbuf_responses <= 0;
            count_dot4_chunks <= 0;
            count_partial_tiles <= 0;
            count_output_tiles <= 0;
            count_fifo_peak <= 0;
        end else begin
            read_valid_pipe[0] <= read_fire;
            if (read_fire)
                read_data_pipe[0] <= read_payload;
            for (index = 1; index < READ_LATENCY; index = index + 1) begin
                read_valid_pipe[index] <= read_valid_pipe[index-1];
                if (read_valid_pipe[index-1])
                    read_data_pipe[index] <= read_data_pipe[index-1];
            end

            if (output_valid_reg && output_ready) begin
                output_valid_reg <= 0;
                count_output_tiles <= count_output_tiles + 1;
            end

            if (weight_fire) begin
                for (memory_lane = 0; memory_lane < N_MEM_LANES;
                     memory_lane = memory_lane + 1) begin
                    write_line = fill_line + memory_lane;
                    write_chunk = write_line / 4;
                    write_quarter = write_line % 4;
                    for (write_bank = 0; write_bank < 4; write_bank = write_bank + 1)
                        wbuf[write_chunk][write_quarter*4 + write_bank]
                            <= weight_lines[(memory_lane*4+write_bank)*128 +: 128];
                end
                count_weight_lines <= count_weight_lines + N_MEM_LANES;
                count_wbuf_bank_writes <= count_wbuf_bank_writes + 4*N_MEM_LANES;
                if (fill_line + N_MEM_LANES == chunks_this_block*4) begin
                    fill_line <= 0;
                    local_row <= 0;
                    chunk_in_block <= 0;
                    state <= STATE_COMPUTE;
                    count_wbuf_load_tiles <= count_wbuf_load_tiles + 1;
                end else begin
                    fill_line <= fill_line + N_MEM_LANES;
                end
            end

            if (read_fire) begin
                count_activation_chunks <= count_activation_chunks + 1;
                count_wbuf_read_issues <= count_wbuf_read_issues + 1;
                if (chunk_in_block == chunks_this_block-1) begin
                    chunk_in_block <= 0;
                    if (local_row == rows_this_tile-1)
                        state <= STATE_WAIT;
                    else
                        local_row <= local_row + 1;
                end else begin
                    chunk_in_block <= chunk_in_block + 1;
                end
            end

            if (read_valid_pipe[READ_LATENCY-1])
                count_wbuf_responses <= count_wbuf_responses + 1;
            if (response_fifo_valid && dense_in_ready)
                count_dot4_chunks <= count_dot4_chunks + 1;
            if (fifo_occupancy > count_fifo_peak)
                count_fifo_peak <= fifo_occupancy;

            if (dense_output_valid && dense_output_ready) begin
                count_partial_tiles <= count_partial_tiles + 1;
                for (index = 0; index < N_TILE; index = index + 1) begin
                    partial_value = $signed(
                        dense_output_accumulators[index*32 +: 32]
                    );
                    combined_value = partial_k_block == 0
                        ? partial_value
                        : partial_sums[partial_row-m_tile_index*M_TILE][index]
                            + partial_value;
                    if (partial_is_final)
                        output_accumulator_reg[index*32 +: 32] <= combined_value;
                    else
                        partial_sums[partial_row-m_tile_index*M_TILE][index]
                            <= combined_value;
                end
                if (partial_is_final) begin
                    output_valid_reg <= 1'b1;
                    output_tag_reg <= partial_row*N_TILES + partial_n_tile;
                    output_mask_reg <= dense_output_mask;
                end
                if (partial_row-m_tile_index*M_TILE == rows_this_tile-1) begin
                    fill_line <= 0;
                    local_row <= 0;
                    chunk_in_block <= 0;
                    if (k_block_index + 1 < K_BLOCKS) begin
                        k_block_index <= k_block_index + 1;
                        state <= STATE_FILL;
                    end else if (n_tile_index + 1 < N_TILES) begin
                        k_block_index <= 0;
                        n_tile_index <= n_tile_index + 1;
                        state <= STATE_FILL;
                    end else if (m_tile_index + 1 < M_TILES) begin
                        k_block_index <= 0;
                        n_tile_index <= 0;
                        m_tile_index <= m_tile_index + 1;
                        state <= STATE_FILL;
                    end else begin
                        all_tiles_complete <= 1'b1;
                        state <= STATE_WAIT;
                    end
                end
            end
        end
    end

    initial begin
        if (N_TILE != 64 || BANKS != 16 || K % 4 != 0 || K_BLOCK % 4 != 0)
            $error("weight-reuse Dense64 dimensions are invalid");
        if (!(N_MEM_LANES == 1 || N_MEM_LANES == 2 || N_MEM_LANES == 4))
            $error("N_MEM_LANES must be 1, 2, or 4");
        if (FIFO_DEPTH <= READ_LATENCY)
            $error("response FIFO depth must exceed read latency");
    end
endmodule

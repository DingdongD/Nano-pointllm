// Production-width operand fabric: four 512-bit ingress beats fill 16 WBUF
// banks, while one independent A4 word is broadcast to 64 shared Dot4 PEs.
module gtsu_dense64_abuf_wbuf_fabric #(
    parameter integer M = 2,
    parameter integer N = 70,
    parameter integer K = 16,
    parameter integer READ_LATENCY = 3,
    parameter integer FIFO_DEPTH = 8,
    parameter integer N_TILE = 64,
    parameter integer BANKS = 16,
    parameter integer N_TILES = (N+N_TILE-1)/N_TILE,
    parameter integer CHUNKS = K/4,
    parameter integer PACKETS = M*N_TILES*CHUNKS,
    parameter integer OUTPUT_TILES = M*N_TILES,
    parameter integer FIFO_W = 32+32+2048+64+2,
    parameter integer FIFO_COUNT_W = $clog2(FIFO_DEPTH+1)
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         line_valid,
    output wire                         line_ready,
    input  wire [31:0]                  line_sequence,
    input  wire [1:0]                   line_quarter,
    input  wire [31:0]                  line_activation,
    input  wire [511:0]                 line_weights,
    output wire                         output_valid,
    input  wire                         output_ready,
    output wire [31:0]                  output_tag,
    output wire [63:0]                  output_column_mask,
    output wire [2047:0]                output_accumulators,
    output wire                         done,
    output wire                         overflow_error,
    output reg  [31:0]                  count_ingress_lines,
    output reg  [31:0]                  count_wbuf_bank_writes,
    output reg  [31:0]                  count_abuf_writes,
    output reg  [31:0]                  count_wbuf_read_issues,
    output reg  [31:0]                  count_wbuf_responses,
    output reg  [31:0]                  count_dot4_chunks,
    output reg  [31:0]                  count_output_tiles,
    output reg  [31:0]                  count_fill_compute_overlap,
    output reg  [31:0]                  count_fifo_peak
);
    reg [127:0] wbuf [0:1][0:BANKS-1];
    reg [31:0] abuf [0:1];
    reg [1:0] slot_active;
    reg [1:0] slot_ready;
    reg [31:0] slot_sequence [0:1];
    reg [2:0] slot_next_quarter [0:1];
    reg [31:0] expected_sequence;

    wire fill_slot = line_sequence[0];
    wire fill_match = slot_active[fill_slot]
        && slot_sequence[fill_slot] == line_sequence
        && slot_next_quarter[fill_slot] == line_quarter;
    assign line_ready = !slot_ready[fill_slot]
        && ((!slot_active[fill_slot] && line_quarter == 0) || fill_match);
    wire line_fire = line_valid && line_ready;

    wire read_slot = expected_sequence[0];
    wire expected_ready = slot_ready[read_slot]
        && slot_sequence[read_slot] == expected_sequence;
    wire [2047:0] read_weights;
    genvar bank;
    generate
        for (bank = 0; bank < BANKS; bank = bank + 1) begin : read_bank
            assign read_weights[bank*128 +: 128] = wbuf[read_slot][bank];
        end
    endgenerate

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
    wire read_fire = expected_ready && expected_sequence < PACKETS && read_credit;

    wire [31:0] sequence_in_tile = expected_sequence % (N_TILES*CHUNKS);
    wire [31:0] n_tile_index = (sequence_in_tile / CHUNKS) % N_TILES;
    wire [31:0] chunk_index = sequence_in_tile % CHUNKS;
    wire [31:0] valid_columns = (n_tile_index == N_TILES-1)
        ? N - (N_TILES-1)*N_TILE : N_TILE;
    wire [63:0] read_mask = (valid_columns >= 64)
        ? 64'hffffffffffffffff : (64'hffffffffffffffff >> (64-valid_columns));
    wire read_first = chunk_index == 0;
    wire read_last = chunk_index == CHUNKS-1;
    wire [FIFO_W-1:0] read_payload = {
        read_last, read_first, read_mask, read_weights,
        abuf[read_slot], expected_sequence
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

    wire [31:0] fifo_sequence = response_fifo_data[31:0];
    wire [31:0] fifo_activation = response_fifo_data[63:32];
    wire [2047:0] fifo_weights = response_fifo_data[2111:64];
    wire [63:0] fifo_mask = response_fifo_data[2175:2112];
    wire fifo_first = response_fifo_data[2176];
    wire fifo_last = response_fifo_data[2177];
    wire dense_in_ready;
    assign response_fifo_ready = dense_in_ready;

    gtsu_dense_dot4_tile #(.N_TILE(64), .TAG_WIDTH(32)) dense_tile (
        .clk(clk), .rst_n(rst_n), .in_valid(response_fifo_valid),
        .in_ready(dense_in_ready), .in_tag(fifo_sequence / CHUNKS),
        .in_first(fifo_first), .in_last(fifo_last),
        .in_column_mask(fifo_mask), .in_a(fifo_activation),
        .in_b(fifo_weights), .out_valid(output_valid),
        .out_ready(output_ready), .out_tag(output_tag),
        .out_column_mask(output_column_mask),
        .out_accumulator(output_accumulators)
    );

    assign done = expected_sequence == PACKETS
        && read_valid_pipe == 0 && fifo_occupancy == 0
        && !output_valid;

    integer index;
    integer write_bank;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            slot_active <= 0;
            slot_ready <= 0;
            slot_sequence[0] <= 0;
            slot_sequence[1] <= 0;
            slot_next_quarter[0] <= 0;
            slot_next_quarter[1] <= 0;
            abuf[0] <= 0;
            abuf[1] <= 0;
            expected_sequence <= 0;
            read_valid_pipe <= 0;
            for (index = 0; index < READ_LATENCY; index = index + 1)
                read_data_pipe[index] <= 0;
            for (index = 0; index < BANKS; index = index + 1) begin
                wbuf[0][index] <= 0;
                wbuf[1][index] <= 0;
            end
            count_ingress_lines <= 0;
            count_wbuf_bank_writes <= 0;
            count_abuf_writes <= 0;
            count_wbuf_read_issues <= 0;
            count_wbuf_responses <= 0;
            count_dot4_chunks <= 0;
            count_output_tiles <= 0;
            count_fill_compute_overlap <= 0;
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
            if (line_fire) begin
                if (!slot_active[fill_slot]) begin
                    slot_active[fill_slot] <= 1'b1;
                    slot_sequence[fill_slot] <= line_sequence;
                    slot_next_quarter[fill_slot] <= 0;
                end
                for (write_bank = 0; write_bank < 4; write_bank = write_bank + 1)
                    wbuf[fill_slot][line_quarter*4 + write_bank]
                        <= line_weights[write_bank*128 +: 128];
                if (line_quarter == 0)
                    abuf[fill_slot] <= line_activation;
                if (line_quarter == 3) begin
                    slot_ready[fill_slot] <= 1'b1;
                    slot_next_quarter[fill_slot] <= 4;
                end else begin
                    slot_next_quarter[fill_slot] <= line_quarter + 1'b1;
                end
                count_ingress_lines <= count_ingress_lines + 1;
                count_wbuf_bank_writes <= count_wbuf_bank_writes + 4;
                if (line_quarter == 0)
                    count_abuf_writes <= count_abuf_writes + 1;
            end
            if (read_fire) begin
                slot_active[read_slot] <= 1'b0;
                slot_ready[read_slot] <= 1'b0;
                slot_next_quarter[read_slot] <= 0;
                expected_sequence <= expected_sequence + 1;
                count_wbuf_read_issues <= count_wbuf_read_issues + 1;
            end
            if (read_valid_pipe[READ_LATENCY-1])
                count_wbuf_responses <= count_wbuf_responses + 1;
            if (response_fifo_valid && dense_in_ready)
                count_dot4_chunks <= count_dot4_chunks + 1;
            if (output_valid && output_ready)
                count_output_tiles <= count_output_tiles + 1;
            if (line_fire && read_fire)
                count_fill_compute_overlap <= count_fill_compute_overlap + 1;
            if (fifo_occupancy > count_fifo_peak)
                count_fifo_peak <= fifo_occupancy;
        end
    end

    initial begin
        if (N_TILE != 64 || BANKS != 16 || (K % 4) != 0)
            $error("production fabric requires N_TILE=64, BANKS=16, K%%4=0");
        if (FIFO_DEPTH <= READ_LATENCY)
            $error("response FIFO depth must exceed read latency");
    end
endmodule

module tb_gtsu_dense64_abuf_wbuf_fabric;
    parameter integer M = 2;
    parameter integer N = 70;
    parameter integer K = 16;
    parameter integer READ_LATENCY = 3;
    parameter integer FIFO_DEPTH = 8;
    parameter integer SOURCE_STALL_MOD = 5;
    parameter integer SOURCE_STALL_PHASE = 2;
    parameter integer OUTPUT_STALL_MOD = 7;
    parameter integer OUTPUT_STALL_PHASE = 3;
    parameter integer TRACE_INTERNAL = 1;
    parameter integer MAX_CYCLES = 2000000;
    parameter integer N_TILE = 64;
    parameter integer N_TILES = (N+N_TILE-1)/N_TILE;
    parameter integer CHUNKS = K/4;
    parameter integer PACKETS = M*N_TILES*CHUNKS;
    parameter integer LINES = PACKETS*4;
    parameter integer OUTPUT_TILES = M*N_TILES;

    reg clk = 0;
    reg rst_n = 0;
    integer cycle = 0;
    integer sent = 0;
    integer output_values = 0;
    integer source_backpressure_cycles = 0;
    integer output_backpressure_cycles = 0;

    wire [31:0] line_sequence = sent / 4;
    wire [1:0] line_quarter = sent % 4;
    wire [31:0] line_activation = make_activation(line_sequence);
    wire [511:0] line_weights = make_weights(line_sequence, line_quarter);
    wire source_gate = (SOURCE_STALL_MOD == 0) ? 1'b1
        : ((cycle % SOURCE_STALL_MOD) != SOURCE_STALL_PHASE);
    wire line_valid = rst_n && sent < LINES && source_gate;
    wire line_ready;
    wire output_gate = (OUTPUT_STALL_MOD == 0) ? 1'b1
        : ((cycle % OUTPUT_STALL_MOD) != OUTPUT_STALL_PHASE);
    wire output_valid;
    wire output_ready = rst_n && output_gate;
    wire [31:0] output_tag;
    wire [63:0] output_column_mask;
    wire [2047:0] output_accumulators;
    wire done;
    wire overflow_error;
    wire [31:0] count_ingress_lines;
    wire [31:0] count_wbuf_bank_writes;
    wire [31:0] count_abuf_writes;
    wire [31:0] count_wbuf_read_issues;
    wire [31:0] count_wbuf_responses;
    wire [31:0] count_dot4_chunks;
    wire [31:0] count_output_tiles;
    wire [31:0] count_fill_compute_overlap;
    wire [31:0] count_fifo_peak;

    function automatic signed [7:0] wrap_i8;
        input integer value;
        integer wrapped;
        begin
            wrapped = ((value + 128) % 256 + 256) % 256 - 128;
            wrap_i8 = wrapped;
        end
    endfunction

    function automatic [31:0] make_activation;
        input [31:0] seq;
        integer row;
        integer chunk;
        integer lane;
        begin
            row = seq / (N_TILES*CHUNKS);
            chunk = seq % CHUNKS;
            make_activation = 0;
            for (lane = 0; lane < 4; lane = lane + 1)
                make_activation[lane*8 +: 8] = wrap_i8(
                    row*17 + (chunk*4+lane)*7 - 33
                );
        end
    endfunction

    function automatic [511:0] make_weights;
        input [31:0] seq;
        input [1:0] quarter;
        integer within_row;
        integer n_tile_index;
        integer chunk;
        integer local_column;
        integer column;
        integer lane;
        begin
            within_row = seq % (N_TILES*CHUNKS);
            n_tile_index = within_row / CHUNKS;
            chunk = within_row % CHUNKS;
            make_weights = 0;
            for (local_column = 0; local_column < 16; local_column = local_column + 1) begin
                column = n_tile_index*N_TILE + quarter*16 + local_column;
                for (lane = 0; lane < 4; lane = lane + 1)
                    make_weights[local_column*32+lane*8 +: 8] = (column < N)
                        ? wrap_i8(column*13 - (chunk*4+lane)*5 + 29) : 0;
            end
        end
    endfunction

    function automatic integer count_mask;
        input [63:0] mask;
        integer mask_index;
        begin
            count_mask = 0;
            for (mask_index = 0; mask_index < 64; mask_index = mask_index + 1)
                count_mask = count_mask + mask[mask_index];
        end
    endfunction

    always #5 clk = ~clk;
    initial begin
        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1;
    end

    gtsu_dense64_abuf_wbuf_fabric #(
        .M(M), .N(N), .K(K), .READ_LATENCY(READ_LATENCY),
        .FIFO_DEPTH(FIFO_DEPTH)
    ) dut (
        .clk(clk), .rst_n(rst_n), .line_valid(line_valid),
        .line_ready(line_ready), .line_sequence(line_sequence),
        .line_quarter(line_quarter), .line_activation(line_activation),
        .line_weights(line_weights), .output_valid(output_valid),
        .output_ready(output_ready), .output_tag(output_tag),
        .output_column_mask(output_column_mask),
        .output_accumulators(output_accumulators), .done(done),
        .overflow_error(overflow_error),
        .count_ingress_lines(count_ingress_lines),
        .count_wbuf_bank_writes(count_wbuf_bank_writes),
        .count_abuf_writes(count_abuf_writes),
        .count_wbuf_read_issues(count_wbuf_read_issues),
        .count_wbuf_responses(count_wbuf_responses),
        .count_dot4_chunks(count_dot4_chunks),
        .count_output_tiles(count_output_tiles),
        .count_fill_compute_overlap(count_fill_compute_overlap),
        .count_fifo_peak(count_fifo_peak)
    );

    integer column;
    integer base_column;
    always @(posedge clk) begin
        if (!rst_n) begin
            cycle <= 0;
            sent <= 0;
            output_values <= 0;
            source_backpressure_cycles <= 0;
            output_backpressure_cycles <= 0;
        end else begin
            if (line_valid && line_ready) begin
                if (TRACE_INTERNAL)
                    $display("TRACE %0d LINE_ACCEPT %0d %0h",
                        cycle, sent, line_quarter);
                sent <= sent + 1;
            end
            if (line_valid && !line_ready)
                source_backpressure_cycles <= source_backpressure_cycles + 1;
            if (dut.read_fire && TRACE_INTERNAL)
                $display("TRACE %0d WBUF_READ_ISSUE %0d 0",
                    cycle, dut.expected_sequence);
            if (dut.read_valid_pipe[READ_LATENCY-1] && TRACE_INTERNAL)
                $display("TRACE %0d WBUF_RESPONSE %0d 0", cycle,
                    dut.read_data_pipe[READ_LATENCY-1][31:0]);
            if (dut.response_fifo_valid && dut.dense_in_ready && TRACE_INTERNAL)
                $display("TRACE %0d DOT4_INPUT %0d 0", cycle,
                    dut.fifo_sequence);
            if (output_valid && !output_ready)
                output_backpressure_cycles <= output_backpressure_cycles + 1;
            if (output_valid && output_ready) begin
                base_column = (output_tag % N_TILES) * N_TILE;
                for (column = 0; column < N_TILE; column = column + 1) begin
                    if (output_column_mask[column]) begin
                        $display("TRACE %0d OUTPUT_ACCEPT %0d %0d", cycle,
                            base_column + column,
                            $signed(output_accumulators[column*32 +: 32]));
                    end
                end
                output_values <= output_values + count_mask(output_column_mask);
                if (count_output_tiles + 1 == OUTPUT_TILES) begin
                    #1;
                    $display("SUMMARY %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                        cycle + 1, count_ingress_lines,
                        count_wbuf_bank_writes, count_abuf_writes,
                        count_wbuf_read_issues, count_wbuf_responses,
                        count_dot4_chunks, count_output_tiles,
                        output_values, count_fill_compute_overlap,
                        source_backpressure_cycles,
                        output_backpressure_cycles, count_fifo_peak,
                        overflow_error, done);
                    $finish;
                end
            end
            cycle <= cycle + 1;
            if (overflow_error)
                $fatal(1, "production WBUF response overflow");
            if (cycle >= MAX_CYCLES)
                $fatal(1, "production Dense fabric timeout");
        end
    end
endmodule

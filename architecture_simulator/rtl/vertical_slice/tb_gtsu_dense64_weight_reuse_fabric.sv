module tb_gtsu_dense64_weight_reuse_fabric;
    parameter integer M = 5;
    parameter integer N = 70;
    parameter integer K = 24;
    parameter integer M_TILE = 3;
    parameter integer K_BLOCK = 16;
    parameter integer N_MEM_LANES = 4;
    parameter integer READ_LATENCY = 3;
    parameter integer FIFO_DEPTH = 8;
    parameter integer WEIGHT_STALL_MOD = 5;
    parameter integer WEIGHT_STALL_PHASE = 2;
    parameter integer ACTIVATION_STALL_MOD = 7;
    parameter integer ACTIVATION_STALL_PHASE = 3;
    parameter integer OUTPUT_STALL_MOD = 11;
    parameter integer OUTPUT_STALL_PHASE = 4;
    parameter integer MAX_CYCLES = 5000000;
    parameter integer TRACE_INTERNAL = 1;
    parameter integer N_TILE = 64;
    parameter integer N_TILES = (N+N_TILE-1)/N_TILE;
    parameter integer CHUNKS = K/4;
    parameter integer BLOCK_CHUNKS = K_BLOCK/4;
    parameter integer OUTPUT_TILES = M*N_TILES;

    reg clk = 0;
    reg rst_n = 0;
    integer cycle = 0;
    integer output_values = 0;
    integer weight_backpressure = 0;
    integer activation_backpressure = 0;
    integer output_backpressure = 0;

    wire weight_ready;
    wire weight_gate = WEIGHT_STALL_MOD == 0 ? 1'b1
        : cycle % WEIGHT_STALL_MOD != WEIGHT_STALL_PHASE;
    wire weight_valid = rst_n && weight_ready && weight_gate;
    wire activation_ready;
    wire activation_gate = ACTIVATION_STALL_MOD == 0 ? 1'b1
        : cycle % ACTIVATION_STALL_MOD != ACTIVATION_STALL_PHASE;
    wire activation_valid = rst_n && activation_ready && activation_gate;
    wire output_gate = OUTPUT_STALL_MOD == 0 ? 1'b1
        : cycle % OUTPUT_STALL_MOD != OUTPUT_STALL_PHASE;
    wire output_ready = rst_n && output_gate;

    wire [31:0] request_weight_n_tile;
    wire [31:0] request_weight_k_block;
    wire [31:0] request_weight_line;
    wire [31:0] request_activation_row;
    wire [31:0] request_activation_chunk;
    wire [N_MEM_LANES*512-1:0] weight_lines = make_weight_group(
        request_weight_n_tile, request_weight_k_block, request_weight_line
    );
    wire [31:0] activation_data = make_activation(
        request_activation_row, request_activation_chunk
    );
    wire output_valid;
    wire [31:0] output_tag;
    wire [63:0] output_column_mask;
    wire [2047:0] output_accumulators;
    wire done;
    wire overflow_error;
    wire [31:0] count_weight_lines;
    wire [31:0] count_wbuf_bank_writes;
    wire [31:0] count_wbuf_load_tiles;
    wire [31:0] count_activation_chunks;
    wire [31:0] count_wbuf_read_issues;
    wire [31:0] count_wbuf_responses;
    wire [31:0] count_dot4_chunks;
    wire [31:0] count_partial_tiles;
    wire [31:0] count_output_tiles;
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
        input [31:0] row;
        input [31:0] chunk;
        integer lane;
        begin
            make_activation = 0;
            for (lane = 0; lane < 4; lane = lane + 1)
                make_activation[lane*8 +: 8] = wrap_i8(
                    row*17 + (chunk*4+lane)*7 - 33
                );
        end
    endfunction

    function automatic [N_MEM_LANES*512-1:0] make_weight_group;
        input [31:0] n_tile;
        input [31:0] k_block;
        input [31:0] first_line;
        integer memory_lane;
        integer line;
        integer chunk;
        integer quarter;
        integer local_column;
        integer column;
        integer lane;
        begin
            make_weight_group = 0;
            for (memory_lane = 0; memory_lane < N_MEM_LANES;
                 memory_lane = memory_lane + 1) begin
                line = first_line + memory_lane;
                chunk = k_block*BLOCK_CHUNKS + line/4;
                quarter = line % 4;
                for (local_column = 0; local_column < 16;
                     local_column = local_column + 1) begin
                    column = n_tile*N_TILE + quarter*16 + local_column;
                    for (lane = 0; lane < 4; lane = lane + 1)
                        make_weight_group[
                            memory_lane*512+local_column*32+lane*8 +: 8
                        ] = column < N
                            ? wrap_i8(column*13-(chunk*4+lane)*5+29) : 0;
                end
            end
        end
    endfunction

    function automatic integer count_mask;
        input [63:0] mask;
        integer index;
        begin
            count_mask = 0;
            for (index = 0; index < 64; index = index + 1)
                count_mask = count_mask + mask[index];
        end
    endfunction

    always #5 clk = ~clk;
    initial begin
        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1;
    end

    gtsu_dense64_weight_reuse_fabric #(
        .M(M), .N(N), .K(K), .M_TILE(M_TILE), .K_BLOCK(K_BLOCK),
        .N_MEM_LANES(N_MEM_LANES), .READ_LATENCY(READ_LATENCY),
        .FIFO_DEPTH(FIFO_DEPTH)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .weight_valid(weight_valid), .weight_ready(weight_ready),
        .weight_lines(weight_lines),
        .activation_valid(activation_valid), .activation_ready(activation_ready),
        .activation_data(activation_data),
        .request_weight_n_tile(request_weight_n_tile),
        .request_weight_k_block(request_weight_k_block),
        .request_weight_line(request_weight_line),
        .request_activation_row(request_activation_row),
        .request_activation_chunk(request_activation_chunk),
        .output_valid(output_valid), .output_ready(output_ready),
        .output_tag(output_tag), .output_column_mask(output_column_mask),
        .output_accumulators(output_accumulators), .done(done),
        .overflow_error(overflow_error),
        .count_weight_lines(count_weight_lines),
        .count_wbuf_bank_writes(count_wbuf_bank_writes),
        .count_wbuf_load_tiles(count_wbuf_load_tiles),
        .count_activation_chunks(count_activation_chunks),
        .count_wbuf_read_issues(count_wbuf_read_issues),
        .count_wbuf_responses(count_wbuf_responses),
        .count_dot4_chunks(count_dot4_chunks),
        .count_partial_tiles(count_partial_tiles),
        .count_output_tiles(count_output_tiles),
        .count_fifo_peak(count_fifo_peak)
    );

    integer lane_index;
    integer column;
    integer row;
    integer base_column;
    always @(posedge clk) begin
        if (!rst_n) begin
            cycle <= 0;
            output_values <= 0;
            weight_backpressure <= 0;
            activation_backpressure <= 0;
            output_backpressure <= 0;
        end else begin
            if (weight_valid && weight_ready)
                for (lane_index = 0; lane_index < N_MEM_LANES;
                     lane_index = lane_index + 1)
                    if (TRACE_INTERNAL)
                        $display("TRACE %0d WEIGHT_LINE_ACCEPT %0d 0", cycle,
                            count_weight_lines + lane_index);
            if (weight_ready && !weight_gate)
                weight_backpressure <= weight_backpressure + 1;
            if (activation_ready && !activation_gate)
                activation_backpressure <= activation_backpressure + 1;
            if (dut.read_fire && TRACE_INTERNAL)
                $display("TRACE %0d WBUF_READ_ISSUE %0d 0", cycle,
                    dut.read_tag);
            if (dut.read_valid_pipe[READ_LATENCY-1] && TRACE_INTERNAL)
                $display("TRACE %0d WBUF_RESPONSE %0d 0", cycle,
                    dut.read_data_pipe[READ_LATENCY-1][31:0]);
            if (dut.response_fifo_valid && dut.dense_in_ready && TRACE_INTERNAL)
                $display("TRACE %0d DOT4_INPUT %0d 0", cycle, dut.fifo_tag);
            if (dut.dense_output_valid && dut.dense_output_ready && TRACE_INTERNAL)
                $display("TRACE %0d PARTIAL_ACCEPT %0d 0", cycle,
                    dut.dense_output_tag);
            if (output_valid && !output_ready)
                output_backpressure <= output_backpressure + 1;
            if (output_valid && output_ready) begin
                row = output_tag / N_TILES;
                base_column = (output_tag % N_TILES) * N_TILE;
                for (column = 0; column < N_TILE; column = column + 1)
                    if (output_column_mask[column])
                        $display("TRACE %0d OUTPUT_ACCEPT %0d %0d", cycle,
                            row*N + base_column + column,
                            $signed(output_accumulators[column*32 +: 32]));
                output_values <= output_values + count_mask(output_column_mask);
                if (count_output_tiles + 1 == OUTPUT_TILES) begin
                    #1;
                    $display("SUMMARY %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                        cycle+1, count_weight_lines, count_wbuf_bank_writes,
                        count_wbuf_load_tiles, count_activation_chunks,
                        count_wbuf_read_issues, count_wbuf_responses,
                        count_dot4_chunks, count_partial_tiles,
                        count_output_tiles, output_values,
                        weight_backpressure, activation_backpressure,
                        output_backpressure, count_fifo_peak,
                        overflow_error, done);
                    $finish;
                end
            end
            cycle <= cycle + 1;
            if (overflow_error)
                $fatal(1, "weight-reuse response overflow");
            if (cycle >= MAX_CYCLES)
                $fatal(1, "weight-reuse Dense timeout");
        end
    end
endmodule

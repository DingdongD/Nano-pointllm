module tb_gtsu_w8_splitk_gemv;
    parameter integer N_OUTPUTS = 4;
    parameter integer SPLIT_K = 2;
    parameter integer K_CHUNKS = 3;
    parameter integer LANES = 4;
    parameter integer SOURCE_LATENCY = 3;
    parameter integer COMP_FIFO_DEPTH = 2;
    parameter integer DECODED_FIFO_DEPTH = 2;
    parameter integer PARTIAL_FIFO_DEPTH = 2;
    parameter integer OUTPUT_STALL_MOD = 5;
    parameter integer OUTPUT_STALL_PHASE = 0;
    parameter integer MAX_CYCLES = 10000;
    parameter integer N_W = (N_OUTPUTS <= 1) ? 1 : $clog2(N_OUTPUTS);
    parameter integer P_W = (SPLIT_K <= 1) ? 1 : $clog2(SPLIT_K);
    parameter integer C_W = (K_CHUNKS <= 1) ? 1 : $clog2(K_CHUNKS);
    localparam integer TOTAL_BEATS = N_OUTPUTS * SPLIT_K * K_CHUNKS;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    integer cycle = 0;
    integer beat_index = 0;
    integer done_cycle = 0;
    wire weight_valid = rst_n && cycle >= SOURCE_LATENCY && beat_index < TOTAL_BEATS;
    wire weight_ready;
    wire [N_W-1:0] weight_n = beat_index / (SPLIT_K * K_CHUNKS);
    wire [P_W-1:0] weight_partition = (beat_index / K_CHUNKS) % SPLIT_K;
    wire [C_W-1:0] weight_chunk = beat_index % K_CHUNKS;
    wire [LANES*8-1:0] weight_data;
    wire [LANES*8-1:0] activation_data;

    wire output_valid;
    wire output_ready = OUTPUT_STALL_MOD == 0
        || cycle % OUTPUT_STALL_MOD != OUTPUT_STALL_PHASE;
    wire [N_W-1:0] output_n;
    wire signed [31:0] output_data;

    wire event_input_accept;
    wire event_unpack_accept;
    wire event_compute_accept;
    wire event_partial_push;
    wire event_output_accept;
    wire [N_W-1:0] event_unpack_n;
    wire [P_W-1:0] event_unpack_partition;
    wire [C_W-1:0] event_unpack_chunk;
    wire [N_W-1:0] event_compute_n;
    wire [P_W-1:0] event_compute_partition;
    wire [C_W-1:0] event_compute_chunk;
    wire signed [31:0] event_compute_dot;
    wire [N_W-1:0] event_partial_n;
    wire [P_W-1:0] event_partial_partition;
    wire signed [31:0] event_partial_data;
    wire [31:0] count_input_beats;
    wire [31:0] count_unpack_beats;
    wire [31:0] count_compute_beats;
    wire [31:0] count_partial_sums;
    wire [31:0] count_outputs;
    wire [31:0] stall_source;
    wire [31:0] stall_unpack;
    wire [31:0] stall_compute;
    wire [31:0] stall_reduction;

    always #5 clk = ~clk;

    function automatic [LANES*8-1:0] make_weight_data(input integer index);
        integer function_lane;
        integer function_value;
        begin
            make_weight_data = 0;
            for (function_lane = 0; function_lane < LANES; function_lane = function_lane + 1) begin
                function_value = (index * LANES + function_lane) % 7 - 3;
                make_weight_data[function_lane*8 +: 8] = function_value[7:0];
            end
        end
    endfunction

    function automatic [LANES*8-1:0] make_activation_data(input integer index);
        integer function_lane;
        integer function_partition;
        integer function_chunk;
        integer function_value;
        begin
            function_partition = (index / K_CHUNKS) % SPLIT_K;
            function_chunk = index % K_CHUNKS;
            make_activation_data = 0;
            for (function_lane = 0; function_lane < LANES; function_lane = function_lane + 1) begin
                function_value = (
                    ((function_partition * K_CHUNKS + function_chunk) * LANES
                    + function_lane) % 5
                ) - 2;
                make_activation_data[function_lane*8 +: 8] = function_value[7:0];
            end
        end
    endfunction

    assign weight_data = make_weight_data(beat_index);
    assign activation_data = make_activation_data(beat_index);

    gtsu_w8_splitk_gemv #(
        .N_OUTPUTS(N_OUTPUTS),
        .SPLIT_K(SPLIT_K),
        .K_CHUNKS(K_CHUNKS),
        .LANES(LANES),
        .COMP_FIFO_DEPTH(COMP_FIFO_DEPTH),
        .DECODED_FIFO_DEPTH(DECODED_FIFO_DEPTH),
        .PARTIAL_FIFO_DEPTH(PARTIAL_FIFO_DEPTH)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .weight_valid(weight_valid), .weight_ready(weight_ready),
        .weight_n(weight_n), .weight_partition(weight_partition),
        .weight_chunk(weight_chunk), .weight_data(weight_data),
        .activation_data(activation_data),
        .output_valid(output_valid), .output_ready(output_ready),
        .output_n(output_n), .output_data(output_data),
        .event_input_accept(event_input_accept),
        .event_unpack_accept(event_unpack_accept),
        .event_compute_accept(event_compute_accept),
        .event_partial_push(event_partial_push),
        .event_output_accept(event_output_accept),
        .event_unpack_n(event_unpack_n),
        .event_unpack_partition(event_unpack_partition),
        .event_unpack_chunk(event_unpack_chunk),
        .event_compute_n(event_compute_n),
        .event_compute_partition(event_compute_partition),
        .event_compute_chunk(event_compute_chunk),
        .event_compute_dot(event_compute_dot),
        .event_partial_n(event_partial_n),
        .event_partial_partition(event_partial_partition),
        .event_partial_data(event_partial_data),
        .count_input_beats(count_input_beats),
        .count_unpack_beats(count_unpack_beats),
        .count_compute_beats(count_compute_beats),
        .count_partial_sums(count_partial_sums),
        .count_outputs(count_outputs),
        .stall_source(stall_source),
        .stall_unpack(stall_unpack),
        .stall_compute(stall_compute),
        .stall_reduction(stall_reduction)
    );

    initial begin
        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            cycle <= 0;
            beat_index <= 0;
        end else begin
            if (event_input_accept) begin
                $display("TRACE %0d INPUT_ACCEPT %0d %0d %0d 0",
                    cycle, weight_n, weight_partition, weight_chunk);
                beat_index <= beat_index + 1;
            end
            if (event_unpack_accept)
                $display("TRACE %0d UNPACK_ACCEPT %0d %0d %0d 0",
                    cycle, event_unpack_n, event_unpack_partition, event_unpack_chunk);
            if (event_compute_accept)
                $display("TRACE %0d COMPUTE_ACCEPT %0d %0d %0d %0d",
                    cycle, event_compute_n, event_compute_partition,
                    event_compute_chunk, event_compute_dot);
            if (event_partial_push)
                $display("TRACE %0d PARTIAL_PUSH %0d %0d %0d %0d",
                    cycle, event_partial_n, event_partial_partition,
                    K_CHUNKS - 1, event_partial_data);
            if (event_output_accept) begin
                $display("TRACE %0d OUTPUT_ACCEPT %0d %0d %0d %0d",
                    cycle, output_n, SPLIT_K - 1, K_CHUNKS - 1, output_data);
                if (output_n == N_OUTPUTS - 1) begin
                    done_cycle = cycle;
                    #1;
                    $display(
                        "SUMMARY %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                        done_cycle + 1,
                        count_input_beats, count_unpack_beats, count_compute_beats,
                        count_partial_sums, count_outputs,
                        stall_source, stall_unpack, stall_compute, stall_reduction
                    );
                    $finish;
                end
            end
            cycle <= cycle + 1;
            if (cycle >= MAX_CYCLES) begin
                $display("ERROR timeout cycle=%0d", cycle);
                $fatal(1);
            end
        end
    end
endmodule

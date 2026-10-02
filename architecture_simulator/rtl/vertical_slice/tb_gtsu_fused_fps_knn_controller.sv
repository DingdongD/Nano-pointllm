module tb_gtsu_fused_fps_knn_controller;
    parameter integer POINTS = 16;
    parameter integer CENTERS = 5;
    parameter integer K = 4;
    parameter integer LANES = 4;
    parameter integer DIST_WIDTH = 18;
    parameter integer SOURCE_STALL_MOD = 5;
    parameter integer SOURCE_STALL_PHASE = 2;
    parameter integer OUTPUT_STALL_MOD = 7;
    parameter integer OUTPUT_STALL_PHASE = 3;
    parameter integer MAX_CYCLES = 100000;
    localparam integer TILES = (POINTS + LANES - 1) / LANES;
    localparam integer TOTAL_BEATS = CENTERS * TILES;
    localparam integer INDEX_WIDTH = (POINTS <= 2) ? 1 : $clog2(POINTS);
    localparam integer TRACE_WIDTH = LANES + LANES*DIST_WIDTH;

    reg clk = 0;
    reg rst_n = 0;
    integer cycle = 0;
    integer trace_index = 0;
    integer input_beats = 0;
    integer distance_values = 0;
    integer round_outputs = 0;
    integer source_backpressure = 0;
    integer output_backpressure = 0;
    integer slot;
    reg [TRACE_WIDTH-1:0] trace [0:TOTAL_BEATS-1];
    reg [1023:0] trace_file;

    wire in_valid = trace_index < TOTAL_BEATS &&
                    (SOURCE_STALL_MOD == 0 || cycle % SOURCE_STALL_MOD != SOURCE_STALL_PHASE);
    wire in_ready;
    wire [LANES-1:0] in_mask = trace[trace_index][LANES-1:0];
    wire [LANES*DIST_WIDTH-1:0] in_distances = trace[trace_index][TRACE_WIDTH-1:LANES];
    wire out_valid;
    wire out_ready = OUTPUT_STALL_MOD == 0 || cycle % OUTPUT_STALL_MOD != OUTPUT_STALL_PHASE;
    wire [INDEX_WIDTH-1:0] out_center, out_next_center;
    wire [K*INDEX_WIDTH-1:0] out_neighbor_indices;
    wire [K*DIST_WIDTH-1:0] out_neighbor_distances;
    wire done;

    function integer count_mask(input [LANES-1:0] value);
        integer index;
        begin
            count_mask = 0;
            for (index = 0; index < LANES; index = index + 1)
                count_mask = count_mask + value[index];
        end
    endfunction

    gtsu_fused_fps_knn_controller #(
        .POINTS(POINTS), .CENTERS(CENTERS), .K(K), .LANES(LANES),
        .DIST_WIDTH(DIST_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_ready(in_ready),
        .in_mask(in_mask), .in_distances(in_distances),
        .out_valid(out_valid), .out_ready(out_ready), .out_center(out_center),
        .out_next_center(out_next_center),
        .out_neighbor_indices(out_neighbor_indices),
        .out_neighbor_distances(out_neighbor_distances), .done(done)
    );

    always #5 clk = ~clk;

    initial begin
        if (!$value$plusargs("TRACE_FILE=%s", trace_file)) begin
            $display("ERROR missing TRACE_FILE");
            $finish_and_return(2);
        end
        $readmemh(trace_file, trace);
        repeat (3) @(posedge clk);
        rst_n <= 1;
    end

    always @(posedge clk) begin
        if (rst_n) begin
            if (in_valid && !in_ready)
                source_backpressure <= source_backpressure + 1;
            if (out_valid && !out_ready)
                output_backpressure <= output_backpressure + 1;
            if (in_valid && in_ready) begin
                $display("TRACE %0d INPUT %0d %0d", cycle,
                         trace_index / TILES, trace_index % TILES);
                trace_index <= trace_index + 1;
                input_beats <= input_beats + 1;
                distance_values <= distance_values + count_mask(in_mask);
            end
            if (out_valid && out_ready) begin
                $display("TRACE %0d ROUND %0d %0d %0d", cycle, round_outputs,
                         out_center, out_next_center);
                for (slot = 0; slot < K; slot = slot + 1)
                    $display("TRACE %0d NEIGHBOR %0d %0d %0d %0d", cycle,
                             round_outputs, slot,
                             out_neighbor_indices[slot*INDEX_WIDTH +: INDEX_WIDTH],
                             out_neighbor_distances[slot*DIST_WIDTH +: DIST_WIDTH]);
                round_outputs <= round_outputs + 1;
            end
            cycle <= cycle + 1;
            if (done) begin
                $display("SUMMARY %0d %0d %0d %0d %0d %0d", cycle,
                         input_beats, distance_values, round_outputs,
                         source_backpressure, output_backpressure);
                $finish;
            end
            if (cycle >= MAX_CYCLES) begin
                $display("ERROR timeout");
                $finish_and_return(3);
            end
        end
    end
endmodule

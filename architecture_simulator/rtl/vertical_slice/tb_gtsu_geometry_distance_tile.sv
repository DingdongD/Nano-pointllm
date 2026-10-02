module tb_gtsu_geometry_distance_tile;
    parameter integer LANES = 8;
    parameter integer COORD_WIDTH = 16;
    parameter integer DIST_WIDTH = 36;
    parameter integer TAG_WIDTH = 16;
    parameter integer TILES = 7;
    parameter integer SOURCE_STALL_MOD = 4;
    parameter integer SOURCE_STALL_PHASE = 1;
    parameter integer OUTPUT_STALL_MOD = 5;
    parameter integer OUTPUT_STALL_PHASE = 2;
    parameter integer MAX_CYCLES = 10000;
    parameter integer TRACE_WIDTH = TAG_WIDTH + (3 + 3*LANES) * COORD_WIDTH;

    reg clk = 0;
    reg rst_n = 0;
    integer cycle = 0;
    integer sent = 0;
    integer input_tiles = 0;
    integer output_points = 0;
    integer source_backpressure_cycles = 0;
    integer output_backpressure_cycles = 0;
    integer lane;
    reg [1023:0] trace_file;
    reg [TRACE_WIDTH-1:0] trace [0:TILES-1];

    wire [TRACE_WIDTH-1:0] current_trace = trace[(sent < TILES) ? sent : 0];
    wire [TAG_WIDTH-1:0] in_tag = current_trace[0 +: TAG_WIDTH];
    wire signed [COORD_WIDTH-1:0] in_q_x = current_trace[TAG_WIDTH +: COORD_WIDTH];
    wire signed [COORD_WIDTH-1:0] in_q_y = current_trace[TAG_WIDTH+COORD_WIDTH +: COORD_WIDTH];
    wire signed [COORD_WIDTH-1:0] in_q_z = current_trace[TAG_WIDTH+2*COORD_WIDTH +: COORD_WIDTH];
    wire [LANES*COORD_WIDTH-1:0] in_point_x;
    wire [LANES*COORD_WIDTH-1:0] in_point_y;
    wire [LANES*COORD_WIDTH-1:0] in_point_z;
    wire source_gate = (SOURCE_STALL_MOD == 0)
                     ? 1'b1 : ((cycle % SOURCE_STALL_MOD) != SOURCE_STALL_PHASE);
    wire in_valid = rst_n && sent < TILES && source_gate;
    wire in_ready;
    wire output_gate = (OUTPUT_STALL_MOD == 0)
                     ? 1'b1 : ((cycle % OUTPUT_STALL_MOD) != OUTPUT_STALL_PHASE);
    wire out_ready = rst_n && output_gate;
    wire out_valid;
    wire [TAG_WIDTH-1:0] out_tag;
    wire [LANES*DIST_WIDTH-1:0] out_distance;

    genvar gi;
    generate
        for (gi = 0; gi < LANES; gi = gi + 1) begin : unpack_trace
            localparam integer BASE = TAG_WIDTH + (3 + 3*gi) * COORD_WIDTH;
            assign in_point_x[gi*COORD_WIDTH +: COORD_WIDTH] =
                current_trace[BASE +: COORD_WIDTH];
            assign in_point_y[gi*COORD_WIDTH +: COORD_WIDTH] =
                current_trace[BASE+COORD_WIDTH +: COORD_WIDTH];
            assign in_point_z[gi*COORD_WIDTH +: COORD_WIDTH] =
                current_trace[BASE+2*COORD_WIDTH +: COORD_WIDTH];
        end
    endgenerate

    always #5 clk = ~clk;
    initial begin
        if (!$value$plusargs("TRACE_FILE=%s", trace_file))
            $fatal(1, "TRACE_FILE plusarg is required");
        $readmemh(trace_file, trace);
        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1;
    end

    gtsu_geometry_distance_tile #(
        .LANES(LANES), .COORD_WIDTH(COORD_WIDTH),
        .DIST_WIDTH(DIST_WIDTH), .TAG_WIDTH(TAG_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_ready(in_ready),
        .in_tag(in_tag), .in_q_x(in_q_x), .in_q_y(in_q_y), .in_q_z(in_q_z),
        .in_point_x(in_point_x), .in_point_y(in_point_y), .in_point_z(in_point_z),
        .out_valid(out_valid), .out_ready(out_ready), .out_tag(out_tag),
        .out_distance(out_distance)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            cycle <= 0;
            sent <= 0;
            input_tiles <= 0;
            output_points <= 0;
            source_backpressure_cycles <= 0;
            output_backpressure_cycles <= 0;
        end else begin
            if (in_valid && in_ready) begin
                $display("TRACE %0d INPUT_ACCEPT %0d -1 0", cycle, in_tag);
                sent <= sent + 1;
                input_tiles <= input_tiles + 1;
            end
            if (in_valid && !in_ready)
                source_backpressure_cycles <= source_backpressure_cycles + 1;
            if (out_valid && !out_ready)
                output_backpressure_cycles <= output_backpressure_cycles + 1;
            if (out_valid && out_ready) begin
                for (lane = 0; lane < LANES; lane = lane + 1)
                    $display("TRACE %0d OUTPUT_ACCEPT %0d %0d %0d",
                        cycle, out_tag, lane,
                        out_distance[lane*DIST_WIDTH +: DIST_WIDTH]);
                output_points <= output_points + LANES;
                if (output_points + LANES == TILES * LANES) begin
                    #1;
                    $display("SUMMARY %0d %0d %0d %0d %0d",
                        cycle + 1, input_tiles, output_points,
                        source_backpressure_cycles, output_backpressure_cycles);
                    $finish;
                end
            end
            cycle <= cycle + 1;
            if (cycle >= MAX_CYCLES)
                $fatal(1, "geometry RTL timeout");
        end
    end
endmodule

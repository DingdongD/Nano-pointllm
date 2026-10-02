module tb_gtsu_shared_dot_geometry;
    parameter integer PE_COUNT = 8;
    parameter integer NORM_WIDTH = 16;
    parameter integer DIST_WIDTH = 18;
    parameter integer TAG_WIDTH = 16;
    parameter integer BEATS = 9;
    parameter integer SOURCE_STALL_MOD = 4;
    parameter integer SOURCE_STALL_PHASE = 1;
    parameter integer OUTPUT_STALL_MOD = 5;
    parameter integer OUTPUT_STALL_PHASE = 2;
    parameter integer MAX_CYCLES = 10000;
    parameter integer LANE_WIDTH = 64 + 2*NORM_WIDTH;
    parameter integer TRACE_WIDTH = TAG_WIDTH + 1 + PE_COUNT*LANE_WIDTH;

    reg clk = 0;
    reg rst_n = 0;
    integer cycle = 0;
    integer sent = 0;
    integer input_beats = 0;
    integer output_lanes = 0;
    integer feature_lanes = 0;
    integer geometry_lanes = 0;
    integer source_backpressure_cycles = 0;
    integer output_backpressure_cycles = 0;
    integer lane;
    reg [1023:0] trace_file;
    reg [TRACE_WIDTH-1:0] trace [0:BEATS-1];

    wire [TRACE_WIDTH-1:0] current_trace = trace[(sent < BEATS) ? sent : 0];
    wire [TAG_WIDTH-1:0] in_tag = current_trace[0 +: TAG_WIDTH];
    wire in_mode_geometry = current_trace[TAG_WIDTH];
    wire [PE_COUNT*32-1:0] in_a;
    wire [PE_COUNT*32-1:0] in_b;
    wire [PE_COUNT*NORM_WIDTH-1:0] in_a_norm;
    wire [PE_COUNT*NORM_WIDTH-1:0] in_b_norm;
    wire source_gate = (SOURCE_STALL_MOD == 0)
                     ? 1'b1 : ((cycle % SOURCE_STALL_MOD) != SOURCE_STALL_PHASE);
    wire in_valid = rst_n && sent < BEATS && source_gate;
    wire in_ready;
    wire output_gate = (OUTPUT_STALL_MOD == 0)
                     ? 1'b1 : ((cycle % OUTPUT_STALL_MOD) != OUTPUT_STALL_PHASE);
    wire out_ready = rst_n && output_gate;
    wire out_valid;
    wire [TAG_WIDTH-1:0] out_tag;
    wire out_mode_geometry;
    wire [PE_COUNT*32-1:0] out_dot;
    wire [PE_COUNT*DIST_WIDTH-1:0] out_distance;

    genvar pe;
    generate
        for (pe = 0; pe < PE_COUNT; pe = pe + 1) begin : unpack_trace
            localparam integer BASE = TAG_WIDTH + 1 + pe*LANE_WIDTH;
            assign in_a[pe*32 +: 32] = current_trace[BASE +: 32];
            assign in_b[pe*32 +: 32] = current_trace[BASE+32 +: 32];
            assign in_a_norm[pe*NORM_WIDTH +: NORM_WIDTH] =
                current_trace[BASE+64 +: NORM_WIDTH];
            assign in_b_norm[pe*NORM_WIDTH +: NORM_WIDTH] =
                current_trace[BASE+64+NORM_WIDTH +: NORM_WIDTH];
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

    gtsu_shared_dot_geometry #(
        .PE_COUNT(PE_COUNT), .NORM_WIDTH(NORM_WIDTH),
        .DIST_WIDTH(DIST_WIDTH), .TAG_WIDTH(TAG_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_ready(in_ready),
        .in_tag(in_tag), .in_mode_geometry(in_mode_geometry),
        .in_a(in_a), .in_b(in_b), .in_a_norm(in_a_norm), .in_b_norm(in_b_norm),
        .out_valid(out_valid), .out_ready(out_ready), .out_tag(out_tag),
        .out_mode_geometry(out_mode_geometry), .out_dot(out_dot),
        .out_distance(out_distance)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            cycle <= 0;
            sent <= 0;
            input_beats <= 0;
            output_lanes <= 0;
            feature_lanes <= 0;
            geometry_lanes <= 0;
            source_backpressure_cycles <= 0;
            output_backpressure_cycles <= 0;
        end else begin
            if (in_valid && in_ready) begin
                $display("TRACE %0d INPUT_ACCEPT %0d -1 0 0", cycle, in_tag);
                sent <= sent + 1;
                input_beats <= input_beats + 1;
            end
            if (in_valid && !in_ready)
                source_backpressure_cycles <= source_backpressure_cycles + 1;
            if (out_valid && !out_ready)
                output_backpressure_cycles <= output_backpressure_cycles + 1;
            if (out_valid && out_ready) begin
                for (lane = 0; lane < PE_COUNT; lane = lane + 1)
                    $display("TRACE %0d OUTPUT_ACCEPT %0d %0d %0d %0d",
                        cycle, out_tag, lane,
                        $signed(out_dot[lane*32 +: 32]),
                        out_distance[lane*DIST_WIDTH +: DIST_WIDTH]);
                output_lanes <= output_lanes + PE_COUNT;
                if (out_mode_geometry)
                    geometry_lanes <= geometry_lanes + PE_COUNT;
                else
                    feature_lanes <= feature_lanes + PE_COUNT;
                if (output_lanes + PE_COUNT == BEATS * PE_COUNT) begin
                    #1;
                    $display("SUMMARY %0d %0d %0d %0d %0d %0d %0d",
                        cycle + 1, input_beats, output_lanes,
                        feature_lanes, geometry_lanes,
                        source_backpressure_cycles, output_backpressure_cycles);
                    $finish;
                end
            end
            cycle <= cycle + 1;
            if (cycle >= MAX_CYCLES)
                $fatal(1, "shared-dot RTL timeout");
        end
    end
endmodule

module tb_gtsu_dense64_dual_output_trace;
    parameter integer BF16_LANES = 16;
    parameter integer TILES = 64;
    parameter integer TRACE_WIDTH = 7664;
    parameter integer MAX_CYCLES = 10000;

    reg clk = 0;
    reg rst_n = 0;
    integer cycle = 0;
    integer source_index = 0;
    integer output_values = 0;
    integer source_backpressure = 0;
    reg [8*1024-1:0] trace_file;
    reg [TRACE_WIDTH-1:0] stimuli [0:TILES-1];

    wire in_valid = rst_n && source_index < TILES;
    wire in_ready;
    wire [TRACE_WIDTH-1:0] stimulus = stimuli[source_index < TILES ? source_index : 0];
    wire [31:0] in_tag = stimulus[0 +: 32];
    wire [63:0] in_mask = stimulus[32 +: 64];
    wire [15:0] in_activation_scale_fp16 = stimulus[96 +: 16];
    wire [2047:0] in_accumulators = stimulus[112 +: 2048];
    wire [2047:0] in_biases = stimulus[2160 +: 2048];
    wire [2047:0] in_multipliers = stimulus[4208 +: 2048];
    wire [383:0] in_right_shifts = stimulus[6256 +: 384];
    wire [1023:0] in_weight_scales_fp16 = stimulus[6640 +: 1024];
    wire out_valid;
    wire [31:0] out_tag;
    wire [63:0] out_mask;
    wire [511:0] out_a8;
    wire [1023:0] out_bf16;
    wire [31:0] count_input_tiles;
    wire [31:0] count_a8_values;
    wire [31:0] count_bf16_groups;
    wire [31:0] count_bf16_values;
    wire [31:0] count_output_tiles;

    always #5 clk = ~clk;
    initial begin
        if (!$value$plusargs("TRACE_FILE=%s", trace_file))
            $fatal(1, "TRACE_FILE is required");
        $readmemh(trace_file, stimuli);
        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1;
    end

    gtsu_dense64_dual_output #(.BF16_LANES(BF16_LANES)) dut (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_ready(in_ready),
        .in_tag(in_tag), .in_mask(in_mask), .in_accumulators(in_accumulators),
        .in_biases(in_biases), .in_multipliers(in_multipliers),
        .in_right_shifts(in_right_shifts),
        .in_activation_scale_fp16(in_activation_scale_fp16),
        .in_weight_scales_fp16(in_weight_scales_fp16),
        .out_valid(out_valid), .out_ready(1'b1), .out_tag(out_tag),
        .out_mask(out_mask), .out_a8(out_a8), .out_bf16(out_bf16),
        .idle(), .count_input_tiles(count_input_tiles),
        .count_a8_values(count_a8_values),
        .count_bf16_groups(count_bf16_groups),
        .count_bf16_values(count_bf16_values),
        .count_output_tiles(count_output_tiles)
    );

    integer lane;
    integer column;
    always @(posedge clk) begin
        if (!rst_n) begin
            cycle <= 0;
            source_index <= 0;
            output_values <= 0;
            source_backpressure <= 0;
        end else begin
            if (in_valid && in_ready) begin
                $display("TRACE %0d INPUT_ACCEPT %0d 0", cycle, in_tag);
                source_index <= source_index + 1;
            end
            if (in_valid && !in_ready)
                source_backpressure <= source_backpressure + 1;
            if (dut.bf16_issue && &dut.bf16_in_ready)
                $display("TRACE %0d BF16_GROUP_ISSUE %0d %0d", cycle,
                    dut.tile_tag, dut.bf16_issue_group);
            if (&dut.bf16_out_valid)
                for (lane = 0; lane < BF16_LANES; lane = lane + 1)
                    $display("TRACE %0d BF16_VALUE %0d %04h", cycle,
                        dut.tile_tag*64+dut.bf16_out_tag[lane*6 +: 6],
                        dut.bf16_out_data[lane*16 +: 16]);
            if (out_valid) begin
                for (column = 0; column < 64; column = column + 1)
                    if (out_mask[column]) begin
                        $display("TRACE %0d A8_OUTPUT %0d %0d", cycle,
                            out_tag*64+column, $signed(out_a8[column*8 +: 8]));
                        $display("TRACE %0d BF16_OUTPUT %0d %04h", cycle,
                            out_tag*64+column, out_bf16[column*16 +: 16]);
                        output_values <= output_values + 1;
                    end
                if (count_output_tiles + 1 == TILES) begin
                    #1;
                    $display("SUMMARY %0d %0d %0d %0d %0d %0d %0d 0",
                        cycle+1, count_input_tiles, count_a8_values,
                        count_bf16_groups, count_bf16_values,
                        count_output_tiles, source_backpressure);
                    $finish;
                end
            end
            cycle <= cycle + 1;
            if (cycle >= MAX_CYCLES)
                $fatal(1, "real Dense dual-output timeout");
        end
    end
endmodule

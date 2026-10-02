module tb_gtsu_dense64_dual_output;
    parameter integer BF16_LANES = 16;
    parameter integer TILES = 3;
    parameter integer SOURCE_STALL_MOD = 5;
    parameter integer SOURCE_STALL_PHASE = 2;
    parameter integer OUTPUT_STALL_MOD = 7;
    parameter integer OUTPUT_STALL_PHASE = 3;
    parameter integer MAX_CYCLES = 10000;

    reg clk = 0;
    reg rst_n = 0;
    integer cycle = 0;
    integer source_index = 0;
    integer source_backpressure = 0;
    integer output_backpressure = 0;

    wire source_gate = SOURCE_STALL_MOD == 0 ? 1'b1
        : cycle % SOURCE_STALL_MOD != SOURCE_STALL_PHASE;
    wire in_valid = rst_n && source_index < TILES && source_gate;
    wire in_ready;
    wire [31:0] in_tag = source_index;
    wire [63:0] in_mask = source_index + 1 < TILES
        ? 64'hffffffffffffffff : 64'h07ffffffffffffff;
    wire [2047:0] in_accumulators = make_accumulators(source_index);
    wire [2047:0] in_biases = make_biases(1'b0);
    wire [2047:0] in_multipliers = make_multipliers(1'b0);
    wire [383:0] in_right_shifts = make_shifts(1'b0);
    wire [15:0] in_activation_scale_fp16 = 16'h226f;
    wire [1023:0] in_weight_scales_fp16 = make_weight_scales(1'b0);
    wire output_gate = OUTPUT_STALL_MOD == 0 ? 1'b1
        : cycle % OUTPUT_STALL_MOD != OUTPUT_STALL_PHASE;
    wire out_valid;
    wire out_ready = rst_n && output_gate;
    wire [31:0] out_tag;
    wire [63:0] out_mask;
    wire [511:0] out_a8;
    wire [1023:0] out_bf16;
    wire idle;
    wire [31:0] count_input_tiles;
    wire [31:0] count_a8_values;
    wire [31:0] count_bf16_groups;
    wire [31:0] count_bf16_values;
    wire [31:0] count_output_tiles;

    function automatic [2047:0] make_accumulators;
        input integer tile;
        integer column;
        integer value;
        begin
            make_accumulators = 0;
            for (column = 0; column < 64; column = column + 1) begin
                value = ((tile*65537 + column*7919 + 12345) % 2000001) - 1000000;
                make_accumulators[column*32 +: 32] = value;
            end
        end
    endfunction

    function automatic [2047:0] make_biases;
        input unused;
        integer column;
        integer value;
        begin
            make_biases = 0;
            for (column = 0; column < 64; column = column + 1) begin
                value = (column % 7) - 3;
                make_biases[column*32 +: 32] = value;
            end
        end
    endfunction

    function automatic [2047:0] make_multipliers;
        input unused;
        integer column;
        begin
            make_multipliers = 0;
            for (column = 0; column < 64; column = column + 1)
                make_multipliers[column*32 +: 32] = 32'h40000000 + column*1024;
        end
    endfunction

    function automatic [383:0] make_shifts;
        input unused;
        integer column;
        begin
            make_shifts = 0;
            for (column = 0; column < 64; column = column + 1)
                make_shifts[column*6 +: 6] = 31 + column % 3;
        end
    endfunction

    function automatic [1023:0] make_weight_scales;
        input unused;
        integer column;
        begin
            make_weight_scales = 0;
            for (column = 0; column < 64; column = column + 1)
                make_weight_scales[column*16 +: 16] = 16'h2000 + column*3;
        end
    endfunction

    always #5 clk = ~clk;
    initial begin
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
        .out_valid(out_valid), .out_ready(out_ready), .out_tag(out_tag),
        .out_mask(out_mask), .out_a8(out_a8), .out_bf16(out_bf16),
        .idle(idle), .count_input_tiles(count_input_tiles),
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
            source_backpressure <= 0;
            output_backpressure <= 0;
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
            if (out_valid && !out_ready)
                output_backpressure <= output_backpressure + 1;
            if (out_valid && out_ready) begin
                for (column = 0; column < 64; column = column + 1)
                    if (out_mask[column]) begin
                        $display("TRACE %0d A8_OUTPUT %0d %0d", cycle,
                            out_tag*64+column, $signed(out_a8[column*8 +: 8]));
                        $display("TRACE %0d BF16_OUTPUT %0d %04h", cycle,
                            out_tag*64+column, out_bf16[column*16 +: 16]);
                    end
                if (count_output_tiles + 1 == TILES) begin
                    #1;
                    $display("SUMMARY %0d %0d %0d %0d %0d %0d %0d %0d",
                        cycle+1, count_input_tiles, count_a8_values,
                        count_bf16_groups, count_bf16_values,
                        count_output_tiles, source_backpressure,
                        output_backpressure);
                    $finish;
                end
            end
            cycle <= cycle + 1;
            if (cycle >= MAX_CYCLES)
                $fatal(1, "Dense dual-output timeout");
        end
    end
endmodule

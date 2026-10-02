// Dense64 post-processing: 64 parallel A8 requant lanes and a configurable
// number of time-multiplexed BF16 dequant lanes.
module gtsu_dense64_dual_output #(
    parameter integer BF16_LANES = 16,
    parameter integer BF16_GROUPS = 64/BF16_LANES
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         in_valid,
    output wire                         in_ready,
    input  wire [31:0]                  in_tag,
    input  wire [63:0]                  in_mask,
    input  wire [2047:0]                in_accumulators,
    input  wire [2047:0]                in_biases,
    input  wire [2047:0]                in_multipliers,
    input  wire [383:0]                 in_right_shifts,
    input  wire [15:0]                  in_activation_scale_fp16,
    input  wire [1023:0]                in_weight_scales_fp16,
    output wire                         out_valid,
    input  wire                         out_ready,
    output wire [31:0]                  out_tag,
    output wire [63:0]                  out_mask,
    output wire [511:0]                 out_a8,
    output wire [1023:0]                out_bf16,
    output wire                         idle,
    output reg  [31:0]                  count_input_tiles,
    output reg  [31:0]                  count_a8_values,
    output reg  [31:0]                  count_bf16_groups,
    output reg  [31:0]                  count_bf16_values,
    output reg  [31:0]                  count_output_tiles
);
    reg active;
    reg [31:0] tile_tag;
    reg [63:0] tile_mask;
    reg [2047:0] accumulator_buffer;
    reg [1023:0] weight_scale_buffer;
    reg [15:0] activation_scale_buffer;
    reg [31:0] bf16_issue_group;
    reg [1023:0] bf16_output_buffer;

    reg output_valid_reg;
    reg [31:0] output_tag_reg;
    reg [63:0] output_mask_reg;
    reg [511:0] a8_output_buffer;
    assign out_valid = output_valid_reg;
    assign out_tag = output_tag_reg;
    assign out_mask = output_mask_reg;
    assign out_a8 = a8_output_buffer;
    assign out_bf16 = bf16_output_buffer;
    assign idle = !active && !output_valid_reg;

    wire [63:0] a8_in_ready;
    wire [63:0] a8_out_valid;
    wire [511:0] a8_out_data;
    assign in_ready = !active && !output_valid_reg && &a8_in_ready;
    wire input_fire = in_valid && in_ready;

    genvar a8_lane;
    generate
        for (a8_lane = 0; a8_lane < 64; a8_lane = a8_lane + 1) begin : a8_lanes
            wire [31:0] unused_tag;
            gtsu_requantize_int32 #(.TAG_WIDTH(32)) requant (
                .clk(clk), .rst_n(rst_n), .in_valid(input_fire),
                .in_ready(a8_in_ready[a8_lane]), .in_tag(in_tag),
                .in_accumulator(in_accumulators[a8_lane*32 +: 32]),
                .in_bias(in_biases[a8_lane*32 +: 32]),
                .in_multiplier(in_multipliers[a8_lane*32 +: 32]),
                .in_right_shift(in_right_shifts[a8_lane*6 +: 6]),
                .out_valid(a8_out_valid[a8_lane]), .out_ready(1'b1),
                .out_tag(unused_tag), .out_data(a8_out_data[a8_lane*8 +: 8])
            );
        end
    endgenerate

    wire bf16_issue = active && bf16_issue_group < BF16_GROUPS;
    wire [BF16_LANES-1:0] bf16_in_ready;
    wire [BF16_LANES-1:0] bf16_out_valid;
    wire [BF16_LANES*16-1:0] bf16_out_data;
    wire [BF16_LANES*6-1:0] bf16_out_tag;
    genvar bf16_lane;
    generate
        for (bf16_lane = 0; bf16_lane < BF16_LANES; bf16_lane = bf16_lane + 1) begin : bf16_lanes
            wire [5:0] column = bf16_issue_group*BF16_LANES + bf16_lane;
            gtsu_dequant_int32_bf16 #(.TAG_WIDTH(6)) dequant (
                .clk(clk), .rst_n(rst_n),
                .in_valid(bf16_issue && &bf16_in_ready),
                .in_ready(bf16_in_ready[bf16_lane]), .in_tag(column),
                .in_accumulator(accumulator_buffer[column*32 +: 32]),
                .in_activation_scale_fp16(activation_scale_buffer),
                .in_weight_scale_fp16(weight_scale_buffer[column*16 +: 16]),
                .out_valid(bf16_out_valid[bf16_lane]), .out_ready(1'b1),
                .out_tag(bf16_out_tag[bf16_lane*6 +: 6]),
                .out_bf16(bf16_out_data[bf16_lane*16 +: 16])
            );
        end
    endgenerate

    integer index;
    integer column_index;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active <= 0;
            tile_tag <= 0;
            tile_mask <= 0;
            accumulator_buffer <= 0;
            weight_scale_buffer <= 0;
            activation_scale_buffer <= 0;
            bf16_issue_group <= 0;
            bf16_output_buffer <= 0;
            output_valid_reg <= 0;
            output_tag_reg <= 0;
            output_mask_reg <= 0;
            a8_output_buffer <= 0;
            count_input_tiles <= 0;
            count_a8_values <= 0;
            count_bf16_groups <= 0;
            count_bf16_values <= 0;
            count_output_tiles <= 0;
        end else begin
            if (output_valid_reg && out_ready) begin
                output_valid_reg <= 0;
                count_output_tiles <= count_output_tiles + 1;
            end
            if (input_fire) begin
                active <= 1;
                tile_tag <= in_tag;
                tile_mask <= in_mask;
                accumulator_buffer <= in_accumulators;
                weight_scale_buffer <= in_weight_scales_fp16;
                activation_scale_buffer <= in_activation_scale_fp16;
                bf16_issue_group <= 0;
                count_input_tiles <= count_input_tiles + 1;
            end
            if (&a8_out_valid) begin
                a8_output_buffer <= a8_out_data;
                count_a8_values <= count_a8_values + 64;
            end
            if (bf16_issue && &bf16_in_ready) begin
                bf16_issue_group <= bf16_issue_group + 1;
                count_bf16_groups <= count_bf16_groups + 1;
            end
            if (&bf16_out_valid) begin
                for (index = 0; index < BF16_LANES; index = index + 1) begin
                    column_index = bf16_out_tag[index*6 +: 6];
                    bf16_output_buffer[column_index*16 +: 16]
                        <= bf16_out_data[index*16 +: 16];
                end
                count_bf16_values <= count_bf16_values + BF16_LANES;
                if (bf16_out_tag[(BF16_LANES-1)*6 +: 6] == 63) begin
                    active <= 0;
                    output_valid_reg <= 1;
                    output_tag_reg <= tile_tag;
                    output_mask_reg <= tile_mask;
                end
            end
        end
    end

    initial begin
        if (!(BF16_LANES == 8 || BF16_LANES == 16
              || BF16_LANES == 32 || BF16_LANES == 64))
            $error("BF16_LANES must be 8, 16, 32, or 64");
    end
endmodule

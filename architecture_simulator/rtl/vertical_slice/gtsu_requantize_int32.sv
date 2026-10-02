// Clean-room PointLLM requant implementation. Pipeline ordering and RNE behavior
// were audited against Compiler_Codes@ad2c31a; no reference source is copied.
// Intentional differences: arbitrary multiplier/shift and symmetric [-127,127].
module gtsu_requantize_int32 #(
    parameter integer TAG_WIDTH = 16
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         in_valid,
    output wire                         in_ready,
    input  wire [TAG_WIDTH-1:0]         in_tag,
    input  wire signed [31:0]           in_accumulator,
    input  wire signed [31:0]           in_bias,
    input  wire [31:0]                  in_multiplier,
    input  wire [5:0]                   in_right_shift,
    output wire                         out_valid,
    input  wire                         out_ready,
    output wire [TAG_WIDTH-1:0]         out_tag,
    output wire signed [7:0]            out_data
);
    reg out_valid_reg;
    reg [TAG_WIDTH-1:0] out_tag_reg;
    reg signed [7:0] out_data_reg;

    assign in_ready = !out_valid_reg || out_ready;
    assign out_valid = out_valid_reg;
    assign out_tag = out_tag_reg;
    assign out_data = out_data_reg;

    function automatic signed [7:0] requantize;
        input signed [31:0] accumulator;
        input signed [31:0] bias;
        input [31:0] multiplier;
        input [5:0] right_shift;
        reg signed [32:0] biased;
        reg signed [65:0] product;
        reg [65:0] magnitude;
        reg [65:0] truncated;
        reg [65:0] remainder;
        reg [65:0] half;
        reg round_up;
        reg signed [66:0] rounded;
        begin
            biased = {accumulator[31], accumulator} + {bias[31], bias};
            product = biased * $signed({1'b0, multiplier});
            magnitude = product[65] ? -product : product;
            if (right_shift == 0) begin
                truncated = magnitude;
                round_up = 1'b0;
            end else begin
                truncated = magnitude >> right_shift;
                remainder = magnitude & ((66'd1 << right_shift) - 1'b1);
                half = 66'd1 << (right_shift - 1'b1);
                round_up = (remainder > half)
                    || ((remainder == half) && truncated[0]);
            end
            rounded = product[65]
                ? -$signed({1'b0, truncated + round_up})
                :  $signed({1'b0, truncated + round_up});
            if (rounded > 127)
                requantize = 8'sd127;
            else if (rounded < -127)
                requantize = -8'sd127;
            else
                requantize = rounded[7:0];
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid_reg <= 1'b0;
            out_tag_reg <= 0;
            out_data_reg <= 0;
        end else begin
            if (out_valid_reg && out_ready)
                out_valid_reg <= 1'b0;
            if (in_valid && in_ready) begin
                out_valid_reg <= 1'b1;
                out_tag_reg <= in_tag;
                out_data_reg <= requantize(
                    in_accumulator, in_bias, in_multiplier, in_right_shift
                );
            end
        end
    end
endmodule

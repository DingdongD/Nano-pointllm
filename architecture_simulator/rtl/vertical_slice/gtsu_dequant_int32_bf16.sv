// Clean-room exact dequantization for PointLLM's integer accumulator path.
// Computes INT32 * FP16 * FP16 and rounds the exact product directly to BF16
// with round-to-nearest, ties-to-even. No Verilog real/DPI arithmetic is used.
module gtsu_dequant_int32_bf16 #(
    parameter integer TAG_WIDTH = 16
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         in_valid,
    output wire                         in_ready,
    input  wire [TAG_WIDTH-1:0]         in_tag,
    input  wire signed [31:0]           in_accumulator,
    input  wire [15:0]                  in_activation_scale_fp16,
    input  wire [15:0]                  in_weight_scale_fp16,
    output wire                         out_valid,
    input  wire                         out_ready,
    output wire [TAG_WIDTH-1:0]         out_tag,
    output wire [15:0]                  out_bf16
);
    reg out_valid_reg;
    reg [TAG_WIDTH-1:0] out_tag_reg;
    reg [15:0] out_bf16_reg;

    assign in_ready = !out_valid_reg || out_ready;
    assign out_valid = out_valid_reg;
    assign out_tag = out_tag_reg;
    assign out_bf16 = out_bf16_reg;

    function automatic [15:0] dequantize;
        input signed [31:0] accumulator;
        input [15:0] scale_a;
        input [15:0] scale_w;
        reg sign_result;
        reg [4:0] exp_a_bits;
        reg [4:0] exp_w_bits;
        reg [10:0] mant_a;
        reg [10:0] mant_w;
        reg signed [8:0] exp_a;
        reg signed [8:0] exp_w;
        reg [32:0] acc_magnitude;
        reg [63:0] product;
        reg [63:0] truncated;
        reg [63:0] remainder;
        reg [63:0] half;
        reg [63:0] significand;
        reg [7:0] exponent_out;
        integer top;
        integer shift;
        integer unbiased;
        integer index;
        begin
            exp_a_bits = scale_a[14:10];
            exp_w_bits = scale_w[14:10];
            mant_a = (exp_a_bits == 0) ? {1'b0, scale_a[9:0]}
                                           : {1'b1, scale_a[9:0]};
            mant_w = (exp_w_bits == 0) ? {1'b0, scale_w[9:0]}
                                           : {1'b1, scale_w[9:0]};
            exp_a = (exp_a_bits == 0) ? -24 : $signed({1'b0, exp_a_bits}) - 25;
            exp_w = (exp_w_bits == 0) ? -24 : $signed({1'b0, exp_w_bits}) - 25;
            sign_result = accumulator[31] ^ scale_a[15] ^ scale_w[15];
            acc_magnitude = accumulator[31]
                ? -$signed({accumulator[31], accumulator})
                :  $signed({accumulator[31], accumulator});
            product = acc_magnitude * mant_a * mant_w;
            if (accumulator == 0 || mant_a == 0 || mant_w == 0) begin
                dequantize = 16'h0000;
            end else if (exp_a_bits == 5'h1f || exp_w_bits == 5'h1f) begin
                dequantize = {sign_result, 8'hff, 7'h00};
            end else begin
                top = -1;
                for (index = 63; index >= 0; index = index - 1)
                    if (top < 0 && product[index])
                        top = index;
                unbiased = top + exp_a + exp_w;
                if (unbiased < -126) begin
                    shift = -(exp_a + exp_w + 133);
                    if (shift <= 0)
                        significand = product << (-shift);
                    else if (shift >= 64)
                        significand = 0;
                    else begin
                        truncated = product >> shift;
                        remainder = product & ((64'd1 << shift) - 1'b1);
                        half = 64'd1 << (shift - 1);
                        significand = truncated + ((remainder > half)
                            || ((remainder == half) && truncated[0]));
                    end
                    if (significand == 0)
                        dequantize = {sign_result, 15'h0000};
                    else if (significand >= 128)
                        dequantize = {sign_result, 8'h01, 7'h00};
                    else
                        dequantize = {sign_result, 8'h00, significand[6:0]};
                end else begin
                    shift = top - 7;
                    if (shift <= 0)
                        significand = product << (-shift);
                    else begin
                        truncated = product >> shift;
                        remainder = product & ((64'd1 << shift) - 1'b1);
                        half = 64'd1 << (shift - 1);
                        significand = truncated + ((remainder > half)
                            || ((remainder == half) && truncated[0]));
                    end
                    if (significand >= 256) begin
                        significand = significand >> 1;
                        unbiased = unbiased + 1;
                    end
                    if (unbiased > 127)
                        dequantize = {sign_result, 8'hff, 7'h00};
                    else begin
                        exponent_out = unbiased + 127;
                        dequantize = {sign_result, exponent_out,
                                      significand[6:0]};
                    end
                end
            end
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid_reg <= 1'b0;
            out_tag_reg <= 0;
            out_bf16_reg <= 0;
        end else begin
            if (out_valid_reg && out_ready)
                out_valid_reg <= 1'b0;
            if (in_valid && in_ready) begin
                out_valid_reg <= 1'b1;
                out_tag_reg <= in_tag;
                out_bf16_reg <= dequantize(
                    in_accumulator, in_activation_scale_fp16,
                    in_weight_scale_fp16
                );
            end
        end
    end
endmodule

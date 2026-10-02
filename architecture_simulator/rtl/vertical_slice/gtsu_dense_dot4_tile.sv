module gtsu_dense_dot4_tile #(
    parameter integer N_TILE = 8,
    parameter integer TAG_WIDTH = 16
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         in_valid,
    output wire                         in_ready,
    input  wire [TAG_WIDTH-1:0]         in_tag,
    input  wire                         in_first,
    input  wire                         in_last,
    input  wire [N_TILE-1:0]            in_column_mask,
    input  wire [31:0]                  in_a,
    input  wire [N_TILE*32-1:0]         in_b,
    output wire                         out_valid,
    input  wire                         out_ready,
    output wire [TAG_WIDTH-1:0]         out_tag,
    output wire [N_TILE-1:0]            out_column_mask,
    output wire [N_TILE*32-1:0]         out_accumulator
);
    wire signed [31:0] dot [0:N_TILE-1];
    reg signed [31:0] accumulator [0:N_TILE-1];
    reg out_valid_reg;
    reg [TAG_WIDTH-1:0] out_tag_reg;
    reg [N_TILE-1:0] out_mask_reg;
    reg [N_TILE*32-1:0] out_accumulator_reg;

    genvar column;
    generate
        for (column = 0; column < N_TILE; column = column + 1) begin : dense_pe
            gtsu_dot4_pe dot_pe (
                .a_data(in_a),
                .b_data(in_b[column*32 +: 32]),
                .dot_data(dot[column])
            );
        end
    endgenerate

    assign in_ready = !out_valid_reg || out_ready;
    assign out_valid = out_valid_reg;
    assign out_tag = out_tag_reg;
    assign out_column_mask = out_mask_reg;
    assign out_accumulator = out_accumulator_reg;

    integer index;
    reg signed [31:0] next_accumulator;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid_reg <= 1'b0;
            out_tag_reg <= 0;
            out_mask_reg <= 0;
            out_accumulator_reg <= 0;
            for (index = 0; index < N_TILE; index = index + 1)
                accumulator[index] <= 0;
        end else begin
            if (out_valid_reg && out_ready)
                out_valid_reg <= 1'b0;
            if (in_valid && in_ready) begin
                for (index = 0; index < N_TILE; index = index + 1) begin
                    next_accumulator = in_first
                        ? dot[index] : accumulator[index] + dot[index];
                    accumulator[index] <= next_accumulator;
                    if (in_last)
                        out_accumulator_reg[index*32 +: 32] <= next_accumulator;
                end
                if (in_last) begin
                    out_valid_reg <= 1'b1;
                    out_tag_reg <= in_tag;
                    out_mask_reg <= in_column_mask;
                end
            end
        end
    end

    initial begin
        if (N_TILE <= 0)
            $error("N_TILE must be positive");
    end
endmodule

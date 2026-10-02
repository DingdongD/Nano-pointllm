module gtsu_shared_dot_geometry #(
    parameter integer PE_COUNT = 8,
    parameter integer NORM_WIDTH = 16,
    parameter integer DIST_WIDTH = 18,
    parameter integer TAG_WIDTH = 16
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              in_valid,
    output wire                              in_ready,
    input  wire [TAG_WIDTH-1:0]              in_tag,
    input  wire                              in_mode_geometry,
    input  wire [PE_COUNT*32-1:0]            in_a,
    input  wire [PE_COUNT*32-1:0]            in_b,
    input  wire [PE_COUNT*NORM_WIDTH-1:0]    in_a_norm,
    input  wire [PE_COUNT*NORM_WIDTH-1:0]    in_b_norm,

    output wire                              out_valid,
    input  wire                              out_ready,
    output wire [TAG_WIDTH-1:0]              out_tag,
    output wire                              out_mode_geometry,
    output wire [PE_COUNT*32-1:0]            out_dot,
    output wire [PE_COUNT*DIST_WIDTH-1:0]    out_distance
);
    reg out_valid_reg;
    reg [TAG_WIDTH-1:0] out_tag_reg;
    reg out_mode_reg;
    reg [PE_COUNT*32-1:0] out_dot_reg;
    reg [PE_COUNT*DIST_WIDTH-1:0] out_distance_reg;
    wire signed [31:0] pe_dot [0:PE_COUNT-1];
    wire [DIST_WIDTH-1:0] pe_distance [0:PE_COUNT-1];

    genvar pe;
    generate
        for (pe = 0; pe < PE_COUNT; pe = pe + 1) begin : shared_pe
            wire signed [32:0] dot_wide = {pe_dot[pe][31], pe_dot[pe]};
            wire signed [32:0] a_norm_wide =
                {{(33-NORM_WIDTH){1'b0}}, in_a_norm[pe*NORM_WIDTH +: NORM_WIDTH]};
            wire signed [32:0] b_norm_wide =
                {{(33-NORM_WIDTH){1'b0}}, in_b_norm[pe*NORM_WIDTH +: NORM_WIDTH]};
            wire signed [32:0] distance_wide =
                a_norm_wide + b_norm_wide - (dot_wide <<< 1);

            gtsu_dot4_pe dot_pe (
                .a_data(in_a[pe*32 +: 32]),
                .b_data(in_b[pe*32 +: 32]),
                .dot_data(pe_dot[pe])
            );
            assign pe_distance[pe] = distance_wide[DIST_WIDTH-1:0];
        end
    endgenerate

    assign in_ready = !out_valid_reg || out_ready;
    assign out_valid = out_valid_reg;
    assign out_tag = out_tag_reg;
    assign out_mode_geometry = out_mode_reg;
    assign out_dot = out_dot_reg;
    assign out_distance = out_distance_reg;

    integer lane;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid_reg <= 1'b0;
            out_tag_reg <= 0;
            out_mode_reg <= 1'b0;
            out_dot_reg <= 0;
            out_distance_reg <= 0;
        end else if (in_ready) begin
            out_valid_reg <= in_valid;
            if (in_valid) begin
                out_tag_reg <= in_tag;
                out_mode_reg <= in_mode_geometry;
                for (lane = 0; lane < PE_COUNT; lane = lane + 1) begin
                    out_dot_reg[lane*32 +: 32] <= pe_dot[lane];
                    out_distance_reg[lane*DIST_WIDTH +: DIST_WIDTH] <=
                        in_mode_geometry ? pe_distance[lane] : 0;
                end
            end
        end
    end

    initial begin
        if (PE_COUNT <= 0)
            $error("PE_COUNT must be positive");
        if (NORM_WIDTH <= 0 || NORM_WIDTH >= 33)
            $error("NORM_WIDTH must be between 1 and 32");
    end
endmodule

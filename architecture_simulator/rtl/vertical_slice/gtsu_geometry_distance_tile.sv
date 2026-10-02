module gtsu_geometry_distance_tile #(
    parameter integer LANES = 8,
    parameter integer COORD_WIDTH = 16,
    parameter integer DIST_WIDTH = 36,
    parameter integer TAG_WIDTH = 16
) (
    input  wire                                 clk,
    input  wire                                 rst_n,
    input  wire                                 in_valid,
    output wire                                 in_ready,
    input  wire [TAG_WIDTH-1:0]                 in_tag,
    input  wire signed [COORD_WIDTH-1:0]        in_q_x,
    input  wire signed [COORD_WIDTH-1:0]        in_q_y,
    input  wire signed [COORD_WIDTH-1:0]        in_q_z,
    input  wire [LANES*COORD_WIDTH-1:0]         in_point_x,
    input  wire [LANES*COORD_WIDTH-1:0]         in_point_y,
    input  wire [LANES*COORD_WIDTH-1:0]         in_point_z,
    output reg                                  out_valid,
    input  wire                                 out_ready,
    output reg [TAG_WIDTH-1:0]                  out_tag,
    output reg [LANES*DIST_WIDTH-1:0]           out_distance
);
    localparam integer DIFF_WIDTH = COORD_WIDTH + 1;
    localparam integer SQUARE_WIDTH = 2 * DIFF_WIDTH;

    wire push = in_valid && in_ready;
    wire pop = out_valid && out_ready;
    assign in_ready = !out_valid || out_ready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_tag <= {TAG_WIDTH{1'b0}};
        end else begin
            if (push) begin
                out_valid <= 1'b1;
                out_tag <= in_tag;
            end else if (pop) begin
                out_valid <= 1'b0;
            end
        end
    end

    genvar lane;
    generate
        for (lane = 0; lane < LANES; lane = lane + 1) begin : gen_lane
            wire signed [COORD_WIDTH-1:0] point_x;
            wire signed [COORD_WIDTH-1:0] point_y;
            wire signed [COORD_WIDTH-1:0] point_z;
            wire signed [DIFF_WIDTH-1:0] diff_x;
            wire signed [DIFF_WIDTH-1:0] diff_y;
            wire signed [DIFF_WIDTH-1:0] diff_z;
            wire signed [SQUARE_WIDTH-1:0] square_x_signed;
            wire signed [SQUARE_WIDTH-1:0] square_y_signed;
            wire signed [SQUARE_WIDTH-1:0] square_z_signed;
            wire [DIST_WIDTH-1:0] distance_sum;

            assign point_x = in_point_x[lane*COORD_WIDTH +: COORD_WIDTH];
            assign point_y = in_point_y[lane*COORD_WIDTH +: COORD_WIDTH];
            assign point_z = in_point_z[lane*COORD_WIDTH +: COORD_WIDTH];
            assign diff_x = $signed({in_q_x[COORD_WIDTH-1], in_q_x})
                          - $signed({point_x[COORD_WIDTH-1], point_x});
            assign diff_y = $signed({in_q_y[COORD_WIDTH-1], in_q_y})
                          - $signed({point_y[COORD_WIDTH-1], point_y});
            assign diff_z = $signed({in_q_z[COORD_WIDTH-1], in_q_z})
                          - $signed({point_z[COORD_WIDTH-1], point_z});
            assign square_x_signed = diff_x * diff_x;
            assign square_y_signed = diff_y * diff_y;
            assign square_z_signed = diff_z * diff_z;
            assign distance_sum = {{(DIST_WIDTH-SQUARE_WIDTH){1'b0}}, square_x_signed}
                                + {{(DIST_WIDTH-SQUARE_WIDTH){1'b0}}, square_y_signed}
                                + {{(DIST_WIDTH-SQUARE_WIDTH){1'b0}}, square_z_signed};

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    out_distance[lane*DIST_WIDTH +: DIST_WIDTH] <= {DIST_WIDTH{1'b0}};
                else if (push)
                    out_distance[lane*DIST_WIDTH +: DIST_WIDTH] <= distance_sum;
            end
        end
    endgenerate
endmodule

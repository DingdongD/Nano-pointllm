module gtsu_fused_geometry_pipeline #(
    parameter integer POINTS = 16,
    parameter integer CENTERS = 5,
    parameter integer K = 4,
    parameter integer LANES = 4,
    parameter integer NORM_WIDTH = 16,
    parameter integer DIST_WIDTH = 18,
    parameter integer INDEX_WIDTH = (POINTS <= 2) ? 1 : $clog2(POINTS)
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              in_valid,
    output wire                              in_ready,
    input  wire [LANES-1:0]                  in_mask,
    input  wire [LANES*32-1:0]               in_query_vectors,
    input  wire [LANES*32-1:0]               in_point_vectors,
    input  wire [LANES*NORM_WIDTH-1:0]       in_query_norms,
    input  wire [LANES*NORM_WIDTH-1:0]       in_point_norms,
    output wire                              out_valid,
    input  wire                              out_ready,
    output wire [INDEX_WIDTH-1:0]            out_center,
    output wire [INDEX_WIDTH-1:0]            out_next_center,
    output wire [K*INDEX_WIDTH-1:0]          out_neighbor_indices,
    output wire [K*DIST_WIDTH-1:0]           out_neighbor_distances,
    output wire                              done
);
    wire distance_valid, distance_ready;
    wire [LANES*DIST_WIDTH-1:0] distances;
    wire [LANES*32-1:0] unused_dot;
    wire unused_mode;
    wire [15:0] unused_tag;
    reg [LANES-1:0] mask_register;

    gtsu_shared_dot_geometry #(
        .PE_COUNT(LANES), .NORM_WIDTH(NORM_WIDTH),
        .DIST_WIDTH(DIST_WIDTH), .TAG_WIDTH(16)
    ) distance_stage (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_ready(in_ready),
        .in_tag(16'b0), .in_mode_geometry(1'b1),
        .in_a(in_query_vectors), .in_b(in_point_vectors),
        .in_a_norm(in_query_norms), .in_b_norm(in_point_norms),
        .out_valid(distance_valid), .out_ready(distance_ready),
        .out_tag(unused_tag), .out_mode_geometry(unused_mode),
        .out_dot(unused_dot), .out_distance(distances)
    );

    gtsu_fused_fps_knn_controller #(
        .POINTS(POINTS), .CENTERS(CENTERS), .K(K), .LANES(LANES),
        .DIST_WIDTH(DIST_WIDTH)
    ) selection_stage (
        .clk(clk), .rst_n(rst_n), .in_valid(distance_valid),
        .in_ready(distance_ready), .in_mask(mask_register),
        .in_distances(distances), .out_valid(out_valid), .out_ready(out_ready),
        .out_center(out_center), .out_next_center(out_next_center),
        .out_neighbor_indices(out_neighbor_indices),
        .out_neighbor_distances(out_neighbor_distances), .done(done)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            mask_register <= 0;
        else if (in_valid && in_ready)
            mask_register <= in_mask;
    end
endmodule

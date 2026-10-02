module gtsu_fused_fps_knn_controller #(
    parameter integer POINTS = 16,
    parameter integer CENTERS = 5,
    parameter integer K = 4,
    parameter integer LANES = 4,
    parameter integer DIST_WIDTH = 18,
    parameter integer INDEX_WIDTH = (POINTS <= 2) ? 1 : $clog2(POINTS),
    parameter integer TILE_WIDTH = ((POINTS + LANES - 1) / LANES <= 2) ? 1 :
                                   $clog2((POINTS + LANES - 1) / LANES)
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              in_valid,
    output wire                              in_ready,
    input  wire [LANES-1:0]                  in_mask,
    input  wire [LANES*DIST_WIDTH-1:0]       in_distances,
    output wire                              out_valid,
    input  wire                              out_ready,
    output wire [INDEX_WIDTH-1:0]            out_center,
    output wire [INDEX_WIDTH-1:0]            out_next_center,
    output wire [K*INDEX_WIDTH-1:0]          out_neighbor_indices,
    output wire [K*DIST_WIDTH-1:0]           out_neighbor_distances,
    output wire                              done
);
    localparam integer TILES = (POINTS + LANES - 1) / LANES;
    localparam [DIST_WIDTH-1:0] INF = {DIST_WIDTH{1'b1}};

    reg [DIST_WIDTH-1:0] min_state [0:LANES-1][0:TILES-1];
    reg [DIST_WIDTH-1:0] top_distance [0:K-1];
    reg [INDEX_WIDTH-1:0] top_index [0:K-1];
    reg [DIST_WIDTH-1:0] max_distance;
    reg [INDEX_WIDTH-1:0] max_index;
    reg [INDEX_WIDTH-1:0] current_center;
    reg [TILE_WIDTH-1:0] tile_index;
    reg [$clog2(CENTERS+1)-1:0] round_index;
    reg out_valid_reg;
    reg done_reg;
    reg [INDEX_WIDTH-1:0] out_center_reg, out_next_center_reg;
    reg [K*INDEX_WIDTH-1:0] out_indices_reg;
    reg [K*DIST_WIDTH-1:0] out_distances_reg;

    reg [DIST_WIDTH-1:0] updated_min [0:LANES-1];
    reg [DIST_WIDTH-1:0] sort_distance [0:K+LANES-1];
    reg [INDEX_WIDTH-1:0] sort_index [0:K+LANES-1];
    reg [DIST_WIDTH-1:0] max_distance_next;
    reg [INDEX_WIDTH-1:0] max_index_next;
    reg [DIST_WIDTH-1:0] swap_distance;
    reg [INDEX_WIDTH-1:0] swap_index;
    integer lane, slot, pass;
    integer point_number;

    always @* begin
        max_distance_next = max_distance;
        max_index_next = max_index;
        for (slot = 0; slot < K; slot = slot + 1) begin
            sort_distance[slot] = top_distance[slot];
            sort_index[slot] = top_index[slot];
        end
        for (lane = 0; lane < LANES; lane = lane + 1) begin
            point_number = tile_index * LANES + lane;
            if (in_mask[lane]) begin
                if (round_index == 0 ||
                    in_distances[lane*DIST_WIDTH +: DIST_WIDTH] < min_state[lane][tile_index])
                    updated_min[lane] = in_distances[lane*DIST_WIDTH +: DIST_WIDTH];
                else
                    updated_min[lane] = min_state[lane][tile_index];
                sort_distance[K+lane] = in_distances[lane*DIST_WIDTH +: DIST_WIDTH];
                sort_index[K+lane] = point_number[INDEX_WIDTH-1:0];
                if (updated_min[lane] > max_distance_next ||
                    (updated_min[lane] == max_distance_next &&
                     point_number[INDEX_WIDTH-1:0] < max_index_next)) begin
                    max_distance_next = updated_min[lane];
                    max_index_next = point_number[INDEX_WIDTH-1:0];
                end
            end else begin
                updated_min[lane] = INF;
                sort_distance[K+lane] = INF;
                sort_index[K+lane] = {INDEX_WIDTH{1'b1}};
            end
        end
        for (pass = 0; pass < K+LANES; pass = pass + 1) begin
            for (slot = 0; slot < K+LANES-1; slot = slot + 1) begin
                if (sort_distance[slot] > sort_distance[slot+1] ||
                    (sort_distance[slot] == sort_distance[slot+1] &&
                     sort_index[slot] > sort_index[slot+1])) begin
                    swap_distance = sort_distance[slot];
                    swap_index = sort_index[slot];
                    sort_distance[slot] = sort_distance[slot+1];
                    sort_index[slot] = sort_index[slot+1];
                    sort_distance[slot+1] = swap_distance;
                    sort_index[slot+1] = swap_index;
                end
            end
        end
    end

    assign in_ready = !out_valid_reg && !done_reg;
    assign out_valid = out_valid_reg;
    assign out_center = out_center_reg;
    assign out_next_center = out_next_center_reg;
    assign out_neighbor_indices = out_indices_reg;
    assign out_neighbor_distances = out_distances_reg;
    assign done = done_reg;

    integer reset_slot;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_center <= 0;
            tile_index <= 0;
            round_index <= 0;
            max_distance <= 0;
            max_index <= 0;
            out_valid_reg <= 0;
            done_reg <= 0;
            out_center_reg <= 0;
            out_next_center_reg <= 0;
            out_indices_reg <= 0;
            out_distances_reg <= 0;
            for (reset_slot = 0; reset_slot < K; reset_slot = reset_slot + 1) begin
                top_distance[reset_slot] <= INF;
                top_index[reset_slot] <= {INDEX_WIDTH{1'b1}};
            end
        end else begin
            if (out_valid_reg && out_ready) begin
                out_valid_reg <= 0;
                if (round_index + 1 == CENTERS) begin
                    done_reg <= 1;
                end else begin
                    round_index <= round_index + 1'b1;
                    current_center <= out_next_center_reg;
                    max_distance <= 0;
                    max_index <= 0;
                    for (reset_slot = 0; reset_slot < K; reset_slot = reset_slot + 1) begin
                        top_distance[reset_slot] <= INF;
                        top_index[reset_slot] <= {INDEX_WIDTH{1'b1}};
                    end
                end
            end
            if (in_valid && in_ready) begin
                for (lane = 0; lane < LANES; lane = lane + 1)
                    if (in_mask[lane])
                        min_state[lane][tile_index] <= updated_min[lane];
                for (slot = 0; slot < K; slot = slot + 1) begin
                    top_distance[slot] <= sort_distance[slot];
                    top_index[slot] <= sort_index[slot];
                end
                max_distance <= max_distance_next;
                max_index <= max_index_next;
                if (tile_index + 1 == TILES) begin
                    tile_index <= 0;
                    out_valid_reg <= 1;
                    out_center_reg <= current_center;
                    out_next_center_reg <= max_index_next;
                    for (slot = 0; slot < K; slot = slot + 1) begin
                        out_indices_reg[slot*INDEX_WIDTH +: INDEX_WIDTH] <= sort_index[slot];
                        out_distances_reg[slot*DIST_WIDTH +: DIST_WIDTH] <= sort_distance[slot];
                    end
                end else begin
                    tile_index <= tile_index + 1'b1;
                end
            end
        end
    end

    initial begin
        if (POINTS <= 0 || CENTERS <= 0 || K <= 0 || LANES <= 0)
            $error("FPS/KNN dimensions must be positive");
        if (CENTERS > POINTS || K > POINTS)
            $error("CENTERS and K cannot exceed POINTS");
    end
endmodule

module gtsu_w8_splitk_gemv #(
    parameter integer N_OUTPUTS = 4,
    parameter integer SPLIT_K = 2,
    parameter integer K_CHUNKS = 3,
    parameter integer LANES = 4,
    parameter integer COMP_FIFO_DEPTH = 2,
    parameter integer DECODED_FIFO_DEPTH = 2,
    parameter integer PARTIAL_FIFO_DEPTH = 2,
    parameter integer N_W = (N_OUTPUTS <= 1) ? 1 : $clog2(N_OUTPUTS),
    parameter integer P_W = (SPLIT_K <= 1) ? 1 : $clog2(SPLIT_K),
    parameter integer C_W = (K_CHUNKS <= 1) ? 1 : $clog2(K_CHUNKS)
) (
    input  wire                         clk,
    input  wire                         rst_n,

    input  wire                         weight_valid,
    output wire                         weight_ready,
    input  wire [N_W-1:0]               weight_n,
    input  wire [P_W-1:0]               weight_partition,
    input  wire [C_W-1:0]               weight_chunk,
    input  wire [LANES*8-1:0]           weight_data,
    input  wire [LANES*8-1:0]           activation_data,

    output wire                         output_valid,
    input  wire                         output_ready,
    output wire [N_W-1:0]               output_n,
    output wire signed [31:0]           output_data,

    output wire                         event_input_accept,
    output wire                         event_unpack_accept,
    output wire                         event_compute_accept,
    output wire                         event_partial_push,
    output wire                         event_output_accept,
    output wire [N_W-1:0]               event_unpack_n,
    output wire [P_W-1:0]               event_unpack_partition,
    output wire [C_W-1:0]               event_unpack_chunk,
    output wire [N_W-1:0]               event_compute_n,
    output wire [P_W-1:0]               event_compute_partition,
    output wire [C_W-1:0]               event_compute_chunk,
    output wire signed [31:0]           event_compute_dot,
    output wire [N_W-1:0]               event_partial_n,
    output wire [P_W-1:0]               event_partial_partition,
    output wire signed [31:0]           event_partial_data,

    output reg  [31:0]                  count_input_beats,
    output reg  [31:0]                  count_unpack_beats,
    output reg  [31:0]                  count_compute_beats,
    output reg  [31:0]                  count_partial_sums,
    output reg  [31:0]                  count_outputs,
    output reg  [31:0]                  stall_source,
    output reg  [31:0]                  stall_unpack,
    output reg  [31:0]                  stall_compute,
    output reg  [31:0]                  stall_reduction
);
    localparam integer WEIGHT_W = LANES * 8;
    localparam integer ACT_W = LANES * 8;
    localparam integer UNPACKED_W = LANES * 16;
    localparam integer COMP_W = N_W + P_W + C_W + ACT_W + WEIGHT_W;
    localparam integer DECODED_W = N_W + P_W + C_W + ACT_W + UNPACKED_W;
    localparam integer PARTIAL_W = N_W + P_W + 32;

    wire [COMP_W-1:0] comp_in_data = {
        weight_n, weight_partition, weight_chunk, activation_data, weight_data
    };
    wire comp_out_valid;
    wire comp_out_ready;
    wire [COMP_W-1:0] comp_out_data;
    wire [$clog2(COMP_FIFO_DEPTH+1)-1:0] comp_occupancy;

    gtsu_rv_fifo #(
        .WIDTH(COMP_W),
        .DEPTH(COMP_FIFO_DEPTH)
    ) compressed_fifo (
        .clk(clk), .rst_n(rst_n),
        .in_valid(weight_valid), .in_ready(weight_ready), .in_data(comp_in_data),
        .out_valid(comp_out_valid), .out_ready(comp_out_ready), .out_data(comp_out_data),
        .occupancy(comp_occupancy)
    );

    wire [WEIGHT_W-1:0] comp_weight = comp_out_data[WEIGHT_W-1:0];
    wire [ACT_W-1:0] comp_activation = comp_out_data[WEIGHT_W +: ACT_W];
    wire [C_W-1:0] comp_chunk = comp_out_data[WEIGHT_W+ACT_W +: C_W];
    wire [P_W-1:0] comp_partition = comp_out_data[WEIGHT_W+ACT_W+C_W +: P_W];
    wire [N_W-1:0] comp_n = comp_out_data[WEIGHT_W+ACT_W+C_W+P_W +: N_W];

    reg unpack_valid;
    reg [N_W-1:0] unpack_n;
    reg [P_W-1:0] unpack_partition;
    reg [C_W-1:0] unpack_chunk;
    reg [ACT_W-1:0] unpack_activation;
    reg [UNPACKED_W-1:0] unpack_weights;
    wire decoded_in_ready;
    wire unpack_ready = !unpack_valid || decoded_in_ready;
    assign comp_out_ready = unpack_ready;

    integer unpack_lane;
    reg [UNPACKED_W-1:0] sign_extended_weights;
    always @* begin
        sign_extended_weights = 0;
        for (unpack_lane = 0; unpack_lane < LANES; unpack_lane = unpack_lane + 1) begin
            sign_extended_weights[unpack_lane*16 +: 16] = {
                {8{comp_weight[unpack_lane*8 + 7]}},
                comp_weight[unpack_lane*8 +: 8]
            };
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            unpack_valid <= 1'b0;
            unpack_n <= 0;
            unpack_partition <= 0;
            unpack_chunk <= 0;
            unpack_activation <= 0;
            unpack_weights <= 0;
        end else if (unpack_ready) begin
            unpack_valid <= comp_out_valid;
            if (comp_out_valid) begin
                unpack_n <= comp_n;
                unpack_partition <= comp_partition;
                unpack_chunk <= comp_chunk;
                unpack_activation <= comp_activation;
                unpack_weights <= sign_extended_weights;
            end
        end
    end

    wire [DECODED_W-1:0] decoded_in_data = {
        unpack_n, unpack_partition, unpack_chunk, unpack_activation, unpack_weights
    };
    wire decoded_out_valid;
    wire decoded_out_ready;
    wire [DECODED_W-1:0] decoded_out_data;
    wire [$clog2(DECODED_FIFO_DEPTH+1)-1:0] decoded_occupancy;

    gtsu_rv_fifo #(
        .WIDTH(DECODED_W),
        .DEPTH(DECODED_FIFO_DEPTH)
    ) decoded_fifo (
        .clk(clk), .rst_n(rst_n),
        .in_valid(unpack_valid), .in_ready(decoded_in_ready), .in_data(decoded_in_data),
        .out_valid(decoded_out_valid), .out_ready(decoded_out_ready), .out_data(decoded_out_data),
        .occupancy(decoded_occupancy)
    );

    wire [UNPACKED_W-1:0] decoded_weights = decoded_out_data[UNPACKED_W-1:0];
    wire [ACT_W-1:0] decoded_activation = decoded_out_data[UNPACKED_W +: ACT_W];
    wire [C_W-1:0] decoded_chunk = decoded_out_data[UNPACKED_W+ACT_W +: C_W];
    wire [P_W-1:0] decoded_partition = decoded_out_data[UNPACKED_W+ACT_W+C_W +: P_W];
    wire [N_W-1:0] decoded_n = decoded_out_data[UNPACKED_W+ACT_W+C_W+P_W +: N_W];

    integer dot_lane;
    reg signed [15:0] dot_weight;
    reg signed [7:0] dot_activation;
    reg signed [31:0] dot_product;
    always @* begin
        dot_product = 0;
        dot_weight = 0;
        dot_activation = 0;
        for (dot_lane = 0; dot_lane < LANES; dot_lane = dot_lane + 1) begin
            dot_weight = $signed(decoded_weights[dot_lane*16 +: 16]);
            dot_activation = $signed(decoded_activation[dot_lane*8 +: 8]);
            dot_product = dot_product + dot_weight * dot_activation;
        end
    end

    reg signed [31:0] compute_accumulator;
    wire decoded_last = decoded_chunk == K_CHUNKS - 1;
    wire partial_in_ready;
    assign decoded_out_ready = !decoded_last || partial_in_ready;
    wire signed [31:0] next_partial =
        (decoded_chunk == 0) ? dot_product : compute_accumulator + dot_product;
    wire partial_in_valid = decoded_out_valid && decoded_out_ready && decoded_last;
    wire [PARTIAL_W-1:0] partial_in_data = {
        decoded_n, decoded_partition, next_partial
    };

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            compute_accumulator <= 0;
        end else if (decoded_out_valid && decoded_out_ready) begin
            compute_accumulator <= next_partial;
        end
    end

    wire partial_out_valid;
    wire partial_out_ready;
    wire [PARTIAL_W-1:0] partial_out_data;
    wire [$clog2(PARTIAL_FIFO_DEPTH+1)-1:0] partial_occupancy;

    gtsu_rv_fifo #(
        .WIDTH(PARTIAL_W),
        .DEPTH(PARTIAL_FIFO_DEPTH)
    ) partial_fifo (
        .clk(clk), .rst_n(rst_n),
        .in_valid(partial_in_valid), .in_ready(partial_in_ready), .in_data(partial_in_data),
        .out_valid(partial_out_valid), .out_ready(partial_out_ready), .out_data(partial_out_data),
        .occupancy(partial_occupancy)
    );

    wire signed [31:0] partial_value = partial_out_data[31:0];
    wire [P_W-1:0] partial_partition = partial_out_data[32 +: P_W];
    wire [N_W-1:0] partial_n = partial_out_data[32+P_W +: N_W];
    reg signed [31:0] reduction_accumulator;
    reg output_valid_reg;
    reg [N_W-1:0] output_n_reg;
    reg signed [31:0] output_data_reg;
    wire reduction_space = !output_valid_reg || output_ready;
    assign partial_out_ready = reduction_space;
    wire signed [31:0] next_reduction =
        (partial_partition == 0) ? partial_value : reduction_accumulator + partial_value;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reduction_accumulator <= 0;
            output_valid_reg <= 1'b0;
            output_n_reg <= 0;
            output_data_reg <= 0;
        end else begin
            if (output_valid_reg && output_ready)
                output_valid_reg <= 1'b0;
            if (partial_out_valid && partial_out_ready) begin
                reduction_accumulator <= next_reduction;
                if (partial_partition == SPLIT_K - 1) begin
                    output_valid_reg <= 1'b1;
                    output_n_reg <= partial_n;
                    output_data_reg <= next_reduction;
                end
            end
        end
    end

    assign output_valid = output_valid_reg;
    assign output_n = output_n_reg;
    assign output_data = output_data_reg;

    assign event_input_accept = weight_valid && weight_ready;
    assign event_unpack_accept = comp_out_valid && comp_out_ready;
    assign event_compute_accept = decoded_out_valid && decoded_out_ready;
    assign event_partial_push = partial_in_valid && partial_in_ready;
    assign event_output_accept = output_valid && output_ready;
    assign event_unpack_n = comp_n;
    assign event_unpack_partition = comp_partition;
    assign event_unpack_chunk = comp_chunk;
    assign event_compute_n = decoded_n;
    assign event_compute_partition = decoded_partition;
    assign event_compute_chunk = decoded_chunk;
    assign event_compute_dot = dot_product;
    assign event_partial_n = decoded_n;
    assign event_partial_partition = decoded_partition;
    assign event_partial_data = next_partial;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count_input_beats <= 0;
            count_unpack_beats <= 0;
            count_compute_beats <= 0;
            count_partial_sums <= 0;
            count_outputs <= 0;
            stall_source <= 0;
            stall_unpack <= 0;
            stall_compute <= 0;
            stall_reduction <= 0;
        end else begin
            if (event_input_accept) count_input_beats <= count_input_beats + 1;
            if (event_unpack_accept) count_unpack_beats <= count_unpack_beats + 1;
            if (event_compute_accept) count_compute_beats <= count_compute_beats + 1;
            if (event_partial_push) count_partial_sums <= count_partial_sums + 1;
            if (event_output_accept) count_outputs <= count_outputs + 1;
            if (weight_valid && !weight_ready) stall_source <= stall_source + 1;
            if (comp_out_valid && !comp_out_ready) stall_unpack <= stall_unpack + 1;
            if (decoded_out_valid && !decoded_out_ready) stall_compute <= stall_compute + 1;
            if (partial_out_valid && !partial_out_ready) stall_reduction <= stall_reduction + 1;
        end
    end
endmodule

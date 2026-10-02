module gtsu_production_linear_controller #(
    parameter integer N_OUTPUTS = 4096,
    parameter integer SPLIT_K = 16,
    parameter integer SCALE_BURSTS = 4096,
    parameter integer WEIGHT_BURSTS = 262144,
    parameter integer ROB_DEPTH = 32,
    parameter integer SRAM_READ_LATENCY = 3,
    parameter integer UNPACK_LATENCY = 1,
    parameter integer BASE_CHUNKS = 4,
    parameter integer EXTRA_PARTITIONS = 0,
    parameter integer WEIGHT_BANK_OFFSET_WORDS = 8,
    parameter integer N_W = (N_OUTPUTS <= 1) ? 1 : $clog2(N_OUTPUTS),
    parameter integer P_W = (SPLIT_K <= 1) ? 1 : $clog2(SPLIT_K),
    parameter integer ROB_W = (ROB_DEPTH <= 1) ? 1 : $clog2(ROB_DEPTH),
    parameter integer PIPE_STAGES = SRAM_READ_LATENCY + UNPACK_LATENCY + 1
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  dma_valid,
    output wire                  dma_ready,
    input  wire                  dma_is_weight,
    input  wire [31:0]           dma_sequence,
    input  wire [N_W-1:0]        dma_output_row,
    input  wire [P_W-1:0]        dma_partition,
    input  wire [7:0]            dma_chunk,
    input  wire [7:0]            dma_chunks_in_split,

    output wire                  event_dma_accept,
    output wire                  event_read_issue,
    output wire                  event_compute,
    output wire                  event_partial,
    output wire                  event_output,
    output wire                  event_bank_conflict,
    output reg  [31:0]           count_dma_accept,
    output reg  [31:0]           count_scale_bursts,
    output reg  [31:0]           count_weight_bursts,
    output reg  [31:0]           count_read_issue,
    output reg  [31:0]           count_compute,
    output reg  [31:0]           count_partial,
    output reg  [31:0]           count_output,
    output reg  [31:0]           count_bank_conflict,
    output reg  [31:0]           count_ordered_wait,
    output reg  [31:0]           max_rob_occupancy
);
    reg [ROB_DEPTH-1:0] rob_valid;
    reg [N_W-1:0] rob_n [0:ROB_DEPTH-1];
    reg [P_W-1:0] rob_partition [0:ROB_DEPTH-1];
    reg [7:0] rob_chunk [0:ROB_DEPTH-1];
    reg [7:0] rob_chunks [0:ROB_DEPTH-1];
    reg [31:0] next_sequence;
    reg [31:0] rob_occupancy;
    reg read_conflict_cooldown;

    wire [ROB_W-1:0] dma_slot = dma_sequence % ROB_DEPTH;
    wire [ROB_W-1:0] issue_slot = next_sequence % ROB_DEPTH;
    wire issue_available = rob_valid[issue_slot];
    wire scales_ready = count_scale_bursts == SCALE_BURSTS;
    assign dma_ready = !dma_is_weight
        || (rob_occupancy < ROB_DEPTH && !rob_valid[dma_slot]);
    assign event_dma_accept = dma_valid && dma_ready;
    assign event_read_issue = scales_ready && issue_available
        && next_sequence < WEIGHT_BURSTS && !read_conflict_cooldown;

    integer prefix_chunks;
    integer activation_group;
    integer weight_group;
    reg issue_conflict;
    always @* begin
        prefix_chunks = rob_partition[issue_slot] * BASE_CHUNKS;
        if (rob_partition[issue_slot] < EXTRA_PARTITIONS)
            prefix_chunks = prefix_chunks + rob_partition[issue_slot];
        else
            prefix_chunks = prefix_chunks + EXTRA_PARTITIONS;
        activation_group = ((prefix_chunks + rob_chunk[issue_slot]) * 4) % 16;
        weight_group = (WEIGHT_BANK_OFFSET_WORDS + (next_sequence % ROB_DEPTH) * 4) % 16;
        issue_conflict = activation_group == weight_group;
    end
    assign event_bank_conflict = event_read_issue && issue_conflict;

    reg [PIPE_STAGES-1:0] pipe_valid;
    reg [N_W-1:0] pipe_n [0:PIPE_STAGES-1];
    reg [P_W-1:0] pipe_partition [0:PIPE_STAGES-1];
    reg [7:0] pipe_chunk [0:PIPE_STAGES-1];
    reg [7:0] pipe_chunks [0:PIPE_STAGES-1];
    assign event_compute = pipe_valid[PIPE_STAGES-1];
    assign event_partial = event_compute
        && pipe_chunk[PIPE_STAGES-1] == pipe_chunks[PIPE_STAGES-1] - 1;
    assign event_output = event_partial
        && pipe_partition[PIPE_STAGES-1] == SPLIT_K - 1;

    integer stage;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rob_valid <= 0;
            next_sequence <= 0;
            rob_occupancy <= 0;
            read_conflict_cooldown <= 0;
            pipe_valid <= 0;
            count_dma_accept <= 0;
            count_scale_bursts <= 0;
            count_weight_bursts <= 0;
            count_read_issue <= 0;
            count_compute <= 0;
            count_partial <= 0;
            count_output <= 0;
            count_bank_conflict <= 0;
            count_ordered_wait <= 0;
            max_rob_occupancy <= 0;
            for (stage = 0; stage < PIPE_STAGES; stage = stage + 1) begin
                pipe_n[stage] <= 0;
                pipe_partition[stage] <= 0;
                pipe_chunk[stage] <= 0;
                pipe_chunks[stage] <= 0;
            end
        end else begin
            for (stage = PIPE_STAGES - 1; stage > 0; stage = stage - 1) begin
                pipe_valid[stage] <= pipe_valid[stage-1];
                pipe_n[stage] <= pipe_n[stage-1];
                pipe_partition[stage] <= pipe_partition[stage-1];
                pipe_chunk[stage] <= pipe_chunk[stage-1];
                pipe_chunks[stage] <= pipe_chunks[stage-1];
            end
            pipe_valid[0] <= 1'b0;
            read_conflict_cooldown <= 1'b0;

            if (event_read_issue) begin
                if (issue_conflict) begin
                    read_conflict_cooldown <= 1'b1;
                    pipe_valid[0] <= 1'b1;
                    pipe_n[0] <= rob_n[issue_slot];
                    pipe_partition[0] <= rob_partition[issue_slot];
                    pipe_chunk[0] <= rob_chunk[issue_slot];
                    pipe_chunks[0] <= rob_chunks[issue_slot];
                end else begin
                    pipe_valid[1] <= 1'b1;
                    pipe_n[1] <= rob_n[issue_slot];
                    pipe_partition[1] <= rob_partition[issue_slot];
                    pipe_chunk[1] <= rob_chunk[issue_slot];
                    pipe_chunks[1] <= rob_chunks[issue_slot];
                end
                rob_valid[issue_slot] <= 1'b0;
                next_sequence <= next_sequence + 1;
                count_read_issue <= count_read_issue + 1;
                if (issue_conflict)
                    count_bank_conflict <= count_bank_conflict + 1;
            end else if (scales_ready && next_sequence < WEIGHT_BURSTS) begin
                count_ordered_wait <= count_ordered_wait + 1;
            end

            if (event_dma_accept) begin
                count_dma_accept <= count_dma_accept + 1;
                if (dma_is_weight) begin
                    rob_valid[dma_slot] <= 1'b1;
                    rob_n[dma_slot] <= dma_output_row;
                    rob_partition[dma_slot] <= dma_partition;
                    rob_chunk[dma_slot] <= dma_chunk;
                    rob_chunks[dma_slot] <= dma_chunks_in_split;
                    count_weight_bursts <= count_weight_bursts + 1;
                end else begin
                    count_scale_bursts <= count_scale_bursts + 1;
                end
            end

            case ({event_dma_accept && dma_is_weight, event_read_issue})
                2'b10: rob_occupancy <= rob_occupancy + 1;
                2'b01: rob_occupancy <= rob_occupancy - 1;
                default: rob_occupancy <= rob_occupancy;
            endcase
            if (rob_occupancy + (event_dma_accept && dma_is_weight) > max_rob_occupancy)
                max_rob_occupancy <= rob_occupancy + (event_dma_accept && dma_is_weight);

            if (event_compute) count_compute <= count_compute + 1;
            if (event_partial) count_partial <= count_partial + 1;
            if (event_output) count_output <= count_output + 1;
        end
    end
endmodule

module tb_gtsu_production_linear_controller;
    parameter integer N_OUTPUTS = 4;
    parameter integer SPLIT_K = 2;
    parameter integer SCALE_BURSTS = 2;
    parameter integer WEIGHT_BURSTS = 24;
    parameter integer TOTAL_REQUESTS = SCALE_BURSTS + WEIGHT_BURSTS;
    parameter integer ROB_DEPTH = 8;
    parameter integer BASE_CHUNKS = 3;
    parameter integer EXTRA_PARTITIONS = 0;
    parameter integer MAX_CYCLES = 20000000;
    parameter integer N_W = (N_OUTPUTS <= 1) ? 1 : $clog2(N_OUTPUTS);
    parameter integer P_W = (SPLIT_K <= 1) ? 1 : $clog2(SPLIT_K);

    reg clk = 0;
    reg rst_n = 0;
    integer cycle = 0;
    integer trace_index = 0;
    reg [127:0] trace [0:TOTAL_REQUESTS-1];
    reg [1023:0] trace_file;

    wire [31:0] trace_cycle = trace[trace_index][31:0];
    wire dma_is_weight = trace[trace_index][32];
    wire [31:0] dma_sequence = trace[trace_index][64:33];
    wire [N_W-1:0] dma_output_row = trace[trace_index][64+N_W:65];
    wire [P_W-1:0] dma_partition = trace[trace_index][80+P_W:81];
    wire [7:0] dma_chunk = trace[trace_index][96:89];
    wire [7:0] dma_chunks = trace[trace_index][104:97];
    wire dma_valid = rst_n && trace_index < TOTAL_REQUESTS && cycle >= trace_cycle;
    wire dma_ready;
    wire event_dma_accept, event_read_issue, event_compute;
    wire event_partial, event_output, event_bank_conflict;
    wire [31:0] count_dma_accept, count_scale_bursts, count_weight_bursts;
    wire [31:0] count_read_issue, count_compute, count_partial, count_output;
    wire [31:0] count_bank_conflict, count_ordered_wait, max_rob_occupancy;

    always #5 clk = ~clk;
    initial begin
        if (!$value$plusargs("TRACE_FILE=%s", trace_file))
            $fatal(1, "TRACE_FILE plusarg is required");
        $readmemh(trace_file, trace);
        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1;
    end

    gtsu_production_linear_controller #(
        .N_OUTPUTS(N_OUTPUTS), .SPLIT_K(SPLIT_K),
        .SCALE_BURSTS(SCALE_BURSTS), .WEIGHT_BURSTS(WEIGHT_BURSTS),
        .ROB_DEPTH(ROB_DEPTH), .BASE_CHUNKS(BASE_CHUNKS),
        .EXTRA_PARTITIONS(EXTRA_PARTITIONS)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .dma_valid(dma_valid), .dma_ready(dma_ready),
        .dma_is_weight(dma_is_weight), .dma_sequence(dma_sequence),
        .dma_output_row(dma_output_row), .dma_partition(dma_partition),
        .dma_chunk(dma_chunk), .dma_chunks_in_split(dma_chunks),
        .event_dma_accept(event_dma_accept), .event_read_issue(event_read_issue),
        .event_compute(event_compute), .event_partial(event_partial),
        .event_output(event_output), .event_bank_conflict(event_bank_conflict),
        .count_dma_accept(count_dma_accept), .count_scale_bursts(count_scale_bursts),
        .count_weight_bursts(count_weight_bursts), .count_read_issue(count_read_issue),
        .count_compute(count_compute), .count_partial(count_partial),
        .count_output(count_output), .count_bank_conflict(count_bank_conflict),
        .count_ordered_wait(count_ordered_wait), .max_rob_occupancy(max_rob_occupancy)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            cycle <= 0;
            trace_index <= 0;
        end else begin
            if (event_dma_accept)
                trace_index <= trace_index + 1;
            if (event_output && count_output + 1 == N_OUTPUTS) begin
                #1;
                $display("SUMMARY %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                    cycle, count_dma_accept, count_scale_bursts,
                    count_weight_bursts, count_read_issue, count_compute,
                    count_partial, count_output, count_bank_conflict,
                    max_rob_occupancy);
                $finish;
            end
            cycle <= cycle + 1;
            if (cycle >= MAX_CYCLES) begin
                $display("TIMEOUT %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                    cycle, trace_index, count_scale_bursts, count_weight_bursts,
                    count_read_issue, count_compute, count_partial, count_output,
                    max_rob_occupancy);
                $fatal(1, "timeout");
            end
        end
    end
endmodule

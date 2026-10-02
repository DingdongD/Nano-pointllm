module tb_gtsu_dense_dramsim_dma_pipeline;
    parameter integer M = 2;
    parameter integer N = 5;
    parameter integer K = 16;
    parameter integer N_TILE = 3;
    parameter integer K_BLOCK = 8;
    parameter integer ROB_DEPTH = 4;
    parameter integer READ_LATENCY = 3;
    parameter integer FIFO_DEPTH = 8;
    parameter [31:0] REQUANT_MULTIPLIER = 32'd1;
    parameter [5:0] REQUANT_RIGHT_SHIFT = 6'd5;
    parameter integer OUTPUT_STALL_MOD = 7;
    parameter integer OUTPUT_STALL_PHASE = 3;
    parameter integer MAX_CYCLES = 100000;
    parameter integer N_TILES = (N+N_TILE-1)/N_TILE;
    parameter integer K_BLOCKS = K/K_BLOCK;
    parameter integer BLOCK_CHUNKS = K_BLOCK/4;
    parameter integer PAYLOADS = M*N_TILES*K_BLOCKS*BLOCK_CHUNKS;
    parameter integer LINES = PAYLOADS/4;
    parameter integer OUTPUT_TILES = M*N_TILES;
    parameter integer TRACE_W = 32+32+512;

    reg clk = 0;
    reg rst_n = 0;
    integer cycle = 0;
    integer completion_index = 0;
    reg [1023:0] trace_file;
    reg [TRACE_W-1:0] trace [0:LINES-1];
    wire [TRACE_W-1:0] current_trace = trace[
        (completion_index < LINES) ? completion_index : 0
    ];
    wire [31:0] completion_available_cycle = current_trace[31:0];
    wire [31:0] completion_tag = current_trace[63:32];
    wire [511:0] completion_data = current_trace[575:64];
    wire completion_valid = rst_n && completion_index < LINES
        && cycle >= completion_available_cycle;
    wire completion_ready;
    wire output_gate = (OUTPUT_STALL_MOD == 0) ? 1'b1
        : ((cycle % OUTPUT_STALL_MOD) != OUTPUT_STALL_PHASE);
    wire output_valid;
    wire output_ready = rst_n && output_gate;
    wire [31:0] output_tag;
    wire [N_TILE-1:0] output_column_mask;
    wire [N_TILE*8-1:0] output_data;
    wire done;
    wire overflow_error;
    wire dma_protocol_error;
    wire [31:0] count_dma_completions;
    wire [31:0] count_dma_words;
    wire [31:0] count_dma_rob_backpressure;
    wire [31:0] count_dma_rob_peak;
    wire [31:0] count_payload_writes;
    wire [31:0] count_sram_read_issues;
    wire [31:0] count_sram_responses;
    wire [31:0] count_dot4_inputs;
    wire [31:0] count_output_tiles;
    wire [31:0] count_read_write_overlap_cycles;

    always #5 clk = ~clk;
    initial begin
        if (!$value$plusargs("TRACE_FILE=%s", trace_file))
            $fatal(1, "TRACE_FILE plusarg is required");
        $readmemh(trace_file, trace);
        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1;
    end

    gtsu_dense_dramsim_dma_pipeline #(
        .M(M), .N(N), .K(K), .N_TILE(N_TILE), .K_BLOCK(K_BLOCK),
        .ROB_DEPTH(ROB_DEPTH), .READ_LATENCY(READ_LATENCY),
        .FIFO_DEPTH(FIFO_DEPTH), .REQUANT_MULTIPLIER(REQUANT_MULTIPLIER),
        .REQUANT_RIGHT_SHIFT(REQUANT_RIGHT_SHIFT)
    ) dut (
        .clk(clk), .rst_n(rst_n), .completion_valid(completion_valid),
        .completion_ready(completion_ready), .completion_tag(completion_tag),
        .completion_data(completion_data), .output_valid(output_valid),
        .output_ready(output_ready), .output_tag(output_tag),
        .output_column_mask(output_column_mask), .output_data(output_data),
        .done(done), .overflow_error(overflow_error),
        .dma_protocol_error(dma_protocol_error),
        .count_dma_completions(count_dma_completions),
        .count_dma_words(count_dma_words),
        .count_dma_rob_backpressure(count_dma_rob_backpressure),
        .count_dma_rob_peak(count_dma_rob_peak),
        .count_payload_writes(count_payload_writes),
        .count_sram_read_issues(count_sram_read_issues),
        .count_sram_responses(count_sram_responses),
        .count_dot4_inputs(count_dot4_inputs),
        .count_output_tiles(count_output_tiles),
        .count_read_write_overlap_cycles(count_read_write_overlap_cycles)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            cycle <= 0;
            completion_index <= 0;
        end else begin
            if (completion_valid && completion_ready) begin
                $display("TRACE %0d DRAM_COMPLETE_ACCEPT %0d %0h",
                    cycle, completion_tag, completion_available_cycle);
                completion_index <= completion_index + 1;
            end
            if (dut.payload_valid && dut.payload_ready) begin
                $display("TRACE %0d DMA_WORD_ACCEPT %0d %0h",
                    cycle, dut.dma.word_ordinal, dut.dma.word_offset);
                $display("TRACE %0d PAYLOAD_ACCEPT %0d %0h",
                    cycle, dut.payload_sequence, dut.payload_chunk);
            end
            if (dut.dense.read_fire)
                $display("TRACE %0d SRAM_READ_ISSUE %0d %0h",
                    cycle, dut.dense.ctrl_read_sequence, dut.dense.ctrl_read_chunk);
            if (dut.dense.sram_c0_rsp_valid)
                $display("TRACE %0d SRAM_RESPONSE %0d %0h", cycle,
                    dut.dense.metadata_sequence[dut.dense.sram_c0_rsp_tag],
                    dut.dense.sram_c0_rsp_tag);
            if (dut.dense.response_fifo_valid && dut.dense.dense_in_ready)
                $display("TRACE %0d DOT4_INPUT %0d 0",
                    cycle, dut.dense.fifo_sequence);
            if (output_valid && output_ready) begin
                $display("TRACE %0d OUTPUT_ACCEPT %0d %0h", cycle,
                    output_tag, {output_column_mask, output_data});
                if (count_output_tiles + 1 == OUTPUT_TILES) begin
                    #1;
                    $display("SUMMARY %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                        cycle + 1, count_dma_completions, count_dma_words,
                        count_dma_rob_backpressure, count_dma_rob_peak,
                        count_payload_writes, count_sram_read_issues,
                        count_sram_responses, count_dot4_inputs,
                        count_output_tiles, count_read_write_overlap_cycles,
                        overflow_error, dma_protocol_error, done);
                    $finish;
                end
            end
            cycle <= cycle + 1;
            if (overflow_error || dma_protocol_error)
                $fatal(1, "DMA/Dense protocol failure");
            if (cycle >= MAX_CYCLES)
                $fatal(1, "DRAMsim3 DMA Dense pipeline timeout");
        end
    end
endmodule

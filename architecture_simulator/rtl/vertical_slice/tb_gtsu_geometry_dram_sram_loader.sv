module tb_gtsu_geometry_dram_sram_loader;
    parameter integer POINTS = 128;
    parameter integer LANES = 64;
    parameter integer ROB_DEPTH = 8;
    parameter integer READ_LATENCY = 3;
    parameter integer LINES = POINTS / 4;
    parameter integer ROWS = POINTS / LANES;
    parameter integer ROW_W = 1;
    parameter integer TRACE_W = 32 + 32 + 512;
    reg clk = 0;
    reg rst_n = 0;
    integer cycle = 0;
    integer completion_index = 0;
    integer responses = 0;
    integer mismatches = 0;
    integer lane, row;
    reg [1023:0] trace_file;
    reg [TRACE_W-1:0] trace [0:LINES-1];
    wire [TRACE_W-1:0] current_trace = trace[
        completion_index < LINES ? completion_index : 0
    ];
    wire [31:0] completion_available_cycle = current_trace[31:0];
    wire [31:0] completion_tag = current_trace[63:32];
    wire [511:0] completion_data = current_trace[575:64];
    wire completion_valid = rst_n && completion_index < LINES &&
        cycle >= completion_available_cycle;
    wire completion_ready;
    reg min_load_valid = 0;
    wire min_load_ready;
    reg [ROW_W-1:0] min_load_row = 0;
    reg [LANES*32-1:0] min_load_values = 0;
    reg tile_req_valid = 0;
    wire tile_req_ready;
    reg [ROW_W-1:0] tile_req_row = 0;
    wire tile_rsp_valid;
    wire [LANES*128-1:0] tile_rsp_point_words;
    wire [LANES*32-1:0] tile_rsp_min_values;
    wire loader_done, dma_protocol_error, sram_collision;
    wire [31:0] count_dma_completions, count_point_words;
    reg [127:0] expected_point;

    function automatic [127:0] make_point_word(input integer value);
        begin
            make_point_word = {
                32'h40000000 + value,
                32'h30000000 + value,
                32'h20000000 + value,
                32'h10000000 + value
            };
        end
    endfunction

    gtsu_geometry_dram_sram_loader #(
        .POINTS(POINTS), .LANES(LANES), .ROB_DEPTH(ROB_DEPTH),
        .READ_LATENCY(READ_LATENCY)
    ) dut (
        .clk(clk), .rst_n(rst_n), .completion_valid(completion_valid),
        .completion_ready(completion_ready), .completion_tag(completion_tag),
        .completion_data(completion_data), .min_load_valid(min_load_valid),
        .min_load_ready(min_load_ready), .min_load_row(min_load_row),
        .min_load_values(min_load_values), .tile_req_valid(tile_req_valid),
        .tile_req_ready(tile_req_ready), .tile_req_row(tile_req_row),
        .tile_rsp_valid(tile_rsp_valid),
        .tile_rsp_point_words(tile_rsp_point_words),
        .tile_rsp_min_values(tile_rsp_min_values), .loader_done(loader_done),
        .dma_protocol_error(dma_protocol_error), .sram_collision(sram_collision),
        .count_dma_completions(count_dma_completions),
        .count_point_words(count_point_words)
    );

    always #5 clk = ~clk;

    initial begin
        if (!$value$plusargs("TRACE_FILE=%s", trace_file))
            $fatal(1, "TRACE_FILE plusarg is required");
        $readmemh(trace_file, trace);
        repeat (3) @(posedge clk);
        @(negedge clk); rst_n = 1;
        wait (loader_done);
        for (row = 0; row < ROWS; row = row + 1) begin
            @(negedge clk);
            for (lane = 0; lane < LANES; lane = lane + 1)
                min_load_values[lane*32 +: 32] = row * 1000 + lane;
            min_load_valid = 1;
            min_load_row = row;
        end
        @(negedge clk); min_load_valid = 0;
        for (row = 0; row < ROWS; row = row + 1) begin
            tile_req_valid = 1;
            tile_req_row = row;
            @(negedge clk);
        end
        tile_req_valid = 0;
        wait (responses == ROWS);
        repeat (2) @(posedge clk);
        $display("SUMMARY %0d %0d %0d %0d %0d %0d %0d", cycle,
                 count_dma_completions, count_point_words, responses,
                 mismatches, dma_protocol_error, sram_collision);
        if (mismatches || dma_protocol_error || sram_collision)
            $finish_and_return(1);
        $finish;
    end

    always @(posedge clk) begin
        if (rst_n) begin
            if (completion_valid && completion_ready) begin
                $display("TRACE %0d DRAM_COMPLETE_ACCEPT %0d %0d", cycle,
                         completion_tag, completion_available_cycle);
                completion_index <= completion_index + 1;
            end
            if (dut.payload_valid && dut.payload_ready)
                $display("TRACE %0d DMA_WORD_ACCEPT %0d %0d", cycle,
                         dut.dma.word_ordinal, dut.dma.word_offset);
            if (tile_req_valid && tile_req_ready)
                $display("TRACE %0d SRAM_READ_ISSUE %0d 0", cycle, tile_req_row);
            if (tile_rsp_valid) begin
                $display("TRACE %0d SRAM_RESPONSE %0d 0", cycle, responses);
                for (lane = 0; lane < LANES; lane = lane + 1) begin
                    expected_point = make_point_word(responses * LANES + lane);
                    if (tile_rsp_point_words[lane*128 +: 128] !== expected_point)
                        mismatches = mismatches + 1;
                    if (tile_rsp_min_values[lane*32 +: 32] !==
                        responses * 1000 + lane)
                        mismatches = mismatches + 1;
                end
                responses = responses + 1;
            end
            if (dma_protocol_error || sram_collision)
                $fatal(1, "geometry DMA/SRAM protocol failure");
            cycle <= cycle + 1;
        end
    end
endmodule

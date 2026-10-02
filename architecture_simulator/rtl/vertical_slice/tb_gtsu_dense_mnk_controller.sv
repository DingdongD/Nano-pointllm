module tb_gtsu_dense_mnk_controller;
    parameter integer M = 3;
    parameter integer N = 70;
    parameter integer K = 132;
    parameter integer N_TILE = 64;
    parameter integer K_BLOCK = 64;
    parameter integer SOURCE_STALL_MOD = 4;
    parameter integer SOURCE_STALL_PHASE = 1;
    parameter integer COMPUTE_STALL_MOD = 5;
    parameter integer COMPUTE_STALL_PHASE = 2;
    parameter integer MAX_CYCLES = 100000;
    parameter integer ROW_W = (M <= 1) ? 1 : $clog2(M);
    parameter integer NT_W = (((N+N_TILE-1)/N_TILE) <= 1) ? 1 : $clog2((N+N_TILE-1)/N_TILE);
    parameter integer KB_W = (((K+K_BLOCK-1)/K_BLOCK) <= 1) ? 1 : $clog2((K+K_BLOCK-1)/K_BLOCK);
    parameter integer CHUNK_W = ((K_BLOCK/4) <= 1) ? 1 : $clog2(K_BLOCK/4);
    parameter integer N_TILES = (N+N_TILE-1)/N_TILE;
    parameter integer TOTAL_CHUNKS = K/4;
    parameter integer BLOCK_CHUNKS = K_BLOCK/4;
    parameter integer K_BLOCKS = (TOTAL_CHUNKS+BLOCK_CHUNKS-1)/BLOCK_CHUNKS;
    parameter integer TOTAL_BLOCKS = M*N_TILES*K_BLOCKS;

    reg clk = 0;
    reg rst_n = 0;
    integer cycle = 0;
    integer sent = 0;
    integer loads = 0;
    integer read_issues = 0;
    integer blocks_released = 0;
    integer load_backpressure_cycles = 0;
    integer compute_wait_cycles = 0;
    integer downstream_stall_cycles = 0;
    integer ping_pong_overlap_cycles = 0;

    wire source_gate = (SOURCE_STALL_MOD == 0) ? 1'b1
        : ((cycle % SOURCE_STALL_MOD) != SOURCE_STALL_PHASE);
    wire load_valid = rst_n && sent < TOTAL_BLOCKS && source_gate;
    wire load_ready;
    wire compute_gate = (COMPUTE_STALL_MOD == 0) ? 1'b1
        : ((cycle % COMPUTE_STALL_MOD) != COMPUTE_STALL_PHASE);
    wire read_valid;
    wire read_ready = rst_n && compute_gate;
    wire [31:0] read_sequence;
    wire [ROW_W-1:0] read_row;
    wire [NT_W-1:0] read_n_tile;
    wire [KB_W-1:0] read_k_block;
    wire [CHUNK_W-1:0] read_chunk;
    wire [N_TILE-1:0] read_column_mask;
    wire read_first;
    wire read_last;
    wire block_release;
    wire done;

    always #5 clk = ~clk;
    initial begin
        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1;
    end

    gtsu_dense_mnk_controller #(
        .M(M), .N(N), .K(K), .N_TILE(N_TILE), .K_BLOCK(K_BLOCK)
    ) dut (
        .clk(clk), .rst_n(rst_n), .load_valid(load_valid),
        .load_ready(load_ready), .load_sequence(sent),
        .read_valid(read_valid), .read_ready(read_ready),
        .read_sequence(read_sequence), .read_row(read_row),
        .read_n_tile(read_n_tile), .read_k_block(read_k_block),
        .read_chunk(read_chunk), .read_column_mask(read_column_mask),
        .read_first(read_first), .read_last(read_last),
        .block_release(block_release), .done(done)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            cycle <= 0;
            sent <= 0;
            loads <= 0;
            read_issues <= 0;
            blocks_released <= 0;
            load_backpressure_cycles <= 0;
            compute_wait_cycles <= 0;
            downstream_stall_cycles <= 0;
            ping_pong_overlap_cycles <= 0;
        end else begin
            if (load_valid && load_ready) begin
                $display("TRACE %0d LOAD_ACCEPT %0d 0 0 0 0 0 0 0",
                    cycle, sent);
                sent <= sent + 1;
                loads <= loads + 1;
            end
            if (load_valid && !load_ready)
                load_backpressure_cycles <= load_backpressure_cycles + 1;
            if (!read_valid)
                compute_wait_cycles <= compute_wait_cycles + 1;
            else if (!read_ready)
                downstream_stall_cycles <= downstream_stall_cycles + 1;
            if (read_valid && dut.buffer_valid[~read_sequence[0]])
                ping_pong_overlap_cycles <= ping_pong_overlap_cycles + 1;
            if (read_valid && read_ready) begin
                $display("TRACE %0d READ_ISSUE %0d %0d %0d %0d %0d %0h %0d %0d",
                    cycle, read_sequence, read_row, read_n_tile, read_k_block,
                    read_chunk, read_column_mask, read_first, read_last);
                read_issues <= read_issues + 1;
            end
            if (block_release) begin
                $display("TRACE %0d BLOCK_RELEASE %0d %0d %0d %0d %0d %0h %0d %0d",
                    cycle, read_sequence, read_row, read_n_tile, read_k_block,
                    read_chunk, read_column_mask, read_first, read_last);
                blocks_released <= blocks_released + 1;
                if (blocks_released + 1 == TOTAL_BLOCKS) begin
                    #1;
                    $display("SUMMARY %0d %0d %0d %0d %0d %0d %0d %0d",
                        cycle + 1, loads, read_issues, blocks_released,
                        load_backpressure_cycles, compute_wait_cycles,
                        downstream_stall_cycles, ping_pong_overlap_cycles);
                    $finish;
                end
            end
            cycle <= cycle + 1;
            if (cycle >= MAX_CYCLES)
                $fatal(1, "dense M/N/K controller timeout");
        end
    end
endmodule

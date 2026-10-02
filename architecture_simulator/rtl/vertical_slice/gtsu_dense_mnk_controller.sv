// Ping-pong block lifecycle follows the registered-buffer interface pattern
// audited in Compiler_Codes LBUF; payload SRAM ports are not integrated here.
module gtsu_dense_mnk_controller #(
    parameter integer M = 3,
    parameter integer N = 70,
    parameter integer K = 132,
    parameter integer N_TILE = 64,
    parameter integer K_BLOCK = 64,
    parameter integer ROW_W = (M <= 1) ? 1 : $clog2(M),
    parameter integer NT_W = (((N+N_TILE-1)/N_TILE) <= 1) ? 1 : $clog2((N+N_TILE-1)/N_TILE),
    parameter integer KB_W = (((K+K_BLOCK-1)/K_BLOCK) <= 1) ? 1 : $clog2((K+K_BLOCK-1)/K_BLOCK),
    parameter integer CHUNK_W = ((K_BLOCK/4) <= 1) ? 1 : $clog2(K_BLOCK/4)
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         load_valid,
    output wire                         load_ready,
    input  wire [31:0]                  load_sequence,
    output wire                         read_valid,
    input  wire                         read_ready,
    output wire [31:0]                  read_sequence,
    output wire [ROW_W-1:0]             read_row,
    output wire [NT_W-1:0]              read_n_tile,
    output wire [KB_W-1:0]              read_k_block,
    output wire [CHUNK_W-1:0]           read_chunk,
    output wire [N_TILE-1:0]            read_column_mask,
    output wire                         read_first,
    output wire                         read_last,
    output wire                         block_release,
    output wire                         done
);
    localparam integer N_TILES = (N + N_TILE - 1) / N_TILE;
    localparam integer TOTAL_CHUNKS = K / 4;
    localparam integer BLOCK_CHUNKS = K_BLOCK / 4;
    localparam integer K_BLOCKS = (TOTAL_CHUNKS + BLOCK_CHUNKS - 1) / BLOCK_CHUNKS;
    localparam integer TOTAL_BLOCKS = M * N_TILES * K_BLOCKS;

    reg [1:0] buffer_valid;
    reg [31:0] buffer_sequence [0:1];
    reg [31:0] expected_sequence;
    reg [ROW_W-1:0] row_counter;
    reg [NT_W-1:0] n_tile_counter;
    reg [KB_W-1:0] k_block_counter;
    reg [CHUNK_W-1:0] chunk_counter;

    wire load_slot = load_sequence[0];
    wire compute_slot = expected_sequence[0];
    wire expected_buffer_ready = buffer_valid[compute_slot]
        && buffer_sequence[compute_slot] == expected_sequence;
    wire [31:0] current_block_chunks = (k_block_counter == K_BLOCKS-1)
        ? TOTAL_CHUNKS - (K_BLOCKS-1)*BLOCK_CHUNKS : BLOCK_CHUNKS;
    wire [31:0] valid_columns = (n_tile_counter == N_TILES-1)
        ? N - (N_TILES-1)*N_TILE : N_TILE;

    assign load_ready = !buffer_valid[load_slot];
    assign read_valid = expected_buffer_ready && expected_sequence < TOTAL_BLOCKS;
    assign read_sequence = expected_sequence;
    assign read_row = row_counter;
    assign read_n_tile = n_tile_counter;
    assign read_k_block = k_block_counter;
    assign read_chunk = chunk_counter;
    assign read_column_mask = (valid_columns >= N_TILE)
        ? {N_TILE{1'b1}} : ({N_TILE{1'b1}} >> (N_TILE-valid_columns));
    assign read_first = k_block_counter == 0 && chunk_counter == 0;
    assign read_last = k_block_counter == K_BLOCKS-1
        && chunk_counter == current_block_chunks-1;
    assign block_release = read_valid && read_ready
        && chunk_counter == current_block_chunks-1;
    assign done = expected_sequence == TOTAL_BLOCKS;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            buffer_valid <= 0;
            buffer_sequence[0] <= 0;
            buffer_sequence[1] <= 0;
            expected_sequence <= 0;
            row_counter <= 0;
            n_tile_counter <= 0;
            k_block_counter <= 0;
            chunk_counter <= 0;
        end else begin
            if (load_valid && load_ready) begin
                buffer_valid[load_slot] <= 1'b1;
                buffer_sequence[load_slot] <= load_sequence;
            end
            if (read_valid && read_ready) begin
                if (chunk_counter == current_block_chunks-1) begin
                    buffer_valid[compute_slot] <= 1'b0;
                    expected_sequence <= expected_sequence + 1;
                    chunk_counter <= 0;
                    if (k_block_counter == K_BLOCKS-1) begin
                        k_block_counter <= 0;
                        if (n_tile_counter == N_TILES-1) begin
                            n_tile_counter <= 0;
                            row_counter <= row_counter + 1'b1;
                        end else begin
                            n_tile_counter <= n_tile_counter + 1'b1;
                        end
                    end else begin
                        k_block_counter <= k_block_counter + 1'b1;
                    end
                end else begin
                    chunk_counter <= chunk_counter + 1'b1;
                end
            end
        end
    end

    initial begin
        if (M <= 0 || N <= 0 || K <= 0 || N_TILE <= 0 || N_TILE > 64
            || K_BLOCK <= 0 || (K % 4) != 0 || (K_BLOCK % 4) != 0)
            $error("invalid dense M/N/K controller parameters");
    end
endmodule

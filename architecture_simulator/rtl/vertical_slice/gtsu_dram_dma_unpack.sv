// Tagged 64-byte DRAM completion ROB with ordered 128-bit word unpack.
module gtsu_dram_dma_unpack #(
    parameter integer LINES = 4,
    parameter integer ROB_DEPTH = 4,
    parameter integer BLOCK_CHUNKS = 2,
    parameter integer SLOT_W = (ROB_DEPTH <= 1) ? 1 : $clog2(ROB_DEPTH)
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         completion_valid,
    output wire                         completion_ready,
    input  wire [31:0]                  completion_tag,
    input  wire [511:0]                 completion_data,
    output wire                         payload_valid,
    input  wire                         payload_ready,
    output wire [31:0]                  payload_sequence,
    output wire [7:0]                   payload_chunk,
    output wire [127:0]                 payload_data,
    output wire                         protocol_error,
    output reg  [31:0]                  count_completion_accepts,
    output reg  [31:0]                  count_payload_words,
    output reg  [31:0]                  count_rob_backpressure_cycles,
    output reg  [31:0]                  count_rob_peak
);
    reg [511:0] rob_data [0:ROB_DEPTH-1];
    reg [31:0] rob_tag [0:ROB_DEPTH-1];
    reg [ROB_DEPTH-1:0] rob_valid;
    reg [31:0] retire_line;
    reg [1:0] word_offset;
    reg [31:0] rob_occupancy;
    reg protocol_error_reg;

    wire [SLOT_W-1:0] completion_slot = completion_tag % ROB_DEPTH;
    wire [SLOT_W-1:0] retire_slot = retire_line % ROB_DEPTH;
    wire completion_in_window = completion_tag >= retire_line
        && completion_tag < retire_line + ROB_DEPTH && completion_tag < LINES;
    wire completion_slot_free = !rob_valid[completion_slot];
    assign completion_ready = completion_in_window && completion_slot_free
        && rob_occupancy < ROB_DEPTH;
    wire completion_fire = completion_valid && completion_ready;

    assign payload_valid = retire_line < LINES && rob_valid[retire_slot]
        && rob_tag[retire_slot] == retire_line;
    wire [31:0] word_ordinal = retire_line * 4 + word_offset;
    assign payload_sequence = word_ordinal / BLOCK_CHUNKS;
    assign payload_chunk = word_ordinal % BLOCK_CHUNKS;
    assign payload_data = rob_data[retire_slot] >> (word_offset * 128);
    wire payload_fire = payload_valid && payload_ready;
    wire line_retire = payload_fire && word_offset == 3;
    assign protocol_error = protocol_error_reg;

    integer index;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rob_valid <= 0;
            retire_line <= 0;
            word_offset <= 0;
            rob_occupancy <= 0;
            protocol_error_reg <= 0;
            count_completion_accepts <= 0;
            count_payload_words <= 0;
            count_rob_backpressure_cycles <= 0;
            count_rob_peak <= 0;
            for (index = 0; index < ROB_DEPTH; index = index + 1) begin
                rob_data[index] <= 0;
                rob_tag[index] <= 0;
            end
        end else begin
            if (completion_valid
                && (completion_tag < retire_line || completion_tag >= LINES))
                protocol_error_reg <= 1'b1;
            if (completion_valid && !completion_ready)
                count_rob_backpressure_cycles <= count_rob_backpressure_cycles + 1;
            if (completion_fire) begin
                rob_valid[completion_slot] <= 1'b1;
                rob_tag[completion_slot] <= completion_tag;
                rob_data[completion_slot] <= completion_data;
                count_completion_accepts <= count_completion_accepts + 1;
                if (!line_retire && rob_occupancy + 1 > count_rob_peak)
                    count_rob_peak <= rob_occupancy + 1;
            end
            if (payload_fire) begin
                count_payload_words <= count_payload_words + 1;
                if (word_offset == 3) begin
                    rob_valid[retire_slot] <= 1'b0;
                    retire_line <= retire_line + 1;
                    word_offset <= 0;
                end else begin
                    word_offset <= word_offset + 1'b1;
                end
            end
            case ({completion_fire, line_retire})
                2'b10: rob_occupancy <= rob_occupancy + 1;
                2'b01: rob_occupancy <= rob_occupancy - 1;
                default: rob_occupancy <= rob_occupancy;
            endcase
        end
    end

    initial begin
        if (LINES <= 0 || ROB_DEPTH <= 0 || BLOCK_CHUNKS <= 0)
            $error("DMA dimensions must be positive");
    end
endmodule

module gtsu_banked_sram_2client #(
    // The correlation testbench overrides these with the Compiler_Codes LBUF
    // geometry (16 x 16384). Compact defaults keep generic logic synthesis
    // independent of a foundry SRAM macro or a huge register-array mapping.
    parameter integer BANKS = 2,
    parameter integer ROWS_PER_BANK = 16,
    parameter integer DATA_W = 128,
    parameter integer READ_LATENCY = 3,
    parameter integer TAG_W = 8,
    parameter integer BANK_W = (BANKS <= 1) ? 1 : $clog2(BANKS),
    parameter integer ROW_W = (ROWS_PER_BANK <= 1) ? 1 : $clog2(ROWS_PER_BANK),
    parameter integer ADDR_W = BANK_W + ROW_W,
    parameter integer BYTE_W = DATA_W / 8
) (
    input  wire                  clk,
    input  wire                  rst_n,

    input  wire                  c0_req_valid,
    output wire                  c0_req_ready,
    input  wire                  c0_req_write,
    input  wire [ADDR_W-1:0]     c0_req_word_addr,
    input  wire [TAG_W-1:0]      c0_req_tag,
    input  wire [DATA_W-1:0]     c0_req_wdata,
    input  wire [BYTE_W-1:0]     c0_req_bwen,

    input  wire                  c1_req_valid,
    output wire                  c1_req_ready,
    input  wire                  c1_req_write,
    input  wire [ADDR_W-1:0]     c1_req_word_addr,
    input  wire [TAG_W-1:0]      c1_req_tag,
    input  wire [DATA_W-1:0]     c1_req_wdata,
    input  wire [BYTE_W-1:0]     c1_req_bwen,

    output wire                  c0_rsp_valid,
    output wire [TAG_W-1:0]      c0_rsp_tag,
    output wire [BANK_W-1:0]     c0_rsp_bank,
    output wire [ROW_W-1:0]      c0_rsp_row,
    output wire [DATA_W-1:0]     c0_rsp_rdata,
    output wire                  c1_rsp_valid,
    output wire [TAG_W-1:0]      c1_rsp_tag,
    output wire [BANK_W-1:0]     c1_rsp_bank,
    output wire [ROW_W-1:0]      c1_rsp_row,
    output wire [DATA_W-1:0]     c1_rsp_rdata,

    output wire                  event_c0_accept,
    output wire                  event_c1_accept
);
    wire [BANK_W-1:0] c0_bank = c0_req_word_addr[BANK_W-1:0];
    wire [BANK_W-1:0] c1_bank = c1_req_word_addr[BANK_W-1:0];
    wire [ROW_W-1:0] c0_row = c0_req_word_addr[ADDR_W-1:BANK_W];
    wire [ROW_W-1:0] c1_row = c1_req_word_addr[ADDR_W-1:BANK_W];
    wire same_port_bank_collision = c0_req_valid && c1_req_valid
        && (c0_req_write == c1_req_write) && (c0_bank == c1_bank);

    // Client 0 is the fixed-priority port on a same-bank, same-direction clash.
    assign c0_req_ready = 1'b1;
    assign c1_req_ready = !same_port_bank_collision;
    assign event_c0_accept = c0_req_valid && c0_req_ready;
    assign event_c1_accept = c1_req_valid && c1_req_ready;

    reg [DATA_W-1:0] memory [0:BANKS-1][0:ROWS_PER_BANK-1];
    reg [READ_LATENCY-1:0] c0_vld_pipe;
    reg [READ_LATENCY-1:0] c1_vld_pipe;
    reg [TAG_W-1:0] c0_tag_pipe [0:READ_LATENCY-1];
    reg [TAG_W-1:0] c1_tag_pipe [0:READ_LATENCY-1];
    reg [BANK_W-1:0] c0_bank_pipe [0:READ_LATENCY-1];
    reg [BANK_W-1:0] c1_bank_pipe [0:READ_LATENCY-1];
    reg [ROW_W-1:0] c0_row_pipe [0:READ_LATENCY-1];
    reg [ROW_W-1:0] c1_row_pipe [0:READ_LATENCY-1];
    reg [DATA_W-1:0] c0_data_pipe [0:READ_LATENCY-1];
    reg [DATA_W-1:0] c1_data_pipe [0:READ_LATENCY-1];

    integer stage;
    integer byte_index;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            c0_vld_pipe <= 0;
            c1_vld_pipe <= 0;
            for (stage = 0; stage < READ_LATENCY; stage = stage + 1) begin
                c0_tag_pipe[stage] <= 0;
                c1_tag_pipe[stage] <= 0;
                c0_bank_pipe[stage] <= 0;
                c1_bank_pipe[stage] <= 0;
                c0_row_pipe[stage] <= 0;
                c1_row_pipe[stage] <= 0;
                c0_data_pipe[stage] <= 0;
                c1_data_pipe[stage] <= 0;
            end
        end else begin
            for (stage = READ_LATENCY - 1; stage > 0; stage = stage - 1) begin
                c0_vld_pipe[stage] <= c0_vld_pipe[stage-1];
                c1_vld_pipe[stage] <= c1_vld_pipe[stage-1];
                c0_tag_pipe[stage] <= c0_tag_pipe[stage-1];
                c1_tag_pipe[stage] <= c1_tag_pipe[stage-1];
                c0_bank_pipe[stage] <= c0_bank_pipe[stage-1];
                c1_bank_pipe[stage] <= c1_bank_pipe[stage-1];
                c0_row_pipe[stage] <= c0_row_pipe[stage-1];
                c1_row_pipe[stage] <= c1_row_pipe[stage-1];
                c0_data_pipe[stage] <= c0_data_pipe[stage-1];
                c1_data_pipe[stage] <= c1_data_pipe[stage-1];
            end
            c0_vld_pipe[0] <= event_c0_accept && !c0_req_write;
            c1_vld_pipe[0] <= event_c1_accept && !c1_req_write;
            if (event_c0_accept && !c0_req_write) begin
                c0_tag_pipe[0] <= c0_req_tag;
                c0_bank_pipe[0] <= c0_bank;
                c0_row_pipe[0] <= c0_row;
                c0_data_pipe[0] <= memory[c0_bank][c0_row];
            end
            if (event_c1_accept && !c1_req_write) begin
                c1_tag_pipe[0] <= c1_req_tag;
                c1_bank_pipe[0] <= c1_bank;
                c1_row_pipe[0] <= c1_row;
                c1_data_pipe[0] <= memory[c1_bank][c1_row];
            end
            if (event_c0_accept && c0_req_write) begin
                for (byte_index = 0; byte_index < BYTE_W; byte_index = byte_index + 1)
                    if (c0_req_bwen[byte_index])
                        memory[c0_bank][c0_row][byte_index*8 +: 8]
                            <= c0_req_wdata[byte_index*8 +: 8];
            end
            if (event_c1_accept && c1_req_write) begin
                for (byte_index = 0; byte_index < BYTE_W; byte_index = byte_index + 1)
                    if (c1_req_bwen[byte_index])
                        memory[c1_bank][c1_row][byte_index*8 +: 8]
                            <= c1_req_wdata[byte_index*8 +: 8];
            end
        end
    end

    assign c0_rsp_valid = c0_vld_pipe[READ_LATENCY-1];
    assign c0_rsp_tag = c0_tag_pipe[READ_LATENCY-1];
    assign c0_rsp_bank = c0_bank_pipe[READ_LATENCY-1];
    assign c0_rsp_row = c0_row_pipe[READ_LATENCY-1];
    assign c0_rsp_rdata = c0_data_pipe[READ_LATENCY-1];
    assign c1_rsp_valid = c1_vld_pipe[READ_LATENCY-1];
    assign c1_rsp_tag = c1_tag_pipe[READ_LATENCY-1];
    assign c1_rsp_bank = c1_bank_pipe[READ_LATENCY-1];
    assign c1_rsp_row = c1_row_pipe[READ_LATENCY-1];
    assign c1_rsp_rdata = c1_data_pipe[READ_LATENCY-1];
endmodule

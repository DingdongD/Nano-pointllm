module gtsu_lbuf_16bank_macro_model #(
    parameter integer BANKS = 16,
    parameter integer ROWS = 4,
    parameter integer DATA_W = 128,
    parameter integer READ_LATENCY = 3,
    parameter integer ROW_W = (ROWS <= 2) ? 1 : $clog2(ROWS),
    parameter integer BYTE_W = DATA_W / 8
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         rd_valid,
    output wire                         rd_ready,
    input  wire [BANKS-1:0]             rd_bank_mask,
    input  wire [BANKS*ROW_W-1:0]       rd_rows,
    output wire                         rd_rsp_valid,
    output wire [BANKS-1:0]             rd_rsp_bank_mask,
    output wire [BANKS*DATA_W-1:0]      rd_rsp_data,
    input  wire                         wr_valid,
    output wire                         wr_ready,
    input  wire [BANKS-1:0]             wr_bank_mask,
    input  wire [BANKS*ROW_W-1:0]       wr_rows,
    input  wire [BANKS*DATA_W-1:0]      wr_data,
    input  wire [BANKS*BYTE_W-1:0]      wr_bwen,
    output wire                         same_address_rw_collision
);
    reg [DATA_W-1:0] memory [0:BANKS-1][0:ROWS-1];
    reg [READ_LATENCY-1:0] valid_pipe;
    reg [BANKS-1:0] mask_pipe [0:READ_LATENCY-1];
    reg [BANKS*DATA_W-1:0] data_pipe [0:READ_LATENCY-1];
    reg collision_comb;
    integer bank, stage, byte_index;

    assign rd_ready = 1'b1;
    assign wr_ready = 1'b1;
    assign rd_rsp_valid = valid_pipe[READ_LATENCY-1];
    assign rd_rsp_bank_mask = mask_pipe[READ_LATENCY-1];
    assign rd_rsp_data = data_pipe[READ_LATENCY-1];
    assign same_address_rw_collision = collision_comb;

    always @* begin
        collision_comb = 1'b0;
        if (rd_valid && wr_valid) begin
            for (bank = 0; bank < BANKS; bank = bank + 1)
                if (rd_bank_mask[bank] && wr_bank_mask[bank] &&
                    rd_rows[bank*ROW_W +: ROW_W] ==
                    wr_rows[bank*ROW_W +: ROW_W])
                    collision_comb = 1'b1;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_pipe <= 0;
            for (stage = 0; stage < READ_LATENCY; stage = stage + 1) begin
                mask_pipe[stage] <= 0;
                data_pipe[stage] <= 0;
            end
        end else begin
            for (stage = READ_LATENCY-1; stage > 0; stage = stage - 1) begin
                valid_pipe[stage] <= valid_pipe[stage-1];
                mask_pipe[stage] <= mask_pipe[stage-1];
                data_pipe[stage] <= data_pipe[stage-1];
            end
            valid_pipe[0] <= rd_valid && rd_ready;
            mask_pipe[0] <= rd_bank_mask;
            for (bank = 0; bank < BANKS; bank = bank + 1)
                if (rd_valid && rd_ready && rd_bank_mask[bank])
                    data_pipe[0][bank*DATA_W +: DATA_W] <=
                        memory[bank][rd_rows[bank*ROW_W +: ROW_W]];
            if (wr_valid && wr_ready) begin
                for (bank = 0; bank < BANKS; bank = bank + 1)
                    if (wr_bank_mask[bank]) begin
                        for (byte_index = 0; byte_index < BYTE_W;
                             byte_index = byte_index + 1)
                            if (wr_bwen[bank*BYTE_W + byte_index])
                                memory[bank][wr_rows[bank*ROW_W +: ROW_W]]
                                    [byte_index*8 +: 8] <=
                                    wr_data[bank*DATA_W + byte_index*8 +: 8];
                    end
            end
        end
    end

    initial begin
        if (BANKS <= 0 || ROWS <= 0 || DATA_W <= 0 || READ_LATENCY <= 0)
            $error("LBUF macro dimensions must be positive");
        if (DATA_W % 8 != 0)
            $error("LBUF macro DATA_W must be byte aligned");
    end
endmodule

module tb_gtsu_banked_sram_2client;
    localparam integer BANKS = 16;
    localparam integer ROWS = 16384;
    localparam integer DATA_W = 128;
    localparam integer READ_LATENCY = 3;
    localparam integer BANK_W = 4;
    localparam integer ROW_W = 14;
    localparam integer ADDR_W = BANK_W + ROW_W;
    localparam integer C0_COUNT = 9;
    localparam integer C1_COUNT = 5;

    reg clk = 0;
    reg rst_n = 0;
    integer cycle = 0;
    integer c0_index = 0;
    integer c1_index = 0;
    integer responses = 0;

    reg [7:0] c0_available [0:C0_COUNT-1];
    reg c0_write [0:C0_COUNT-1];
    reg [ADDR_W-1:0] c0_addr [0:C0_COUNT-1];
    reg [7:0] c0_tag [0:C0_COUNT-1];
    reg [DATA_W-1:0] c0_data [0:C0_COUNT-1];
    reg [7:0] c1_available [0:C1_COUNT-1];
    reg c1_write [0:C1_COUNT-1];
    reg [ADDR_W-1:0] c1_addr [0:C1_COUNT-1];
    reg [7:0] c1_tag [0:C1_COUNT-1];
    reg [DATA_W-1:0] c1_data [0:C1_COUNT-1];

    wire c0_req_valid = rst_n && c0_index < C0_COUNT
        && cycle >= c0_available[c0_index];
    wire c1_req_valid = rst_n && c1_index < C1_COUNT
        && cycle >= c1_available[c1_index];
    wire c0_req_ready;
    wire c1_req_ready;
    wire c0_req_write = c0_write[c0_index];
    wire c1_req_write = c1_write[c1_index];
    wire [ADDR_W-1:0] c0_req_addr = c0_addr[c0_index];
    wire [ADDR_W-1:0] c1_req_addr = c1_addr[c1_index];
    wire [7:0] c0_req_tag = c0_tag[c0_index];
    wire [7:0] c1_req_tag = c1_tag[c1_index];
    wire [DATA_W-1:0] c0_req_data = c0_data[c0_index];
    wire [DATA_W-1:0] c1_req_data = c1_data[c1_index];
    wire event_c0_accept;
    wire event_c1_accept;
    wire c0_rsp_valid;
    wire c1_rsp_valid;
    wire [7:0] c0_rsp_tag;
    wire [7:0] c1_rsp_tag;
    wire [BANK_W-1:0] c0_rsp_bank;
    wire [BANK_W-1:0] c1_rsp_bank;
    wire [ROW_W-1:0] c0_rsp_row;
    wire [ROW_W-1:0] c1_rsp_row;
    wire [DATA_W-1:0] c0_rsp_data;
    wire [DATA_W-1:0] c1_rsp_data;

    always #5 clk = ~clk;

    initial begin
        c0_available[0]=0; c0_write[0]=1; c0_addr[0]=0;  c0_tag[0]=0;  c0_data[0]='h11;
        c0_available[1]=1; c0_write[1]=1; c0_addr[1]=1;  c0_tag[1]=1;  c0_data[1]='h22;
        c0_available[2]=2; c0_write[2]=1; c0_addr[2]=16; c0_tag[2]=2;  c0_data[2]='h33;
        c0_available[3]=3; c0_write[3]=1; c0_addr[3]=3;  c0_tag[3]=6;  c0_data[3]=0;
        c0_available[4]=4; c0_write[4]=0; c0_addr[4]=0;  c0_tag[4]=10; c0_data[4]=0;
        c0_available[5]=5; c0_write[5]=0; c0_addr[5]=1;  c0_tag[5]=11; c0_data[5]=0;
        c0_available[6]=6; c0_write[6]=0; c0_addr[6]=16; c0_tag[6]=12; c0_data[6]=0;
        c0_available[7]=10; c0_write[7]=0; c0_addr[7]=3; c0_tag[7]=13; c0_data[7]=0;
        c0_available[8]=11; c0_write[8]=0; c0_addr[8]=3; c0_tag[8]=14; c0_data[8]=0;

        c1_available[0]=0; c1_write[0]=1; c1_addr[0]=0; c1_tag[0]=3;  c1_data[0]='hAA;
        c1_available[1]=1; c1_write[1]=1; c1_addr[1]=2; c1_tag[1]=4;  c1_data[1]='hBB;
        c1_available[2]=4; c1_write[2]=0; c1_addr[2]=0; c1_tag[2]=20; c1_data[2]=0;
        c1_available[3]=5; c1_write[3]=0; c1_addr[3]=2; c1_tag[3]=21; c1_data[3]=0;
        c1_available[4]=10; c1_write[4]=1; c1_addr[4]=3; c1_tag[4]=5; c1_data[4]='hCC;

        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1;
    end

    gtsu_banked_sram_2client #(
        .BANKS(BANKS), .ROWS_PER_BANK(ROWS), .DATA_W(DATA_W),
        .READ_LATENCY(READ_LATENCY), .TAG_W(8)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .c0_req_valid(c0_req_valid), .c0_req_ready(c0_req_ready),
        .c0_req_write(c0_req_write), .c0_req_word_addr(c0_req_addr),
        .c0_req_tag(c0_req_tag), .c0_req_wdata(c0_req_data), .c0_req_bwen(16'hffff),
        .c1_req_valid(c1_req_valid), .c1_req_ready(c1_req_ready),
        .c1_req_write(c1_req_write), .c1_req_word_addr(c1_req_addr),
        .c1_req_tag(c1_req_tag), .c1_req_wdata(c1_req_data), .c1_req_bwen(16'hffff),
        .c0_rsp_valid(c0_rsp_valid), .c0_rsp_tag(c0_rsp_tag),
        .c0_rsp_bank(c0_rsp_bank), .c0_rsp_row(c0_rsp_row), .c0_rsp_rdata(c0_rsp_data),
        .c1_rsp_valid(c1_rsp_valid), .c1_rsp_tag(c1_rsp_tag),
        .c1_rsp_bank(c1_rsp_bank), .c1_rsp_row(c1_rsp_row), .c1_rsp_rdata(c1_rsp_data),
        .event_c0_accept(event_c0_accept), .event_c1_accept(event_c1_accept)
    );

    task print_accept;
        input integer client;
        input integer is_write;
        input integer tag;
        input integer word_addr;
        input [DATA_W-1:0] data;
        begin
            if (is_write)
                $display("TRACE %0d WRITE_ACCEPT %0d %0d %0d %0d %0d",
                    cycle, client, tag, word_addr % BANKS, word_addr / BANKS, data[31:0]);
            else
                $display("TRACE %0d READ_ACCEPT %0d %0d %0d %0d 0",
                    cycle, client, tag, word_addr % BANKS, word_addr / BANKS);
        end
    endtask

    always @(posedge clk) begin
        if (!rst_n) begin
            cycle <= 0;
            c0_index <= 0;
            c1_index <= 0;
            responses <= 0;
        end else begin
            if (event_c0_accept) begin
                print_accept(0, c0_req_write, c0_req_tag, c0_req_addr, c0_req_data);
                c0_index <= c0_index + 1;
            end
            if (event_c1_accept) begin
                print_accept(1, c1_req_write, c1_req_tag, c1_req_addr, c1_req_data);
                c1_index <= c1_index + 1;
            end
            if (c0_rsp_valid) begin
                $display("TRACE %0d READ_COMPLETE 0 %0d %0d %0d %0d",
                    cycle, c0_rsp_tag, c0_rsp_bank, c0_rsp_row, c0_rsp_data[31:0]);
            end
            if (c1_rsp_valid) begin
                $display("TRACE %0d READ_COMPLETE 1 %0d %0d %0d %0d",
                    cycle, c1_rsp_tag, c1_rsp_bank, c1_rsp_row, c1_rsp_data[31:0]);
            end
            responses <= responses + c0_rsp_valid + c1_rsp_valid;
            if (responses + c0_rsp_valid + c1_rsp_valid == 7) begin
                #1;
                $display("SUMMARY %0d", cycle + 1);
                $finish;
            end
            cycle <= cycle + 1;
            if (cycle > 100) $fatal(1, "timeout");
        end
    end
endmodule

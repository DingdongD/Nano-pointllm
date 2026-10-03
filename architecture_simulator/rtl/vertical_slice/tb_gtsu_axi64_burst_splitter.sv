module tb_gtsu_axi64_burst_splitter;
    reg clk = 0;
    reg rst_n = 0;
    reg cmd_valid = 0;
    wire cmd_ready;
    reg [63:0] cmd_address = 0;
    reg [23:0] cmd_beats = 0;
    reg enforce_4k_boundary = 1;
    wire burst_valid;
    wire burst_ready;
    wire [63:0] burst_address;
    wire [7:0] burst_len;
    wire [8:0] burst_beats;
    wire burst_last;
    wire done;
    integer cycle = 0;
    integer command_count = 0;
    integer burst_count = 0;
    integer mismatch = 0;
    reg [63:0] expected_address [0:7];
    reg [8:0] expected_beats [0:7];

    assign burst_ready = cycle % 5 != 2;

    gtsu_axi64_burst_splitter dut (
        .clk(clk), .rst_n(rst_n), .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
        .cmd_address(cmd_address), .cmd_beats(cmd_beats),
        .enforce_4k_boundary(enforce_4k_boundary),
        .burst_valid(burst_valid), .burst_ready(burst_ready),
        .burst_address(burst_address), .burst_len(burst_len),
        .burst_beats(burst_beats), .burst_last(burst_last), .done(done)
    );

    always #5 clk = ~clk;

    initial begin
        expected_address[0] = 64'h1000; expected_beats[0] = 64;
        expected_address[1] = 64'h2000; expected_beats[1] = 64;
        expected_address[2] = 64'h3000; expected_beats[2] = 2;
        expected_address[3] = 64'h1fc0; expected_beats[3] = 1;
        expected_address[4] = 64'h2000; expected_beats[4] = 64;
        expected_address[5] = 64'h3000; expected_beats[5] = 1;
        expected_address[6] = 64'h4000; expected_beats[6] = 256;
        expected_address[7] = 64'h8000; expected_beats[7] = 44;
        repeat (3) @(posedge clk);
        rst_n = 1;
        @(negedge clk);
        cmd_valid = 1; cmd_address = 64'h1000; cmd_beats = 130;
        @(negedge clk);
        cmd_valid = 0;
        wait (done);
        @(negedge clk);
        cmd_valid = 1; cmd_address = 64'h1fc0; cmd_beats = 66;
        @(negedge clk);
        cmd_valid = 0;
        wait (done);
        @(negedge clk);
        enforce_4k_boundary = 0;
        cmd_valid = 1; cmd_address = 64'h4000; cmd_beats = 300;
        @(negedge clk);
        cmd_valid = 0;
        wait (done);
        repeat (2) @(posedge clk);
        $display("SUMMARY %0d %0d %0d %0d", cycle, command_count,
                 burst_count, mismatch);
        if (mismatch) $finish_and_return(1);
        $finish;
    end

    always @(posedge clk) begin
        if (rst_n) begin
            if (cmd_valid && cmd_ready) begin
                $display("TRACE %0d COMMAND %0d %0h %0d", cycle,
                         command_count, cmd_address, cmd_beats);
                command_count = command_count + 1;
            end
            if (burst_valid && burst_ready) begin
                $display("TRACE %0d BURST %0d %0h %0d %0d", cycle,
                         burst_count, burst_address, burst_beats, burst_last);
                if (burst_address != expected_address[burst_count] ||
                    burst_beats != expected_beats[burst_count] ||
                    burst_len != expected_beats[burst_count] - 1)
                    mismatch = mismatch + 1;
                burst_count = burst_count + 1;
            end
            cycle = cycle + 1;
        end
    end
endmodule

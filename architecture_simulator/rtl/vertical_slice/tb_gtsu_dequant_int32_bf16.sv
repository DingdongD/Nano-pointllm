module tb_gtsu_dequant_int32_bf16;
    parameter integer TAG_WIDTH = 16;
    parameter integer VECTORS = 12;
    parameter integer SOURCE_STALL_MOD = 4;
    parameter integer SOURCE_STALL_PHASE = 1;
    parameter integer OUTPUT_STALL_MOD = 5;
    parameter integer OUTPUT_STALL_PHASE = 2;
    parameter integer MAX_CYCLES = 10000;
    parameter integer TRACE_WIDTH = TAG_WIDTH + 32 + 16 + 16;

    reg clk = 0;
    reg rst_n = 0;
    integer cycle = 0;
    integer sent = 0;
    integer received = 0;
    integer source_backpressure_cycles = 0;
    integer output_backpressure_cycles = 0;
    reg [1023:0] trace_file;
    reg [TRACE_WIDTH-1:0] trace [0:VECTORS-1];
    wire [TRACE_WIDTH-1:0] current_trace = trace[(sent < VECTORS) ? sent : 0];
    wire [TAG_WIDTH-1:0] in_tag = current_trace[0 +: TAG_WIDTH];
    wire signed [31:0] in_accumulator = current_trace[TAG_WIDTH +: 32];
    wire [15:0] in_activation_scale_fp16 = current_trace[TAG_WIDTH+32 +: 16];
    wire [15:0] in_weight_scale_fp16 = current_trace[TAG_WIDTH+48 +: 16];
    wire source_gate = (SOURCE_STALL_MOD == 0) ? 1'b1
        : ((cycle % SOURCE_STALL_MOD) != SOURCE_STALL_PHASE);
    wire output_gate = (OUTPUT_STALL_MOD == 0) ? 1'b1
        : ((cycle % OUTPUT_STALL_MOD) != OUTPUT_STALL_PHASE);
    wire in_valid = rst_n && sent < VECTORS && source_gate;
    wire in_ready;
    wire out_valid;
    wire out_ready = rst_n && output_gate;
    wire [TAG_WIDTH-1:0] out_tag;
    wire [15:0] out_bf16;

    always #5 clk = ~clk;
    initial begin
        if (!$value$plusargs("TRACE_FILE=%s", trace_file))
            $fatal(1, "TRACE_FILE plusarg is required");
        $readmemh(trace_file, trace);
        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1;
    end

    gtsu_dequant_int32_bf16 #(.TAG_WIDTH(TAG_WIDTH)) dut (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_ready(in_ready),
        .in_tag(in_tag), .in_accumulator(in_accumulator),
        .in_activation_scale_fp16(in_activation_scale_fp16),
        .in_weight_scale_fp16(in_weight_scale_fp16),
        .out_valid(out_valid), .out_ready(out_ready), .out_tag(out_tag),
        .out_bf16(out_bf16)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            cycle <= 0;
            sent <= 0;
            received <= 0;
            source_backpressure_cycles <= 0;
            output_backpressure_cycles <= 0;
        end else begin
            if (in_valid && in_ready) begin
                $display("TRACE %0d INPUT_ACCEPT %0d 0", cycle, in_tag);
                sent <= sent + 1;
            end
            if (in_valid && !in_ready)
                source_backpressure_cycles <= source_backpressure_cycles + 1;
            if (out_valid && !out_ready)
                output_backpressure_cycles <= output_backpressure_cycles + 1;
            if (out_valid && out_ready) begin
                $display("TRACE %0d OUTPUT_ACCEPT %0d %0h", cycle, out_tag, out_bf16);
                received <= received + 1;
                if (received + 1 == VECTORS) begin
                    #1;
                    $display("SUMMARY %0d %0d %0d %0d %0d", cycle + 1,
                        sent, received, source_backpressure_cycles,
                        output_backpressure_cycles);
                    $finish;
                end
            end
            cycle <= cycle + 1;
            if (cycle >= MAX_CYCLES)
                $fatal(1, "BF16 dequant RTL timeout");
        end
    end
endmodule

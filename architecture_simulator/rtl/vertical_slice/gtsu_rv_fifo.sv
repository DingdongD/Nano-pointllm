module gtsu_rv_fifo #(
    parameter integer WIDTH = 32,
    parameter integer DEPTH = 2,
    parameter integer PTR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    parameter integer COUNT_W = $clog2(DEPTH + 1)
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 in_valid,
    output wire                 in_ready,
    input  wire [WIDTH-1:0]     in_data,
    output wire                 out_valid,
    input  wire                 out_ready,
    output wire [WIDTH-1:0]     out_data,
    output wire [COUNT_W-1:0]   occupancy
);
    reg [WIDTH-1:0] storage [0:DEPTH-1];
    reg [PTR_W-1:0] read_ptr;
    reg [PTR_W-1:0] write_ptr;
    reg [COUNT_W-1:0] count;

    wire push = in_valid && in_ready;
    wire pop = out_valid && out_ready;

    assign in_ready = count < DEPTH;
    assign out_valid = count != 0;
    assign out_data = storage[read_ptr];
    assign occupancy = count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_ptr <= 0;
            write_ptr <= 0;
            count <= 0;
        end else begin
            if (push) begin
                storage[write_ptr] <= in_data;
                write_ptr <= (write_ptr == DEPTH - 1) ? 0 : write_ptr + 1'b1;
            end
            if (pop) begin
                read_ptr <= (read_ptr == DEPTH - 1) ? 0 : read_ptr + 1'b1;
            end
            case ({push, pop})
                2'b10: count <= count + 1'b1;
                2'b01: count <= count - 1'b1;
                default: count <= count;
            endcase
        end
    end
endmodule

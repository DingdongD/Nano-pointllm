module gtsu_axi64_burst_splitter #(
    parameter integer ADDR_W = 64,
    parameter integer COUNT_W = 24,
    parameter [8:0] MAX_BURST_BEATS = 9'd256
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     cmd_valid,
    output wire                     cmd_ready,
    input  wire [ADDR_W-1:0]        cmd_address,
    input  wire [COUNT_W-1:0]       cmd_beats,
    input  wire                     enforce_4k_boundary,
    output wire                     burst_valid,
    input  wire                     burst_ready,
    output wire [ADDR_W-1:0]        burst_address,
    output wire [7:0]               burst_len,
    output wire [8:0]               burst_beats,
    output wire                     burst_last,
    output reg                      done
);
    reg active;
    reg [ADDR_W-1:0] current_address;
    reg [COUNT_W-1:0] remaining_beats;
    wire [6:0] beats_to_4k = 7'd64 - {1'b0, current_address[11:6]};
    wire [COUNT_W-1:0] beats_to_4k_w =
        {{(COUNT_W-7){1'b0}}, beats_to_4k};
    wire [COUNT_W-1:0] max_burst_beats_w =
        {{(COUNT_W-9){1'b0}}, MAX_BURST_BEATS};
    wire [COUNT_W-1:0] max_limited =
        remaining_beats > max_burst_beats_w ?
        max_burst_beats_w : remaining_beats;
    wire [COUNT_W-1:0] boundary_limited =
        enforce_4k_boundary && max_limited > beats_to_4k_w ?
        beats_to_4k_w : max_limited;

    assign cmd_ready = !active;
    assign burst_valid = active;
    assign burst_address = current_address;
    assign burst_beats = boundary_limited[8:0];
    assign burst_len = boundary_limited[7:0] - 1'b1;
    assign burst_last = active && remaining_beats == boundary_limited;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active <= 0;
            current_address <= 0;
            remaining_beats <= 0;
            done <= 0;
        end else begin
            done <= 0;
            if (cmd_valid && cmd_ready) begin
                active <= 1;
                current_address <= cmd_address;
                remaining_beats <= cmd_beats;
            end else if (burst_valid && burst_ready) begin
                if (burst_last) begin
                    active <= 0;
                    remaining_beats <= 0;
                    done <= 1;
                end else begin
                    current_address <= current_address +
                        ({{(ADDR_W-COUNT_W){1'b0}}, boundary_limited} << 6);
                    remaining_beats <= remaining_beats - boundary_limited;
                end
            end
        end
    end

    initial begin
        if (COUNT_W < 9 || ADDR_W < COUNT_W)
            $error("AXI address/count widths cannot represent a full burst");
        if (MAX_BURST_BEATS <= 0 || MAX_BURST_BEATS > 256)
            $error("AXI burst length must be in [1,256]");
    end
endmodule

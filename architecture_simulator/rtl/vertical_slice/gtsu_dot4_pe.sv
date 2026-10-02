(* keep_hierarchy = "yes" *)
module gtsu_dot4_pe (
    input  wire [31:0]              a_data,
    input  wire [31:0]              b_data,
    output wire signed [31:0]       dot_data
);
    wire signed [7:0] a0 = a_data[7:0];
    wire signed [7:0] a1 = a_data[15:8];
    wire signed [7:0] a2 = a_data[23:16];
    wire signed [7:0] a3 = a_data[31:24];
    wire signed [7:0] b0 = b_data[7:0];
    wire signed [7:0] b1 = b_data[15:8];
    wire signed [7:0] b2 = b_data[23:16];
    wire signed [7:0] b3 = b_data[31:24];

    wire signed [15:0] product0 = a0 * b0;
    wire signed [15:0] product1 = a1 * b1;
    wire signed [15:0] product2 = a2 * b2;
    wire signed [15:0] product3 = a3 * b3;
    wire signed [16:0] sum01 =
        {{1{product0[15]}}, product0} + {{1{product1[15]}}, product1};
    wire signed [16:0] sum23 =
        {{1{product2[15]}}, product2} + {{1{product3[15]}}, product3};
    wire signed [17:0] sum =
        {{1{sum01[16]}}, sum01} + {{1{sum23[16]}}, sum23};

    assign dot_data = {{14{sum[17]}}, sum};
endmodule

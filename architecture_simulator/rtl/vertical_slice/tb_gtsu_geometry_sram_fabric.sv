module tb_gtsu_geometry_sram_fabric;
    localparam integer POINTS = 128;
    localparam integer LANES = 64;
    localparam integer ROWS = 2;
    localparam integer READ_LATENCY = 3;
    localparam integer ROW_W = 1;
    localparam integer INDEX_W = 7;

    reg clk = 0;
    reg rst_n = 0;
    reg point_load_valid = 0;
    wire point_load_ready;
    reg [INDEX_W-1:0] point_load_index = 0;
    reg [127:0] point_load_word = 0;
    reg min_load_valid = 0;
    wire min_load_ready;
    reg [ROW_W-1:0] min_load_row = 0;
    reg [LANES*32-1:0] min_load_values = 0;
    reg tile_req_valid = 0;
    wire tile_req_ready;
    reg [ROW_W-1:0] tile_req_row = 0;
    wire tile_rsp_valid;
    wire [LANES*128-1:0] tile_rsp_point_words;
    wire [LANES*32-1:0] tile_rsp_min_values;
    reg min_update_valid = 0;
    wire min_update_ready;
    reg [ROW_W-1:0] min_update_row = 0;
    reg [LANES-1:0] min_update_mask = 0;
    reg [LANES*32-1:0] min_update_values = 0;
    wire collision;
    integer cycle = 0;
    integer point_loads = 0;
    integer min_loads = 0;
    integer requests = 0;
    integer responses = 0;
    integer mismatches = 0;
    integer collisions = 0;
    integer index, lane;
    integer expected_index;
    reg [127:0] expected_point;
    reg [31:0] expected_min;

    function automatic [127:0] make_point_word(input integer value);
        begin
            make_point_word = {
                32'h40000000 + value,
                32'h30000000 + value,
                32'h20000000 + value,
                32'h10000000 + value
            };
        end
    endfunction

    gtsu_geometry_sram_fabric #(
        .POINTS(POINTS), .LANES(LANES), .ROWS(ROWS),
        .READ_LATENCY(READ_LATENCY)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .point_load_valid(point_load_valid), .point_load_ready(point_load_ready),
        .point_load_index(point_load_index), .point_load_word(point_load_word),
        .min_load_valid(min_load_valid), .min_load_ready(min_load_ready),
        .min_load_row(min_load_row), .min_load_values(min_load_values),
        .tile_req_valid(tile_req_valid), .tile_req_ready(tile_req_ready),
        .tile_req_row(tile_req_row), .tile_rsp_valid(tile_rsp_valid),
        .tile_rsp_point_words(tile_rsp_point_words),
        .tile_rsp_min_values(tile_rsp_min_values),
        .min_update_valid(min_update_valid), .min_update_ready(min_update_ready),
        .min_update_row(min_update_row), .min_update_mask(min_update_mask),
        .min_update_values(min_update_values),
        .same_address_rw_collision(collision)
    );

    always #5 clk = ~clk;

    initial begin
        repeat (3) @(posedge clk);
        rst_n = 1;
        #1;
        min_load_valid = 1;
        min_update_valid = 1;
        #1;
        if (!min_load_ready || min_update_ready)
            $fatal(1, "min-state writer priority violates ready/valid contract");
        min_load_valid = 0;
        min_update_valid = 0;
        for (index = 0; index < POINTS; index = index + 1) begin
            @(negedge clk);
            point_load_valid = 1;
            point_load_index = index;
            point_load_word = make_point_word(index);
        end
        @(negedge clk);
        point_load_valid = 0;
        for (index = 0; index < ROWS; index = index + 1) begin
            for (lane = 0; lane < LANES; lane = lane + 1)
                min_load_values[lane*32 +: 32] = index * 1000 + lane;
            min_load_valid = 1;
            min_load_row = index;
            @(negedge clk);
        end
        min_load_valid = 0;
        tile_req_valid = 1;
        tile_req_row = 0;
        @(negedge clk);
        tile_req_row = 1;
        @(negedge clk);
        tile_req_valid = 0;
        wait (responses == 2);
        @(negedge clk);
        for (lane = 0; lane < LANES; lane = lane + 1) begin
            min_update_values[lane*32 +: 32] = 9000 + lane;
            min_update_mask[lane] = !(lane & 1);
        end
        min_update_valid = 1;
        min_update_row = 0;
        @(negedge clk);
        min_update_valid = 0;
        tile_req_valid = 1;
        tile_req_row = 0;
        @(negedge clk);
        tile_req_valid = 0;
        wait (responses == 3);
        repeat (2) @(posedge clk);
        $display("SUMMARY %0d %0d %0d %0d %0d %0d %0d", cycle,
                 point_loads, min_loads, requests, responses, mismatches, collisions);
        if (mismatches || collisions)
            $finish_and_return(1);
        $finish;
    end

    always @(posedge clk) begin
        if (rst_n) begin
            if (point_load_valid && point_load_ready)
                point_loads = point_loads + 1;
            if (min_load_valid && min_load_ready)
                min_loads = min_loads + 1;
            if (tile_req_valid && tile_req_ready) begin
                $display("TRACE %0d ISSUE %0d %0d", cycle, requests, tile_req_row);
                requests = requests + 1;
            end
            if (collision) collisions = collisions + 1;
            if (tile_rsp_valid) begin
                $display("TRACE %0d RESPONSE %0d", cycle, responses);
                for (lane = 0; lane < LANES; lane = lane + 1) begin
                    expected_index = (responses == 1) ? LANES + lane : lane;
                    expected_point = make_point_word(expected_index);
                    if (tile_rsp_point_words[lane*128 +: 128] !== expected_point)
                        mismatches = mismatches + 1;
                    if (responses == 1)
                        expected_min = 1000 + lane;
                    else if (responses == 2 && !(lane & 1))
                        expected_min = 9000 + lane;
                    else
                        expected_min = lane;
                    if (tile_rsp_min_values[lane*32 +: 32] !== expected_min)
                        mismatches = mismatches + 1;
                end
                responses = responses + 1;
            end
            cycle = cycle + 1;
        end
    end
endmodule

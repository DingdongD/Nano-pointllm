module gtsu_geometry_sram_fabric #(
    parameter integer POINTS = 8192,
    parameter integer LANES = 64,
    parameter integer BANKS_PER_GROUP = 16,
    parameter integer POINT_GROUPS = LANES / BANKS_PER_GROUP,
    parameter integer ROWS = (POINTS + LANES - 1) / LANES,
    parameter integer READ_LATENCY = 3,
    parameter integer ROW_W = (ROWS <= 2) ? 1 : $clog2(ROWS),
    parameter integer INDEX_W = (POINTS <= 2) ? 1 : $clog2(POINTS)
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         point_load_valid,
    output wire                         point_load_ready,
    input  wire [INDEX_W-1:0]           point_load_index,
    input  wire [127:0]                 point_load_word,
    input  wire                         min_load_valid,
    output wire                         min_load_ready,
    input  wire [ROW_W-1:0]             min_load_row,
    input  wire [LANES*32-1:0]          min_load_values,
    input  wire                         tile_req_valid,
    output wire                         tile_req_ready,
    input  wire [ROW_W-1:0]             tile_req_row,
    output wire                         tile_rsp_valid,
    output wire [LANES*128-1:0]         tile_rsp_point_words,
    output wire [LANES*32-1:0]          tile_rsp_min_values,
    input  wire                         min_update_valid,
    output wire                         min_update_ready,
    input  wire [ROW_W-1:0]             min_update_row,
    input  wire [LANES-1:0]             min_update_mask,
    input  wire [LANES*32-1:0]          min_update_values,
    output wire                         same_address_rw_collision
);
    localparam integer POINT_BANKS = POINT_GROUPS * BANKS_PER_GROUP;
    localparam integer BANK_W = $clog2(BANKS_PER_GROUP);
    localparam integer GROUP_W = $clog2(POINT_GROUPS);
    localparam integer LANE_W = $clog2(LANES);
    wire [POINT_GROUPS-1:0] point_rsp_valid;
    wire [POINT_GROUPS-1:0] point_collision;
    wire [POINT_GROUPS*BANKS_PER_GROUP*128-1:0] point_rsp_data;
    wire min_rsp_valid, min_collision;
    wire [BANKS_PER_GROUP*128-1:0] min_rsp_data;
    wire [ROW_W-1:0] point_load_row;
    wire [GROUP_W-1:0] point_load_group =
        point_load_index[BANK_W +: GROUP_W];
    wire [BANK_W-1:0] point_load_bank = point_load_index[BANK_W-1:0];
    wire min_write_valid = min_load_valid ||
        (min_update_valid && min_update_ready);
    wire [ROW_W-1:0] min_write_row = min_load_valid ? min_load_row : min_update_row;
    wire [LANES*32-1:0] min_write_values =
        min_load_valid ? min_load_values : min_update_values;
    wire [LANES-1:0] min_write_mask =
        min_load_valid ? {LANES{1'b1}} : min_update_mask;

    assign point_load_ready = 1'b1;
    assign min_load_ready = 1'b1;
    assign min_update_ready = !min_load_valid;
    assign tile_req_ready = 1'b1;
    assign tile_rsp_valid = min_rsp_valid && (&point_rsp_valid);
    assign tile_rsp_point_words = point_rsp_data;
    assign tile_rsp_min_values = min_rsp_data;
    assign same_address_rw_collision = min_collision || (|point_collision);

    generate
        if (ROWS <= 1) begin : single_row_address
            assign point_load_row = {ROW_W{1'b0}};
        end else begin : multi_row_address
            assign point_load_row = point_load_index[LANE_W +: ROW_W];
        end
    endgenerate

    genvar group;
    generate
        for (group = 0; group < POINT_GROUPS; group = group + 1) begin : point_group
            wire [BANKS_PER_GROUP-1:0] write_mask =
                (point_load_valid && point_load_group == group) ?
                ({{(BANKS_PER_GROUP-1){1'b0}}, 1'b1} << point_load_bank) : 0;
            wire [BANKS_PER_GROUP*ROW_W-1:0] read_rows =
                {BANKS_PER_GROUP{tile_req_row}};
            wire [BANKS_PER_GROUP*ROW_W-1:0] write_rows =
                {BANKS_PER_GROUP{point_load_row}};
            wire [BANKS_PER_GROUP*128-1:0] write_data =
                {{(BANKS_PER_GROUP*128-128){1'b0}}, point_load_word}
                << (point_load_bank * 128);
            wire [BANKS_PER_GROUP*16-1:0] write_bwen =
                {{(BANKS_PER_GROUP*16-16){1'b0}}, 16'hffff}
                << (point_load_bank * 16);
            wire [BANKS_PER_GROUP-1:0] unused_rsp_mask;
            wire unused_read_ready, unused_write_ready;
            gtsu_lbuf_16bank_macro_model #(
                .BANKS(BANKS_PER_GROUP), .ROWS(ROWS), .DATA_W(128),
                .READ_LATENCY(READ_LATENCY)
            ) point_sram (
                .clk(clk), .rst_n(rst_n),
                .rd_valid(tile_req_valid), .rd_ready(unused_read_ready),
                .rd_bank_mask({BANKS_PER_GROUP{1'b1}}), .rd_rows(read_rows),
                .rd_rsp_valid(point_rsp_valid[group]),
                .rd_rsp_bank_mask(unused_rsp_mask),
                .rd_rsp_data(point_rsp_data[
                    group*BANKS_PER_GROUP*128 +: BANKS_PER_GROUP*128]),
                .wr_valid(point_load_valid && point_load_group == group),
                .wr_ready(unused_write_ready),
                .wr_bank_mask(write_mask), .wr_rows(write_rows),
                .wr_data(write_data), .wr_bwen(write_bwen),
                .same_address_rw_collision(point_collision[group])
            );
        end
    endgenerate

    wire [BANKS_PER_GROUP*ROW_W-1:0] min_read_rows =
        {BANKS_PER_GROUP{tile_req_row}};
    wire [BANKS_PER_GROUP*ROW_W-1:0] min_write_rows =
        {BANKS_PER_GROUP{min_write_row}};
    wire [BANKS_PER_GROUP-1:0] min_bank_mask;
    wire [BANKS_PER_GROUP*16-1:0] min_bwen;
    genvar bank, slot, byte_number;
    generate
        for (bank = 0; bank < BANKS_PER_GROUP; bank = bank + 1) begin : min_pack
            assign min_bank_mask[bank] = |min_write_mask[bank*4 +: 4];
            for (slot = 0; slot < 4; slot = slot + 1) begin : min_slot
                for (byte_number = 0; byte_number < 4;
                     byte_number = byte_number + 1) begin : min_byte
                    assign min_bwen[bank*16 + slot*4 + byte_number] =
                        min_write_mask[bank*4 + slot];
                end
            end
        end
    endgenerate
    wire [BANKS_PER_GROUP-1:0] unused_min_rsp_mask;
    wire unused_min_read_ready, unused_min_write_ready;
    gtsu_lbuf_16bank_macro_model #(
        .BANKS(BANKS_PER_GROUP), .ROWS(ROWS), .DATA_W(128),
        .READ_LATENCY(READ_LATENCY)
    ) min_sram (
        .clk(clk), .rst_n(rst_n),
        .rd_valid(tile_req_valid), .rd_ready(unused_min_read_ready),
        .rd_bank_mask({BANKS_PER_GROUP{1'b1}}), .rd_rows(min_read_rows),
        .rd_rsp_valid(min_rsp_valid), .rd_rsp_bank_mask(unused_min_rsp_mask),
        .rd_rsp_data(min_rsp_data),
        .wr_valid(min_write_valid), .wr_ready(unused_min_write_ready),
        .wr_bank_mask(min_bank_mask),
        .wr_rows(min_write_rows), .wr_data(min_write_values), .wr_bwen(min_bwen),
        .same_address_rw_collision(min_collision)
    );

    initial begin
        if (LANES != 64 || BANKS_PER_GROUP != 16 || POINT_GROUPS != 4)
            $error("geometry SRAM fabric is locked to 4x16 banks for 64 lanes");
        if (ROWS > 1 && INDEX_W < LANE_W + ROW_W)
            $error("point index width cannot encode all SRAM rows");
    end
endmodule

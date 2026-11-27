/*
 * machllm
 * Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
 *
 * Simulation top: engine + DDR3 controller (stream port) + DRAM model.
 * The CPU side is the testbench: the engine's bus and the controller's
 * wishbone port are top-level ports.
 */

`default_nettype none

module tb_mem_top #(
	parameter CLK_MHZ = 48           // controller clock; the DRAM runs at twice this
) (
	input wire clk,
	input wire rst,

	input wire bus_req,
	input wire bus_we,
	input wire [16:2] bus_adr,
	input wire [31:0] bus_wdat,
	input wire [3:0] bus_sel,
	output wire bus_ack,
	output wire [31:0] bus_rdat,

	input wire [26:0] wb_adr,
	input wire [31:0] wb_wdat,
	output wire [31:0] wb_rdat,
	input wire wb_we,
	input wire [3:0] wb_sel,
	input wire wb_stb,
	output wire wb_ack,

	input wire [1:0] st_gap,
	output wire init_done,
	output wire busy
);
	wire sq_valid, sq_ready, sd_valid;
	wire [24:0] sq_blk;
	wire [8:0] sq_len;
	wire [127:0] sd_data;

	mlm_engine eng (
		.clk(clk), .rst(rst),
		.bus_req_i(bus_req), .bus_we_i(bus_we), .bus_adr_i(bus_adr), .bus_wdat_i(bus_wdat),
		.bus_sel_i(bus_sel), .bus_ack(bus_ack), .bus_rdat(bus_rdat),
		.sq_valid(sq_valid), .sq_ready(sq_ready), .sq_blk(sq_blk), .sq_len(sq_len),
		.sd_valid(sd_valid), .sd_data(sd_data), .busy_o(busy)
	);

	wire cs_n, ras_n, cas_n, we_n, wren, rden, rvalid;
	wire [15:0] a, wmask;
	wire [2:0] ba;
	wire [127:0] wdata, rdata;
	wire rst_n_o, cke_o, odt_o, rd_timeout;

	mlm_ddr3_ctrl #(.ROW_BITS(15), .TRFC_NS(260), .INIT_DIV(100), .CLK_MHZ(CLK_MHZ)) ctrl (
		.clk_i(clk), .rst_i(rst),
		.wb_adr_i(wb_adr), .wb_dat_i(wb_wdat), .wb_dat_o(wb_rdat), .wb_we_i(wb_we),
		.wb_sel_i(wb_sel), .wb_stb_i(wb_stb), .wb_cyc_i(wb_stb), .wb_ack_o(wb_ack),
		.sq_valid_i(sq_valid), .sq_ready_o(sq_ready), .sq_blk_i(sq_blk), .sq_len_i(sq_len),
		.sd_valid_o(sd_valid), .sd_data_o(sd_data),
		.wq_valid_i(1'b0), .wq_ready_o(), .wq_blk_i(25'd0), .wq_data_i(128'd0),
		.phy_ready_i(1'b1), .init_done_o(init_done), .rd_timeout_o(rd_timeout),
		.no_refresh_i(1'b0), .wr_noread_i(1'b0), .rdbuf_off_i(1'b0), .st_gap_i(st_gap),
		.phy_reset_n_o(rst_n_o), .phy_cke_o(cke_o), .phy_odt_o(odt_o),
		.phy_cs_n_o(cs_n), .phy_ras_n_o(ras_n), .phy_cas_n_o(cas_n), .phy_we_n_o(we_n),
		.phy_a_o(a), .phy_ba_o(ba),
		.phy_wren_o(wren), .phy_wdata_o(wdata), .phy_wmask_o(wmask),
		.phy_rden_o(rden), .phy_rvalid_i(rvalid), .phy_rdata_i(rdata)
	);

	mlm_dram_model #(.TCK_PS(500000 / CLK_MHZ)) dram (
		.clk(clk), .cs_n(cs_n), .ras_n(ras_n), .cas_n(cas_n), .we_n(we_n),
		.a(a), .ba(ba), .wren(wren), .wdata(wdata), .wmask(wmask), .rden(rden),
		.rvalid(rvalid), .rdata(rdata)
	);
endmodule

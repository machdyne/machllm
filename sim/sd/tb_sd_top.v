/*
 * machllm
 * Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
 *
 * Simulation top: SD host -> DDR3 controller block writes -> DRAM model.
 * The card is the C++ model on the pins; the CPU is the testbench.
 */
`default_nettype none
module tb_sd_top (
	input wire clk,
	input wire rst,
	input wire bus_req,
	input wire bus_we,
	input wire [5:2] bus_adr,
	input wire [31:0] bus_wdat,
	output wire bus_ack,
	output wire [31:0] bus_rdat,
	output wire sd_sck,
	output wire sd_cmd_o,
	output wire sd_cmd_oe,
	input wire sd_cmd_i,
	input wire [3:0] sd_dat_i,
	output wire init_done
);
	wire wq_valid, wq_ready;
	wire [24:0] wq_blk;
	wire [127:0] wq_data;
	mlm_sd sd (
		.clk(clk), .rst(rst),
		.bus_req(bus_req), .bus_we(bus_we), .bus_adr(bus_adr), .bus_wdat(bus_wdat),
		.bus_ack(bus_ack), .bus_rdat(bus_rdat),
		.wq_valid(wq_valid), .wq_ready(wq_ready), .wq_blk(wq_blk), .wq_data(wq_data),
		.sd_sck(sd_sck), .sd_cmd_o(sd_cmd_o), .sd_cmd_oe(sd_cmd_oe), .sd_cmd_i(sd_cmd_i),
		.sd_dat_i(sd_dat_i)
	);
	wire cs_n, ras_n, cas_n, we_n, wren, rden, rvalid;
	wire [15:0] a, wmask;
	wire [2:0] ba;
	wire [127:0] wdata, rdata;
	wire [31:0] wb_rdat;
	wire wb_ack, rst_n_o, cke_o, odt_o, rd_timeout, sq_ready, sd_valid;
	wire [127:0] sd_data;
	mlm_ddr3_ctrl #(.ROW_BITS(15), .TRFC_NS(260), .INIT_DIV(100)) ctrl (
		.clk_i(clk), .rst_i(rst),
		.wb_adr_i(27'd0), .wb_dat_i(32'd0), .wb_dat_o(wb_rdat), .wb_we_i(1'b0),
		.wb_sel_i(4'd0), .wb_stb_i(1'b0), .wb_cyc_i(1'b0), .wb_ack_o(wb_ack),
		.sq_valid_i(1'b0), .sq_ready_o(sq_ready), .sq_blk_i(25'd0), .sq_len_i(9'd0),
		.sd_valid_o(sd_valid), .sd_data_o(sd_data),
		.wq_valid_i(wq_valid), .wq_ready_o(wq_ready), .wq_blk_i(wq_blk), .wq_data_i(wq_data),
		.phy_ready_i(1'b1), .init_done_o(init_done), .rd_timeout_o(rd_timeout),
		.no_refresh_i(1'b0), .wr_noread_i(1'b0), .rdbuf_off_i(1'b0), .st_gap_i(2'd0),
		.phy_reset_n_o(rst_n_o), .phy_cke_o(cke_o), .phy_odt_o(odt_o),
		.phy_cs_n_o(cs_n), .phy_ras_n_o(ras_n), .phy_cas_n_o(cas_n), .phy_we_n_o(we_n),
		.phy_a_o(a), .phy_ba_o(ba),
		.phy_wren_o(wren), .phy_wdata_o(wdata), .phy_wmask_o(wmask),
		.phy_rden_o(rden), .phy_rvalid_i(rvalid), .phy_rdata_i(rdata)
	);
	mlm_dram_model dram (
		.clk(clk), .cs_n(cs_n), .ras_n(ras_n), .cas_n(cas_n), .we_n(we_n),
		.a(a), .ba(ba), .wren(wren), .wdata(wdata), .wmask(wmask), .rden(rden),
		.rvalid(rvalid), .rdata(rdata)
	);
endmodule

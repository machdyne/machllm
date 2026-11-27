/*
 * machllm
 * Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
 *
 * Sergei ML2 (LFE5U-45F, MT41K256M16TW-107 512 MB DDR3L): clocks,
 * reset and pins around rtl/mlm_soc.v. Pin names as Zeitlos's
 * boards/sergei_ml2.lpf.
 */

`default_nettype none

// system clock (MHz); the DDR3 clock is twice this. The Makefile sets
// it (and the firmware's path) in $(OUT)/cfg.vh.
`ifdef MLM_CFG
`include "cfg.vh"
`endif
`ifndef SYS_MHZ
`define SYS_MHZ 48
`endif
`ifndef MEM_MHZ
`define MEM_MHZ `SYS_MHZ
`endif
`ifndef FW_HEX
`define FW_HEX "build/sergei_ml2/fw.hex"
`endif

module top (
	input wire CLK_48,
	output wire LED_B,
	output wire UART0_TX,
	input wire UART0_RX,

	output wire SD_SCK,
	inout wire SD_MOSI,              // CMD
	input wire SD_MISO,              // DAT0
	input wire SD_D1,
	input wire SD_D2,
	input wire SD_SS,                // DAT3 (only read: never driven)

	output wire [15:0] ddr3_a,
	output wire [2:0] ddr3_ba,
	output wire ddr3_ras_n,
	output wire ddr3_cas_n,
	output wire ddr3_we_n,
	output wire ddr3_cs_n,
	output wire ddr3_cke,
	output wire ddr3_odt,
	output wire ddr3_reset_n,
	output wire [1:0] ddr3_dm,
	inout wire [15:0] ddr3_dq,
	inout wire [1:0] ddr3_dqs_p,
	output wire ddr3_clk_p
);
	// One clock domain (MEM_MHZ = SYS_MHZ = 48): everything on the DDR3
	// system clock. Two: the memory side on the DDR3 system
	// clock (MEM_MHZ), the CPU on the PLL's free-running ~50 MHz.
	localparam TWO_CLK = (`MEM_MHZ != `SYS_MHZ);
	wire mclk, init_clk, eclk, ddrdel, pause, phy_rst, phy_ready, locked;

	mlm_ddr3_clk #(.EDGE_MHZ(2 * `MEM_MHZ)) clocks (
		.clk48_i(CLK_48), .rst_i(1'b0),
		.sys_clk_o(mclk), .eclk_o(eclk), .ddrdel_o(ddrdel), .pause_o(pause),
		.phy_rst_o(phy_rst), .ready_o(phy_ready), .locked_o(locked), .init_clk_o(init_clk)
	);
	wire clk = TWO_CLK ? init_clk : mclk;

	// reset: held until the PLL locks, then 1024 cycles more (each domain)
	reg [10:0] rcnt = 11'd0;
	wire rst = !rcnt[10];
	always @(posedge clk)
		if (!locked) rcnt <= 11'd0;
		else if (!rcnt[10]) rcnt <= rcnt + 11'd1;
	reg [10:0] mcnt = 11'd0;
	wire mrst = !mcnt[10];
	always @(posedge mclk)
		if (!locked) mcnt <= 11'd0;
		else if (!mcnt[10]) mcnt <= mcnt + 11'd1;

	wire cmd_o, cmd_oe;
	assign SD_MOSI = cmd_oe ? cmd_o : 1'bz;

	mlm_soc #(.INIT_FILE(`FW_HEX), .SYS_MHZ(`SYS_MHZ), .TWO_CLK(TWO_CLK), .MEM_MHZ(`MEM_MHZ)) soc (
		.clk(clk), .rst(rst), .mclk(mclk), .mrst(mrst),
		.eclk(eclk), .ddrdel(ddrdel), .pause(pause), .phy_rst(phy_rst),
		.phy_ready(phy_ready), .pll_locked(locked),
		.ddr3_a(ddr3_a), .ddr3_ba(ddr3_ba), .ddr3_ras_n(ddr3_ras_n), .ddr3_cas_n(ddr3_cas_n),
		.ddr3_we_n(ddr3_we_n), .ddr3_cs_n(ddr3_cs_n), .ddr3_cke(ddr3_cke), .ddr3_odt(ddr3_odt),
		.ddr3_reset_n(ddr3_reset_n), .ddr3_dm(ddr3_dm), .ddr3_dq(ddr3_dq),
		.ddr3_dqs_p(ddr3_dqs_p), .ddr3_clk_p(ddr3_clk_p),
		.uart_tx(UART0_TX), .uart_rx(UART0_RX),
		.sd_sck(SD_SCK), .sd_cmd_o(cmd_o), .sd_cmd_oe(cmd_oe), .sd_cmd_i(SD_MOSI),
		.sd_dat_i({SD_SS, SD_D2, SD_D1, SD_MISO}),
		.led(LED_B)
	);
endmodule

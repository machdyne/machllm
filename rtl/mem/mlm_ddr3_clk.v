/*
 * machllm -- DDR3 clocking: Zeitlos's ddr3_clk (rtl/mem/ddr3_clk.v),
 * with the edge clock as a parameter. EDGE_MHZ 96 builds exactly
 * Zeitlos's clocks (pll2); 128, 150 and 200 use mlm_pll. The system
 * clock is always EDGE_MHZ / 2 (CLKDIVF), phase-locked to the edge
 * clock; see ddr3_clk.v for why.
 */
`default_nettype none
module mlm_ddr3_clk #(
	parameter EDGE_MHZ = 96
) (
	input wire clk48_i,
	input wire rst_i,
	output wire sys_clk_o,
	output wire eclk_o,
	output wire ddrdel_o,
	output wire pause_o,
	output wire phy_rst_o,
	output wire ready_o,
	output wire locked_o,
	output wire init_clk_o,   // the free-running ~48/50 MHz (the CPU's clock with two domains)
	output wire clk12_o       // 12 MHz for USB (mlm_pll builds only; 0 with EDGE_MHZ 96)
);
	wire clk12;
	wire eclk_raw, init_clk, pll_locked;
	wire eclk_stop, eclk_reset;
	generate if (EDGE_MHZ == 96) begin : g_pll2
		pll2 pll (.clkin(clk48_i), .clkout0(eclk_raw), .clkout2(init_clk), .locked(pll_locked));
		assign clk12 = 1'b0;
	end else begin : g_pll
		mlm_pll #(.EDGE_MHZ(EDGE_MHZ)) pll (.clkin(clk48_i), .clkout0(eclk_raw), .clkout2(init_clk),
		                                     .clkout3(clk12), .locked(pll_locked));
	end endgenerate
	ECLKSYNCB eclksync (.ECLKI(eclk_raw), .STOP(eclk_stop), .ECLKO(eclk_o));
	CLKDIVF #(.DIV("2.0")) clkdiv (.CLKI(eclk_o), .RST(eclk_reset), .ALIGNWD(1'b0), .CDIVX(sys_clk_o));
	ddr3_phy_init init (
		.clk_i(init_clk), .rst_i(rst_i | ~pll_locked), .eclk_i(eclk_o),
		.ddrdel_o(ddrdel_o), .eclk_stop_o(eclk_stop), .eclk_reset_o(eclk_reset),
		.pause_o(pause_o), .ready_o(ready_o)
	);
	assign locked_o = pll_locked;
	assign init_clk_o = init_clk;
	assign clk12_o = clk12;
	assign phy_rst_o = eclk_reset;
endmodule

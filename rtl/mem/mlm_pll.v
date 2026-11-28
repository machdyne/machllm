/*
 * machllm -- the DDR3 edge clock above Zeitlos's 96 MHz.
 *
 * Same shape as Zeitlos's pll2 (CLKOP = edge clock, CLKOS2 = a free
 * running ~48 MHz for the PHY init sequencer), for EDGE_MHZ 128, 150
 * or 200 from the 48 MHz oscillator:
 *   128: 48 / 3 x 8  = 128, VCO 640 (CLKOP_DIV 5), CLKOS2 640/13 = 49.2
 *   150: 48 / 8 x 25 = 150, VCO 600 (CLKOP_DIV 4), CLKOS2 600/12 = 50
 *   180: 48 / 4 x 15 = 180, VCO 540 (CLKOP_DIV 3), CLKOS2 540/10 = 54 (the CPU's clock; 540/9 = 60 misses timing in the CPU's load path)
 *   200: 48 / 6 x 25 = 200, VCO 600 (CLKOP_DIV 3), CLKOS2 600/12 = 50
 */
`default_nettype none
module mlm_pll #(
	parameter EDGE_MHZ = 150
) (
	input wire clkin,
	output wire clkout0,
	output wire clkout2,
	output wire clkout3,      // 12 MHz (USB, rtl/mlm_ui.v): VCO / 45 (540), / 50 (600), / 53 (640: 12.08)
	output wire locked
);
	localparam CLKI_DIV  = (EDGE_MHZ == 200) ? 6 : (EDGE_MHZ == 180) ? 4 : (EDGE_MHZ == 128) ? 3 : 8;
	localparam CLKOP_DIV = (EDGE_MHZ == 200 || EDGE_MHZ == 180) ? 3 : (EDGE_MHZ == 128) ? 5 : 4;
	localparam CLKFB_DIV = (EDGE_MHZ == 180) ? 15 : (EDGE_MHZ == 128) ? 8 : 25;
	localparam OS2_DIV   = (EDGE_MHZ == 180) ? 10 : (EDGE_MHZ == 128) ? 13 : 12;
	localparam OS3_DIV   = (EDGE_MHZ == 180) ? 45 : (EDGE_MHZ == 128) ? 53 : 50;
	(* ICP_CURRENT="12" *) (* LPF_RESISTOR="8" *) (* MFG_ENABLE_FILTEROPAMP="1" *) (* MFG_GMCREF_SEL="2" *)
	EHXPLLL #(
		.PLLRST_ENA("DISABLED"), .INTFB_WAKE("DISABLED"), .STDBY_ENABLE("DISABLED"),
		.DPHASE_SOURCE("DISABLED"),
		.OUTDIVIDER_MUXA("DIVA"), .OUTDIVIDER_MUXB("DIVB"),
		.OUTDIVIDER_MUXC("DIVC"), .OUTDIVIDER_MUXD("DIVD"),
		.CLKI_DIV(CLKI_DIV),
		.CLKOP_ENABLE("ENABLED"), .CLKOP_DIV(CLKOP_DIV), .CLKOP_CPHASE(CLKOP_DIV - 1), .CLKOP_FPHASE(0),
		.CLKOS_ENABLE("DISABLED"), .CLKOS_DIV(6), .CLKOS_CPHASE(2), .CLKOS_FPHASE(0),
		.CLKOS2_ENABLE("ENABLED"), .CLKOS2_DIV(OS2_DIV), .CLKOS2_CPHASE(OS2_DIV - 1), .CLKOS2_FPHASE(0),
		.CLKOS3_ENABLE("ENABLED"), .CLKOS3_DIV(OS3_DIV), .CLKOS3_CPHASE(OS3_DIV - 1), .CLKOS3_FPHASE(0),
		.FEEDBK_PATH("CLKOP"), .CLKFB_DIV(CLKFB_DIV)
	) pll_i (
		.RST(1'b0), .STDBY(1'b0), .CLKI(clkin),
		.CLKOP(clkout0), .CLKOS(), .CLKOS2(clkout2), .CLKOS3(clkout3),
		.CLKFB(clkout0), .CLKINTFB(),
		.PHASESEL0(1'b0), .PHASESEL1(1'b0), .PHASEDIR(1'b1), .PHASESTEP(1'b1),
		.PHASELOADREG(1'b1), .PLLWAKESYNC(1'b0), .ENCLKOP(1'b0),
		.LOCK(locked)
	);
endmodule

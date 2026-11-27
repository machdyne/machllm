/*
 * machllm -- Phase 5 timing experiment (not a product build).
 *
 * The DDR3 controller and PHY alone, with the system clock = PHY SCLK =
 * EDGE_MHZ / 2, a stream reader that keeps the stream port busy and a
 * CPU-port exerciser. Answers one question: does the memory side close
 * timing at 75 MHz (DDR3 at 150) and at 100 MHz (DDR3 at 200)?
 * With ENGINE=1 the engine sits on the stream in the same clock domain.
 */
`default_nettype none
module top #(
	parameter EDGE_MHZ = 150,
	parameter ENGINE = 0
) (
	input wire CLK_48,
	output wire LED_B,
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
	wire eclk_raw, init_clk, locked, eclk, clk, ddrdel, eclk_stop, eclk_reset, pause, phy_ready;
	mlm_pll #(.EDGE_MHZ(EDGE_MHZ)) pll (.clkin(CLK_48), .clkout0(eclk_raw), .clkout2(init_clk), .locked(locked));
	ECLKSYNCB eclksync (.ECLKI(eclk_raw), .STOP(eclk_stop), .ECLKO(eclk));
	CLKDIVF #(.DIV("2.0")) clkdiv (.CLKI(eclk), .RST(eclk_reset), .ALIGNWD(1'b0), .CDIVX(clk));
	ddr3_phy_init init (
		.clk_i(init_clk), .rst_i(~locked), .eclk_i(eclk),
		.ddrdel_o(ddrdel), .eclk_stop_o(eclk_stop), .eclk_reset_o(eclk_reset),
		.pause_o(pause), .ready_o(phy_ready)
	);

	reg [10:0] rcnt = 11'd0;
	wire rst = !rcnt[10];
	always @(posedge clk)
		if (!locked) rcnt <= 11'd0;
		else if (!rcnt[10]) rcnt <= rcnt + 11'd1;

	// CPU-port exerciser: an LFSR picks address, data, we, sel
	reg [31:0] lf = 32'h1;
	always @(posedge clk) lf <= {lf[30:0], lf[31] ^ lf[21] ^ lf[1] ^ lf[0]};
	reg cpu_req;
	wire cpu_ack;
	wire [31:0] cpu_rd;
	reg [31:0] cpu_acc;
	always @(posedge clk)
		if (rst) begin cpu_req <= 1'b0; cpu_acc <= 32'd0; end
		else if (cpu_ack) begin cpu_req <= 1'b0; cpu_acc <= cpu_acc ^ cpu_rd; end
		else if (lf[7:0] == 8'h5a) cpu_req <= 1'b1;
	reg [26:0] cpu_adr;
	reg [31:0] cpu_wd;
	reg cpu_we;
	reg [3:0] cpu_sel;
	always @(posedge clk) if (!cpu_req) begin
		cpu_adr <= lf[26:0]; cpu_wd <= {lf[15:0], lf[31:16]}; cpu_we <= lf[30]; cpu_sel <= lf[11:8];
	end

	// stream port
	wire sq_valid, sq_ready, sdv;
	wire [24:0] sq_blk;
	wire [8:0] sq_len;
	wire [127:0] sdd;
	reg [127:0] sacc;

	generate if (ENGINE) begin : g_eng
		// engine registers driven from the LFSR (keeps every path alive)
		wire eng_ack;
		wire [31:0] eng_rd;
		reg ereq;
		always @(posedge clk)
			if (rst) ereq <= 1'b0;
			else if (eng_ack) ereq <= 1'b0;
			else if (lf[5:0] == 6'h21) ereq <= 1'b1;
		reg [14:0] eadr;
		reg [31:0] ewd;
		reg ewe;
		always @(posedge clk) if (!ereq) begin eadr <= lf[14:0]; ewd <= lf; ewe <= lf[29]; end
		always @(posedge clk) sacc <= sacc ^ {96'd0, eng_rd};
		mlm_engine eng (
			.clk(clk), .rst(rst),
			.bus_req(ereq), .bus_we(ewe), .bus_adr(eadr), .bus_wdat(ewd),
			.bus_sel(4'hf), .bus_ack(eng_ack), .bus_rdat(eng_rd),
			.sq_valid(sq_valid), .sq_ready(sq_ready), .sq_blk(sq_blk), .sq_len(sq_len),
			.sd_valid(sdv), .sd_data(sdd), .busy_o()
		);
	end else begin : g_rd
		reg [24:0] blk;
		always @(posedge clk)
			if (rst) blk <= 25'd0;
			else if (sq_ready) blk <= blk + 25'd16;
		assign sq_valid = !rst;
		assign sq_blk = blk;
		assign sq_len = 9'd16;
		always @(posedge clk) if (sdv) sacc <= sacc ^ sdd;
	end endgenerate

	mlm_ddr3 #(.ROW_BITS(15), .TRFC_NS(260), .WCOMBINE(0), .CLK_MHZ(EDGE_MHZ / 2)) ddr3 (
		.clk_i(clk), .rst_i(rst),
		.eclk_i(eclk), .ddrdel_i(ddrdel), .pause_i(pause), .phy_rst_i(eclk_reset),
		.phy_ready_i(phy_ready), .pll_locked_i(locked),
		.wb_adr_i(cpu_adr), .wb_dat_i(cpu_wd), .wb_dat_o(cpu_rd),
		.wb_we_i(cpu_we), .wb_sel_i(cpu_sel), .wb_stb_i(cpu_req), .wb_cyc_i(cpu_req),
		.wb_ack_o(cpu_ack),
		.sq_valid_i(sq_valid), .sq_ready_o(sq_ready), .sq_blk_i(sq_blk), .sq_len_i(sq_len),
		.sd_valid_o(sdv), .sd_data_o(sdd),
		.wq_valid_i(1'b0), .wq_ready_o(), .wq_blk_i(25'd0), .wq_data_i(128'd0),
		.reg_adr_i(2'd0), .reg_dat_i(32'd0), .reg_dat_o(),
		.reg_we_i(1'b0), .reg_stb_i(1'b0), .reg_cyc_i(1'b0), .reg_ack_o(),
		.ddr3_a(ddr3_a), .ddr3_ba(ddr3_ba), .ddr3_ras_n(ddr3_ras_n), .ddr3_cas_n(ddr3_cas_n),
		.ddr3_we_n(ddr3_we_n), .ddr3_cs_n(ddr3_cs_n), .ddr3_cke(ddr3_cke), .ddr3_odt(ddr3_odt),
		.ddr3_reset_n(ddr3_reset_n), .ddr3_dm(ddr3_dm), .ddr3_dq(ddr3_dq),
		.ddr3_dqs_p(ddr3_dqs_p), .ddr3_clk_p(ddr3_clk_p)
	);

	assign LED_B = ^sacc ^ ^cpu_acc;
endmodule

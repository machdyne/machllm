/*
 * Zeitlos SOC
 * Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
 *
 * DDR3 subsystem
 *
 * machllm: Zeitlos's rtl/mem/ddr3.v (b62a74bf) with the machllm
 * controller (rtl/mem/mlm_ddr3_ctrl.v) and its stream and block-write
 * ports passed through. Registers and PHY unchanged.
 *
 * Two wishbone ports:
 *
 *   memory     the DRAM, at MEM_MAIN
 *   registers  status and tuning, for the BIOS to train the read path
 *
 * and the controller (ddr3_ctrl.v) and PHY (ddr3_phy_ecp5.v) between
 * them. Clocks come from ddr3_clk.v.
 *
 * -- registers, word offsets --
 *
 *   0  STATUS  read only
 *        [31:16] 0xdd30        identifies THIS design; see below
 *        [7]     pll locked
 *        [6]     read timeout  sticky: a read the PHY never answered
 *        [5:4]   burstdet      sticky per lane, cleared by each read
 *        [3:2]   datavalid     live, per lane
 *        [1]     PHY ready     DLL locked and the sequence has run
 *        [0]     init done     DRAM initialised by the controller
 *
 *   1  TUNE    read/write, every field used at its full width
 *        [2:0]   write delay   cycles, WRITE command to write data
 *        [6:4]   read gate 0   cycles, READ command to gate, lane 0
 *        [10:8]  read gate 1
 *        [14:12] rdclksel 0    DQSBUFM read clock phase, lane 0
 *        [18:16] rdclksel 1
 *        [23:20] read offset 0 beats into the capture history, lane 0
 *        [27:24] read offset 1
 *
 *   2  CTRL    read/write
 *        [0]     hold refresh  while training; deferred, not skipped
 *        [1]     clear status  write 1, self-clearing
 *        [2]     training writes  writes skip their read; see
 *                                 ddr3_ctrl.v. Whole blocks only.
 *        [3]     reads bypass     every read goes to the DRAM, never
 *                                 the controller's block register
 *        [5:4]   stream gap       machllm: idle cycles after each stream
 *                                 READ (0 = back to back)
 *
 * STATUS is assembled by BIT POSITION, one assignment per field.
 * Last time a concatenation silently truncated when a field was added,
 * and a live signal placed inside MAGIC overwrote the build identifier
 * so the running bitstream reported itself as an older one -- twice.
 * MAGIC differs from that design's (0xdd1x) so the two cannot be
 * mistaken for each other on a console.
 */

`default_nettype none

module mlm_ddr3 #(
	parameter ROW_BITS = 15,
	parameter TRFC_NS = 260,
	// ddr3_ctrl.v, "write combining": stores into the block register,
	// a block to the DRAM at a time
	parameter WCOMBINE = 0,
	// system (= PHY SCLK) clock; the DRAM clock is twice this
	parameter CLK_MHZ = 48
) (
	input wire clk_i,
	input wire rst_i,

	// -- from ddr3_clk -------------------------------------------
	input wire eclk_i,
	input wire ddrdel_i,
	input wire pause_i,
	input wire phy_rst_i,
	input wire phy_ready_i,
	input wire pll_locked_i,

	// -- memory ----------------------------------------------------
	input wire [26:0] wb_adr_i,
	input wire [31:0] wb_dat_i,
	output wire [31:0] wb_dat_o,
	input wire wb_we_i,
	input wire [3:0] wb_sel_i,
	input wire wb_stb_i,
	input wire wb_cyc_i,
	output wire wb_ack_o,
	output wire [127:0] wb_line_o,
	output wire wb_line_v_o,

	// -- stream reads, block writes (mlm_ddr3_ctrl.v) ----------------
	input wire sq_valid_i,
	output wire sq_ready_o,
	input wire [ROW_BITS+9:0] sq_blk_i,
	input wire [8:0] sq_len_i,
	output wire sd_valid_o,
	output wire [127:0] sd_data_o,
	input wire wq_valid_i,
	output wire wq_ready_o,
	input wire [ROW_BITS+9:0] wq_blk_i,
	input wire [127:0] wq_data_i,

	// -- registers -------------------------------------------------
	input wire [1:0] reg_adr_i,
	input wire [31:0] reg_dat_i,
	output reg [31:0] reg_dat_o,
	input wire reg_we_i,
	input wire reg_stb_i,
	input wire reg_cyc_i,
	output reg reg_ack_o,

	// -- pins ------------------------------------------------------
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

	localparam [15:0] MAGIC = 16'hdd30;

	// Defaults are starting points for training, not answers.
	localparam [31:0] TUNE_RESET = {
		4'd0,
		4'd8,       // [27:24] read offset 1: mid-range
		4'd8,       // [23:20] read offset 0
		1'b0, 3'd2, // [18:16] rdclksel 1: LiteX found 2 (+-1) on ML2
		1'b0, 3'd2, // [14:12] rdclksel 0
		1'b0, 3'd3, // [10:8]  read gate 1: LiteDRAM's position, CL=6
		1'b0, 3'd3, // [6:4]   read gate 0
		1'b0, 3'd3  // [2:0]   write delay: CWL is 3 system cycles
	};

	reg [31:0] tune;

	// A TUNE write is applied in the middle of a PAUSE of the DQS
	// buffers, as LiteDRAM does: its PAUSE input is held for as long as
	// software is adjusting a module's delay, so READCLKSEL never
	// changes under a running buffer. Changing it live can glitch the
	// buffer mid-change. The first working scans here showed read
	// windows one or two phases wide that moved between boots, against
	// about three on this board under LiteX.
	//
	// The register write is not acknowledged until the pause has ended,
	// so the CPU cannot issue its next DRAM read into a paused buffer.
	reg [31:0] tune_next;
	reg [3:0] tune_seq;
	wire tune_pause = (tune_seq != 4'd0);
	reg no_refresh;
	reg wr_noread;
	reg rdbuf_off;
	reg sts_clear;
	reg [1:0] st_gap;           // machllm: CTRL[5:4]

	// -- controller <-> PHY --------------------------------------
	wire phy_reset_n, phy_cke, phy_odt;
	wire phy_cs_n, phy_ras_n, phy_cas_n, phy_we_n;
	wire [15:0] phy_a;
	wire [2:0] phy_ba;
	wire phy_wren, phy_rden, phy_rvalid;
	wire [127:0] phy_wdata, phy_rdata;
	wire [15:0] phy_wmask;

	wire init_done, rd_timeout;
	wire [1:0] datavalid, burstdet;

	// The CPU port, registered once: its decode (read buffer hit, write
	// queue) is otherwise a long path from wherever the request comes
	// from. A request held until ack; the registered copy drops with the
	// ack, so the controller never sees one twice.
	reg [26:0] c_adr;
	reg [31:0] c_dat;
	reg c_we, c_stb;
	reg [3:0] c_sel;
	always @(posedge clk_i) begin
		c_stb <= wb_stb_i && wb_cyc_i && !wb_ack_o && !rst_i;
		c_adr <= wb_adr_i; c_dat <= wb_dat_i; c_we <= wb_we_i; c_sel <= wb_sel_i;
	end

	mlm_ddr3_ctrl #(
		.CLK_MHZ(CLK_MHZ),
		.ROW_BITS(ROW_BITS),
		.TRFC_NS(TRFC_NS),
		.WCOMBINE(WCOMBINE)
	) ctrl (
		.clk_i(clk_i), .rst_i(rst_i),
		.wb_adr_i(c_adr), .wb_dat_i(c_dat), .wb_dat_o(wb_dat_o),
		.wb_we_i(c_we), .wb_sel_i(c_sel),
		.wb_stb_i(c_stb), .wb_cyc_i(c_stb), .wb_ack_o(wb_ack_o),
		.wb_line_o(wb_line_o), .wb_line_v_o(wb_line_v_o),
		.sq_valid_i(sq_valid_i), .sq_ready_o(sq_ready_o), .sq_blk_i(sq_blk_i),
		.sq_len_i(sq_len_i), .sd_valid_o(sd_valid_o), .sd_data_o(sd_data_o),
		.wq_valid_i(wq_valid_i), .wq_ready_o(wq_ready_o), .wq_blk_i(wq_blk_i),
		.wq_data_i(wq_data_i),
		.phy_ready_i(phy_ready_i),
		.init_done_o(init_done),
		.rd_timeout_o(rd_timeout),
		.no_refresh_i(no_refresh),
		.wr_noread_i(wr_noread),
		.rdbuf_off_i(rdbuf_off),
		.st_gap_i(st_gap),
		.phy_reset_n_o(phy_reset_n), .phy_cke_o(phy_cke),
		.phy_odt_o(phy_odt), .phy_cs_n_o(phy_cs_n),
		.phy_ras_n_o(phy_ras_n), .phy_cas_n_o(phy_cas_n),
		.phy_we_n_o(phy_we_n),
		.phy_a_o(phy_a), .phy_ba_o(phy_ba),
		.phy_wren_o(phy_wren), .phy_wdata_o(phy_wdata),
		.phy_wmask_o(phy_wmask),
		.phy_rden_o(phy_rden),
		.phy_rvalid_i(phy_rvalid), .phy_rdata_i(phy_rdata)
	);

	ddr3_phy_ecp5 phy (
		.clk_i(clk_i), .eclk_i(eclk_i), .rst_i(rst_i),
		.ddrdel_i(ddrdel_i), .pause_i(pause_i | tune_pause),
		.phy_rst_i(phy_rst_i),

		.ctl_reset_n_i(phy_reset_n), .ctl_cke_i(phy_cke),
		.ctl_odt_i(phy_odt), .ctl_cs_n_i(phy_cs_n),
		.ctl_ras_n_i(phy_ras_n), .ctl_cas_n_i(phy_cas_n),
		.ctl_we_n_i(phy_we_n),
		.ctl_a_i(phy_a), .ctl_ba_i(phy_ba),
		.ctl_wren_i(phy_wren), .ctl_wdata_i(phy_wdata),
		.ctl_wmask_i(phy_wmask),
		.ctl_rden_i(phy_rden),
		.ctl_rvalid_o(phy_rvalid), .ctl_rdata_o(phy_rdata),

		.cfg_wr_delay_i(tune[2:0]),
		.cfg_rd_gate0_i(tune[6:4]),
		.cfg_rd_gate1_i(tune[10:8]),
		.cfg_rdclksel0_i(tune[14:12]),
		.cfg_rdclksel1_i(tune[18:16]),
		.cfg_rd_off0_i(tune[23:20]),
		.cfg_rd_off1_i(tune[27:24]),

		.sts_datavalid_o(datavalid),
		.sts_burstdet_o(burstdet),
		.sts_clear_i(sts_clear),

		.ddr3_a(ddr3_a), .ddr3_ba(ddr3_ba),
		.ddr3_ras_n(ddr3_ras_n), .ddr3_cas_n(ddr3_cas_n),
		.ddr3_we_n(ddr3_we_n), .ddr3_cs_n(ddr3_cs_n),
		.ddr3_cke(ddr3_cke), .ddr3_odt(ddr3_odt),
		.ddr3_reset_n(ddr3_reset_n),
		.ddr3_dm(ddr3_dm), .ddr3_dq(ddr3_dq),
		.ddr3_dqs_p(ddr3_dqs_p), .ddr3_clk_p(ddr3_clk_p)
	);

	// -- STATUS, by position -------------------------------------
	reg [31:0] status;
	always @(*) begin
		status = 32'd0;
		status[31:16] = MAGIC;
		status[7]     = pll_locked_i;
		status[6]     = rd_timeout;
		status[5:4]   = burstdet;
		status[3:2]   = datavalid;
		status[1]     = phy_ready_i;
		status[0]     = init_done;
	end

	// -- register port -------------------------------------------
	always @(posedge clk_i) begin
		reg_ack_o <= 1'b0;
		sts_clear <= 1'b0;

		if (rst_i) begin
			tune <= TUNE_RESET;
			tune_next <= TUNE_RESET;
			tune_seq <= 4'd0;
			no_refresh <= 1'b0;
			wr_noread <= 1'b0;
			rdbuf_off <= 1'b0;
			st_gap <= 2'd0;
		end else if (tune_seq != 4'd0) begin
			// Paused: seven cycles before the change, seven after.
			tune_seq <= tune_seq - 4'd1;
			if (tune_seq == 4'd8) tune <= tune_next;
			if (tune_seq == 4'd1) reg_ack_o <= 1'b1;
		end else if (reg_cyc_i && reg_stb_i && !reg_ack_o
		             && reg_adr_i == 2'd1 && reg_we_i) begin
			reg_dat_o <= tune;
			tune_next <= reg_dat_i;
			tune_seq <= 4'd15;
		end else if (reg_cyc_i && reg_stb_i && !reg_ack_o) begin
			reg_ack_o <= 1'b1;
			case (reg_adr_i)
			2'd0: reg_dat_o <= status;
			2'd1: reg_dat_o <= tune;
			2'd2: begin
				reg_dat_o <= {26'd0, st_gap, rdbuf_off, wr_noread, 1'b0, no_refresh};
				if (reg_we_i) begin
					st_gap <= reg_dat_i[5:4];
					no_refresh <= reg_dat_i[0];
					sts_clear <= reg_dat_i[1];
					wr_noread <= reg_dat_i[2];
					rdbuf_off <= reg_dat_i[3];
				end
			end
			default: reg_dat_o <= 32'd0;
			endcase
		end
	end

endmodule

`default_nettype wire

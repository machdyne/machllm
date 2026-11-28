/*
 * machllm
 * Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
 *
 * mlm_soc: the machllm system, everything but the clocks and the pins.
 *
 *   zeitlos32ps (RV32IM, Zeitlos's core, unchanged) on one port:
 *
 *   0x0000_0000  RAM      BRAM_KB of block RAM: firmware, stack, buffers
 *   0x4000_0000  MAIN     512 MB DDR3 (mlm_ddr3: model image, KV cache)
 *   0x7000_0000  SYSCTL   0 LED, 4 ID "MLLM", 8 BUILD (parameter), 12 LINE,
 *                         16 RECONFIG: the key 0x5A52_4254 ("ZRBT") pulls
 *                         PROGRAMN low, the FPGA reloads (from flash 0:
 *                         machboot); reads 0x5A52_0001
 *   0x7600_0000  UI       the screen and the keyboard (UI = 1; rtl/mlm_ui.v)
 *   0x7000_0700  DDR3     Zeitlos's DDR3 registers (status, tune, ctrl)
 *   0x7100_0000  UART
 *   0x7200_0000  SD
 *   0x8000_0000  ENGINE   64 KB window (rtl/mlm_engine.v)
 *   0x8100_0000  DMA      main memory <-> RAM (rtl/mlm_dma.v)
 *
 * The RAM has a second port on the memory clock: the engine's copy-out
 * writes results into it, its packer reads vectors from it, and the
 * DMA unit moves blocks between it and main memory.
 *
 * The RAM answers in one cycle and takes a request every cycle; every
 * other target holds the port until it answers.
 *
 * Reserved for the text terminal and USB keyboard: about
 * 8K LUTs and 12 block RAMs; see docs/resources.md.
 */

`default_nettype none

module mlm_soc #(
	parameter BRAM_KB = 96,
	parameter SYS_MHZ = 48,               // clk (the CPU's)
	// Two clock domains: the CPU, its RAM, UART and SYSCTL on
	// clk; DDR3, engine and SD host on mclk at MEM_MHZ (the DDR3 runs at
	// twice that). The CPU's accesses to the mclk side cross through one
	// handshake (below). TWO_CLK = 0: everything on clk, mclk unused.
	parameter BAUD = 1000000,             // console (machboot's USB bridge runs at 1M)
	parameter TWO_CLK = 0,
	parameter MEM_MHZ = SYS_MHZ,
	parameter INIT_FILE = "",
	parameter [31:0] BUILD = 32'h0,
	parameter SIM_DDR3 = 0,               // simulation: controller + sim/mem DRAM model, no PHY
	parameter UI = 0                      // the screen and keyboard (rtl/mlm_ui.v)
) (
	input wire clk,
	input wire rst,
	input wire mclk,                      // TWO_CLK only
	input wire mrst,                      // TWO_CLK only: reset, synchronous to mclk

	// DDR3 clocking, from ddr3_clk
	input wire eclk,
	input wire ddrdel,
	input wire pause,
	input wire phy_rst,
	input wire phy_ready,
	input wire pll_locked,

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
	output wire ddr3_clk_p,

	output wire uart_tx,
	input wire uart_rx,

	output wire sd_sck,
	output wire sd_cmd_o,
	output wire sd_cmd_oe,
	input wire sd_cmd_i,
	input wire [3:0] sd_dat_i,

	output reg led,
	output reg reconfig,              // 1: pull PROGRAMN low (never out of reset)

	// UI = 1 only
	input wire pclk,
	input wire bclk,
	input wire clk12,
	output wire [1:0] tmds_clk,
	output wire [1:0] tmds_r,
	output wire [1:0] tmds_g,
	output wire [1:0] tmds_b,
	inout wire usb_dp,
	inout wire usb_dm
);

	wire fclk = TWO_CLK ? mclk : clk;
	wire frst = TWO_CLK ? mrst : rst;

	// ---------------------------------------------------------------
	// CPU

	wire m_req, m_we, m_instr;
	wire [31:0] m_adr, m_wdat;
	wire [3:0] m_sel;
	wire m_rdy, m_ack;
	wire [31:0] m_rdat;
	wire [31:0] eoi;
	wire trap;

	zeitlos32ps #(
		.PROGADDR_RESET(32'h0000_0000),
		.PROGADDR_IRQ(32'h0000_0010),
		.ENABLE_MUL(1), .ENABLE_DIV(1), .FAST_MUL(1),
		.ENABLE_COUNTERS(1), .ENABLE_MPU(0), .ENABLE_ROT(0)
	) cpu (
		.clk(clk), .rst(rst),
		.mpu_en(1'b0), .mpu_enforce(1'b0),
		.p_privreg(1'b0), .p_gate(1'b0), .p_own(1'b0),
		.m_priv(), .m_pc(), .i_priv(),
		.fv_ev(), .fv_adr(), .fv_pc(), .fv_why(),
		.m_req(m_req), .m_adr(m_adr), .m_we(m_we), .m_sel(m_sel), .m_wdat(m_wdat),
		.m_instr(m_instr), .m_rdy(m_rdy), .m_ack(m_ack), .m_rdat(m_rdat),
		.mtu_base(32'h0), .irq(32'h0), .eoi(eoi), .trap(trap)
	);

	// ---------------------------------------------------------------
	// decode

	wire t_ram   = (m_adr[31:28] == 4'h0);
	wire t_main  = (m_adr[31:29] == 3'b010);
	wire t_sys   = (m_adr[31:24] == 8'h70) && (m_adr[11:8] != 4'h7);
	wire t_dreg  = (m_adr[31:24] == 8'h70) && (m_adr[11:8] == 4'h7);
	wire t_uart  = (m_adr[31:24] == 8'h71);
	wire t_sd    = (m_adr[31:24] == 8'h72);
	wire t_eng   = (m_adr[31:28] == 4'h8);
	wire t_dma   = t_eng && m_adr[24];      // 0x8100_0000: the DMA unit (on the engine's path)
	wire t_vu    = (m_adr[31:24] == 8'h73);
	wire t_ui    = (m_adr[31:24] == 8'h76) && UI;

	// a slow access in progress: its target, address and data held
	reg pend;
	reg [2:0] pt;                   // 1 main 2 dreg 3 uart 4 sd 5 eng 6 sys 7 vu
	reg [31:0] p_adr, p_wdat;
	reg [3:0] p_sel;
	reg p_we;

	assign m_rdy = !pend;
	wire take = m_req && m_rdy;

	// ---------------------------------------------------------------
	// RAM

	localparam WORDS = BRAM_KB * 256;
	localparam AW = $clog2(WORDS);
	reg [31:0] ram [0:WORDS-1];
	initial if (INIT_FILE != "") $readmemh(INIT_FILE, ram);
	reg [31:0] ram_q;
	reg ram_ack;
	// the port: the CPU's, or the vector unit's while it runs (the CPU
	// is then stalled on its access to the unit and cannot take one)
	wire vu_own, vu_we;
	wire [AW-1:0] vu_a;
	wire [31:0] vu_wd;
	wire [AW-1:0] ram_a = vu_own ? vu_a : m_adr[AW+1:2];
	wire ram_en = vu_own || (take && t_ram);
	wire [3:0] ram_bw = vu_own ? {4{vu_we}} : (m_we ? m_sel : 4'd0);
	wire [31:0] ram_d = vu_own ? vu_wd : m_wdat;
	always @(posedge clk) begin
		ram_ack <= take && t_ram && !rst;
		if (ram_en) begin
			if (ram_bw[0]) ram[ram_a][7:0]   <= ram_d[7:0];
			if (ram_bw[1]) ram[ram_a][15:8]  <= ram_d[15:8];
			if (ram_bw[2]) ram[ram_a][23:16] <= ram_d[23:16];
			if (ram_bw[3]) ram[ram_a][31:24] <= ram_d[31:24];
			ram_q <= ram[ram_a];
		end
	end
	// the second port, on the memory clock: the engine's (copy-out
	// writes, packer reads) while it uses it, else the DMA unit's
	wire e_pb_we, e_pb_busy, d_pb_we;
	wire [AW-1:0] e_pb_wa, e_pb_ra, d_pb_a;
	wire [31:0] e_pb_wd, d_pb_wd;
	reg [31:0] pb_q0, pb_q;           // the read, then registered again (the 24-way
	                                  // mux over the block RAMs is most of a cycle)
	// (the choice registered: the engine's busy flag rises at least two
	// cycles before its first access, and the two never overlap)
	reg e_pb_own;
	always @(posedge fclk) e_pb_own <= e_pb_busy && !frst;
	wire [AW-1:0] pb_a = e_pb_we ? e_pb_wa : e_pb_own ? e_pb_ra : d_pb_a;
	wire pb_we = e_pb_we || d_pb_we;
	wire [31:0] pb_wd = e_pb_we ? e_pb_wd : d_pb_wd;
	always @(posedge fclk) begin
		if (pb_we) ram[pb_a] <= pb_wd;
		pb_q0 <= ram[pb_a];
		pb_q <= pb_q0;
	end

	// the vector unit (rtl/mlm_vu.v)
	wire vu_ack, rq_vu;
	wire [31:0] vu_rdat;
	mlm_vu #(.AW(AW)) vu (
		.clk(clk), .rst(rst),
		.bus_req(rq_vu), .bus_we(p_we), .bus_adr(p_adr[5:2]), .bus_wdat(p_wdat),
		.bus_ack(vu_ack), .bus_rdat(vu_rdat),
		.ram_own(vu_own), .ram_a(vu_a), .ram_we(vu_we), .ram_wd(vu_wd), .ram_q(ram_q)
	);

	// ---------------------------------------------------------------
	// main memory's last read line, on this side of the crossing (two
	// clocks): reads in it answer the next cycle instead of ~20 later.
	// Off until the firmware sets SYSCTL+12 bit 0 (after DDR3 training,
	// which needs every read to reach the DRAM); dropped on every
	// write to main memory, every DDR3 register access and every SD
	// block write. Nothing else writes main memory (the engine reads).
	wire s_wq_valid;                // an SD block write (below)
	reg ln_en, ln_v, ln_ack;
	reg [4:0] ln_quiet;             // no fills until 31 cycles after one: a read may race it
	reg [24:0] ln_tag;
	reg [127:0] ln_d;
	reg [31:0] ln_q;
	wire ln_hit = TWO_CLK && ln_v && t_main && !m_we && ln_tag == m_adr[28:4];
	always @(posedge clk) begin
		ln_ack <= take && ln_hit && !rst;
		if (take && ln_hit) ln_q <= ln_d[{m_adr[3:2], 5'd0} +: 32];
		if (rst || s_wq_valid || (take && t_dma && m_we)) ln_quiet <= 5'd31;
		else if (ln_quiet != 5'd0) ln_quiet <= ln_quiet - 5'd1;
		if (rst || !ln_en) ln_v <= 1'b0;
		else if ((take && (t_main || t_dreg) && !ln_hit && (m_we || t_dreg)) || s_wq_valid || ln_quiet != 5'd0 ||
		         (take && t_dma && m_we)) ln_v <= 1'b0;   // the DMA unit may write main memory
		else if (pend && pt == 3'd1 && !p_we && b_ack) begin
			ln_v <= b_line_v;
			ln_tag <= p_adr[28:4];
			ln_d <= b_line;
		end
	end


	// ---------------------------------------------------------------
	// slow targets

	wire main_ack, dreg_ack, uart_ack, sd_ack, eng_ack;
	wire [127:0] main_line;
	wire main_line_v;
	wire [31:0] main_rdat, dreg_rdat, uart_rdat, sd_rdat, eng_rdat;
	reg sys_ack;
	reg [31:0] sys_rdat;

	wire s_ack = (TWO_CLK ? (p_mem && b_ack) : q_ack) || (TWO_CLK && pt == 3'd4 && sd_ack) ||
	             (pt == 3'd3 && uart_ack) || (pt == 3'd6 && sys_ack) || (pt == 3'd7 && vu_ack);
	reg s_ackd;
	reg [31:0] s_rdat;

	always @(posedge clk) begin
		s_ackd <= 1'b0;
		if (rst) begin
			pend <= 1'b0;
			pt <= 3'd0;
		end else if (take && !t_ram && !t_ui && !ln_hit) begin
			pend <= 1'b1;
			pt <= t_main ? 3'd1 : t_dreg ? 3'd2 : t_uart ? 3'd3 : t_sd ? 3'd4 :
			      t_eng ? 3'd5 : t_vu ? 3'd7 : 3'd6;
			p_adr <= m_adr; p_wdat <= m_wdat; p_sel <= m_sel; p_we <= m_we;
		end else if (pend && s_ack) begin
			pend <= 1'b0;
			pt <= 3'd0;
			s_ackd <= 1'b1;
			s_rdat <= p_mem ? (TWO_CLK ? b_rdat : q_rdat) :
			          (pt == 3'd3) ? uart_rdat : (pt == 3'd7) ? vu_rdat : (pt == 3'd4) ? sd_rdat : sys_rdat;
		end
	end

	// the request lines stay up until the target answers, and drop the
	// cycle it does (each target ignores a request while its ack is high)
	wire rq_uart = pend && pt == 3'd3 && !uart_ack;
	assign rq_vu = pend && pt == 3'd7 && !vu_ack;

	// The memory-side targets (main, dreg, sd, eng) see q_*: the held
	// access itself with one clock, or its copy on mclk with two.
	wire q_pend;
	wire [2:0] q_pt;
	wire [31:0] q_adr, q_wdat;
	wire [3:0] q_sel;
	wire q_we;
	wire rq_main_b = q_pend && q_pt == 3'd1;    // the bridge's
	wire dma_own;                               // the DMA unit has the word port
	wire rq_main = rq_main_b && !dma_own;
	wire rq_dreg = q_pend && q_pt == 3'd2;
	// the SD host: on the memory side with one clock, on the CPU's side
	// with two (it is slow; its block writes cross below)
	wire rq_sd   = TWO_CLK ? (pend && pt == 3'd4 && !sd_ack) : (q_pend && q_pt == 3'd4 && !sd_ack);
	wire dma_ack;
	wire [31:0] dma_rdat;
	wire rq_eng  = q_pend && q_pt == 3'd5 && !q_adr[24] && !eng_ack;
	wire rq_dma  = q_pend && q_pt == 3'd5 && q_adr[24] && !dma_ack;
	wire q_ack = (q_pt == 3'd1 && main_ack && !dma_own) || (q_pt == 3'd2 && dreg_ack) ||
	             (q_pt == 3'd4 && sd_ack && !TWO_CLK) || (q_pt == 3'd5 && (eng_ack || dma_ack));
	wire [31:0] q_rdat = (q_pt == 3'd1) ? main_rdat : (q_pt == 3'd2) ? dreg_rdat :
	                     (q_pt == 3'd4) ? sd_rdat : q_adr[24] ? dma_rdat : eng_rdat;
	wire b_ack;                     // the mclk side answered (TWO_CLK)
	wire [31:0] b_rdat;
	wire [127:0] b_line;            // and the line of a main-memory read
	wire b_line_v;
	wire p_mem = (pt == 3'd1) || (pt == 3'd2) || (pt == 3'd5) || (pt == 3'd4 && !TWO_CLK);

	generate if (TWO_CLK) begin : g_bridge
		// clk -> mclk: a toggle per access, two flops of
		// synchronisation each way. p_* and pt are held (pend) until
		// the answer comes back, so the far side may sample them once
		// the toggle has crossed; b_rdat is likewise held when the
		// answer's toggle crosses back.
		reg s_tgl, s_iss;
		reg [2:0] s_sy;
		reg f_tgl, f_pend;
		reg [2:0] f_sy;
		reg [2:0] f_pt;
		reg [31:0] f_adr, f_wdat, f_rdat;
		reg [127:0] f_line;
		reg f_line_v;
		reg [3:0] f_sel;
		reg f_we;
		always @(posedge clk)
			if (rst) begin s_tgl <= 1'b0; s_iss <= 1'b0; s_sy <= 3'd0; end
			else begin
				s_sy <= {s_sy[1:0], f_tgl};
				if (pend && p_mem && !s_iss) begin s_tgl <= ~s_tgl; s_iss <= 1'b1; end
				else if (b_ack) s_iss <= 1'b0;
			end
		assign b_ack = s_sy[2] ^ s_sy[1];
		always @(posedge mclk)
			if (mrst) begin f_tgl <= 1'b0; f_pend <= 1'b0; f_sy <= 3'd0; f_pt <= 3'd0; end
			else begin
				f_sy <= {f_sy[1:0], s_tgl};
				if (f_sy[2] ^ f_sy[1]) begin
					f_pend <= 1'b1;
					f_pt <= pt; f_adr <= p_adr; f_wdat <= p_wdat; f_sel <= p_sel; f_we <= p_we;
				end else if (f_pend && q_ack) begin
					f_pend <= 1'b0;
					f_pt <= 3'd0;
					f_rdat <= q_rdat;
					// the read's whole line, if the controller's buffer
					// holds it (checked: the word read is in it)
					f_line <= main_line;
					f_line_v <= f_pt == 3'd1 && !f_we && main_line_v &&
					            main_line[{f_adr[3:2], 5'd0} +: 32] == main_rdat;
					f_tgl <= ~f_tgl;
				end
			end
		assign q_pend = f_pend; assign q_pt = f_pt; assign q_adr = f_adr;
		assign q_wdat = f_wdat; assign q_sel = f_sel; assign q_we = f_we;
		assign b_rdat = f_rdat;
		assign b_line = f_line; assign b_line_v = f_line_v;
	end else begin : g_one
		assign b_line = 128'd0; assign b_line_v = 1'b0;
		assign q_pend = pend; assign q_pt = pt; assign q_adr = p_adr;
		assign q_wdat = p_wdat; assign q_sel = p_sel; assign q_we = p_we;
		assign b_ack = 1'b0; assign b_rdat = 32'd0;
	end endgenerate

	// the screen and keyboard: answers the next cycle, as the RAM
	wire ui_ack;
	wire [31:0] ui_q;
	generate if (UI) begin : g_ui
		mlm_ui ui (
			.clk(clk), .rst(rst), .adr(m_adr[13:0]), .wdat(m_wdat), .we(m_we),
			.stb(take && t_ui), .rdat(ui_q), .ack(ui_ack),
			.pclk(pclk), .bclk(bclk),
			.tmds_clk(tmds_clk), .tmds_r(tmds_r), .tmds_g(tmds_g), .tmds_b(tmds_b),
			.clk12(clk12), .usb_dp(usb_dp), .usb_dm(usb_dm)
		);
	end else begin : g_noui
		assign ui_ack = 1'b0;
		assign ui_q = 32'd0;
		assign tmds_clk = 2'b00; assign tmds_r = 2'b00; assign tmds_g = 2'b00; assign tmds_b = 2'b00;
	end endgenerate

	assign m_ack = ram_ack || ln_ack || ui_ack || s_ackd;
	assign m_rdat = ram_ack ? ram_q : ln_ack ? ln_q : ui_ack ? ui_q : s_rdat;

	// ---------------------------------------------------------------
	// SYSCTL

	always @(posedge clk) begin
		sys_ack <= 1'b0;
		if (rst) begin led <= 1'b0; ln_en <= 1'b0; reconfig <= 1'b0; end
		else if (pend && pt == 3'd6 && !sys_ack) begin
			sys_ack <= 1'b1;
			case (p_adr[4:2])
			3'd0: begin if (p_we) led <= p_wdat[0]; sys_rdat <= {31'd0, led}; end
			3'd1: sys_rdat <= 32'h4d4c4c4d;
			3'd2: sys_rdat <= BUILD;
			3'd3: begin if (p_we) ln_en <= p_wdat[0]; sys_rdat <= {31'd0, ln_en}; end
			3'd4: begin if (p_we && p_wdat == 32'h5a524254) reconfig <= 1'b1; sys_rdat <= 32'h5a520001; end
			default: sys_rdat <= 32'd0;
			endcase
		end
	end

	// ---------------------------------------------------------------
	// DDR3

	wire sq_valid, sq_ready, sdv, wq_valid, wq_ready;
	wire [24:0] sq_blk, wq_blk;
	wire [8:0] sq_len;
	wire [127:0] sdd, wq_data;

	// the controller's ports, shared: the word port (the bridge, or the
	// DMA unit's byte stores while it owns it), the stream (the engine,
	// or a DMA read: never both at once), block writes (the SD loader
	// or the DMA unit: whoever has one waiting, kept until it is taken)
	wire d_sqv, d_rd_act, d_wqv, d_wb_want, d_wb_req;
	wire [24:0] d_sqblk, d_wqblk;
	wire [8:0] d_sqlen;
	wire [127:0] d_wqdata;
	wire [26:0] d_wb_adr;
	wire [3:0] d_wb_sel;
	wire [31:0] d_wb_dat;
	reg dma_own_r;
	assign dma_own = dma_own_r;
	always @(posedge fclk)
		if (frst) dma_own_r <= 1'b0;
		else if (!dma_own_r && d_wb_want && !rq_main_b) dma_own_r <= 1'b1;
		else if (dma_own_r && !d_wb_want) dma_own_r <= 1'b0;
	wire [26:0] c_adr = dma_own ? d_wb_adr : q_adr[28:2];
	wire [31:0] c_wdat = dma_own ? d_wb_dat : q_wdat;
	wire c_we = dma_own ? 1'b1 : q_we;
	wire [3:0] c_sel = dma_own ? d_wb_sel : q_sel;
	wire c_stb = dma_own ? d_wb_req : rq_main;
	wire c_sqv = sq_valid || d_sqv;
	wire [24:0] c_sqblk = d_sqv ? d_sqblk : sq_blk;
	wire [8:0] c_sqlen = d_sqv ? d_sqlen : sq_len;
	reg wq_dma;                     // the block-write port's owner
	always @(posedge fclk)
		if (frst) wq_dma <= 1'b0;
		else if (!(wq_dma ? d_wqv : wq_valid)) wq_dma <= !wq_valid && d_wqv;
	wire c_wqv = wq_dma ? d_wqv : wq_valid;
	wire [24:0] c_wqblk = wq_dma ? d_wqblk : wq_blk;
	wire [127:0] c_wqdata = wq_dma ? d_wqdata : wq_data;

	generate if (SIM_DDR3) begin : sim_ddr3
		// the controller against the checking DRAM model; the
		// registers answer as a trained PHY would (STATUS init done)
		wire cs_n, ras_n, cas_n, we_n, wren, rden, rvalid, init_done;
		wire [15:0] a, wmask;
		wire [2:0] ba;
		wire [127:0] wdata, rdata;
		mlm_ddr3_ctrl #(.ROW_BITS(15), .TRFC_NS(260), .INIT_DIV(100), .CLK_MHZ(MEM_MHZ)) ctrl (
			.clk_i(fclk), .rst_i(frst),
			.wb_adr_i(c_adr), .wb_dat_i(c_wdat), .wb_dat_o(main_rdat), .wb_we_i(c_we),
			.wb_sel_i(c_sel), .wb_stb_i(c_stb), .wb_cyc_i(c_stb), .wb_ack_o(main_ack),
			.wb_line_o(main_line), .wb_line_v_o(main_line_v),
			.sq_valid_i(c_sqv), .sq_ready_o(sq_ready), .sq_blk_i(c_sqblk), .sq_len_i(c_sqlen),
			.sd_valid_o(sdv), .sd_data_o(sdd),
			.wq_valid_i(c_wqv), .wq_ready_o(wq_ready), .wq_blk_i(c_wqblk), .wq_data_i(c_wqdata),
			.phy_ready_i(1'b1), .init_done_o(init_done), .rd_timeout_o(),
			.no_refresh_i(1'b0), .wr_noread_i(1'b0), .rdbuf_off_i(1'b0), .st_gap_i(2'd0),
			.phy_reset_n_o(), .phy_cke_o(), .phy_odt_o(),
			.phy_cs_n_o(cs_n), .phy_ras_n_o(ras_n), .phy_cas_n_o(cas_n), .phy_we_n_o(we_n),
			.phy_a_o(a), .phy_ba_o(ba),
			.phy_wren_o(wren), .phy_wdata_o(wdata), .phy_wmask_o(wmask),
			.phy_rden_o(rden), .phy_rvalid_i(rvalid), .phy_rdata_i(rdata)
		);
		mlm_dram_model #(.TCK_PS(500000 / MEM_MHZ)) dram (
			.clk(fclk), .cs_n(cs_n), .ras_n(ras_n), .cas_n(cas_n), .we_n(we_n),
			.a(a), .ba(ba), .wren(wren), .wdata(wdata), .wmask(wmask), .rden(rden),
			.rvalid(rvalid), .rdata(rdata)
		);
		reg dack;
		reg [31:0] drd;
		always @(posedge fclk) begin
			dack <= rq_dreg && !dack;
			drd <= (q_adr[3:2] == 2'd0) ? {16'hdd30, 14'd0, 1'b1, init_done} : 32'd0;
		end
		assign dreg_ack = dack;
		assign dreg_rdat = drd;
	end else begin : hw_ddr3
	mlm_ddr3 #(.ROW_BITS(15), .TRFC_NS(260), .WCOMBINE(0), .CLK_MHZ(MEM_MHZ)) ddr3 (
			.clk_i(fclk), .rst_i(frst),
			.eclk_i(eclk), .ddrdel_i(ddrdel), .pause_i(pause), .phy_rst_i(phy_rst),
			.phy_ready_i(phy_ready), .pll_locked_i(pll_locked),
			.wb_adr_i(c_adr), .wb_dat_i(c_wdat), .wb_dat_o(main_rdat),
			.wb_we_i(c_we), .wb_sel_i(c_sel), .wb_stb_i(c_stb), .wb_cyc_i(c_stb),
			.wb_ack_o(main_ack), .wb_line_o(main_line), .wb_line_v_o(main_line_v),
			.sq_valid_i(c_sqv), .sq_ready_o(sq_ready), .sq_blk_i(c_sqblk), .sq_len_i(c_sqlen),
			.sd_valid_o(sdv), .sd_data_o(sdd),
			.wq_valid_i(c_wqv), .wq_ready_o(wq_ready), .wq_blk_i(c_wqblk), .wq_data_i(c_wqdata),
			.reg_adr_i(q_adr[3:2]), .reg_dat_i(q_wdat), .reg_dat_o(dreg_rdat),
			.reg_we_i(q_we), .reg_stb_i(rq_dreg), .reg_cyc_i(rq_dreg), .reg_ack_o(dreg_ack),
			.ddr3_a(ddr3_a), .ddr3_ba(ddr3_ba), .ddr3_ras_n(ddr3_ras_n), .ddr3_cas_n(ddr3_cas_n),
			.ddr3_we_n(ddr3_we_n), .ddr3_cs_n(ddr3_cs_n), .ddr3_cke(ddr3_cke), .ddr3_odt(ddr3_odt),
			.ddr3_reset_n(ddr3_reset_n), .ddr3_dm(ddr3_dm), .ddr3_dq(ddr3_dq),
			.ddr3_dqs_p(ddr3_dqs_p), .ddr3_clk_p(ddr3_clk_p)
		);

	end endgenerate

	// ---------------------------------------------------------------
	// engine, SD, UART

	mlm_engine eng (
		.clk(fclk), .rst(frst),
		.bus_req_i(rq_eng), .bus_we_i(q_we), .bus_adr_i(q_adr[16:2]), .bus_wdat_i(q_wdat),
		.bus_sel_i(q_sel), .bus_ack(eng_ack), .bus_rdat(eng_rdat),
		.sq_valid(sq_valid), .sq_ready(sq_ready && !d_sqv), .sq_blk(sq_blk), .sq_len(sq_len),
		.sd_valid(sdv && !d_rd_act), .sd_data(sdd), .busy_o(eng_busy),
		.pb_we(e_pb_we), .pb_wa(e_pb_wa), .pb_wd(e_pb_wd), .pb_ra(e_pb_ra), .pb_q(pb_q),
		.pb_busy(e_pb_busy)
	);
	wire eng_busy;

	mlm_dma #(.RAM_AW(AW)) dma (
		.clk(fclk), .rst(frst),
		.bus_req(rq_dma), .bus_we(q_we), .bus_adr(q_adr[4:2]), .bus_wdat(q_wdat),
		.bus_ack(dma_ack), .bus_rdat(dma_rdat),
		.pb_we(d_pb_we), .pb_a(d_pb_a), .pb_wd(d_pb_wd), .pb_q(pb_q),
		.sq_valid(d_sqv), .sq_ready(sq_ready && d_sqv), .sq_blk(d_sqblk), .sq_len(d_sqlen),
		.sd_valid(sdv && d_rd_act), .sd_data(sdd), .rd_act(d_rd_act),
		.wq_valid(d_wqv), .wq_ready(wq_ready && wq_dma), .wq_blk(d_wqblk), .wq_data(d_wqdata),
		.wb_want(d_wb_want), .wb_own(dma_own), .wb_req(d_wb_req), .wb_adr(d_wb_adr),
		.wb_sel(d_wb_sel), .wb_dat(d_wb_dat), .wb_ack(main_ack && dma_own),
		.busy()
	);
`ifndef SYNTHESIS
	// the firmware's promises: the second RAM port and the stream are
	// never wanted by the engine and the DMA unit at once
	always @(posedge fclk) if (!frst) begin
		if (e_pb_busy && dma.op != 2'd0) begin
			$display("mlm_soc: the engine and the DMA unit both use the RAM's second port");
			$stop;
		end
		if ((d_sqv || d_rd_act) && (sq_valid || eng_busy)) begin
			$display("mlm_soc: a DMA read while the engine streams");
			$stop;
		end
	end
`endif

	// SD block writes into DDR3: straight through with one clock; with
	// two, a toggle each way per 16-byte block (the SD host holds the
	// block until it is taken, ~100 MB/s of headroom over the card)
	wire s_wq_ready;
	wire [24:0] s_wq_blk;
	wire [127:0] s_wq_data;
	generate if (TWO_CLK) begin : g_wq
		reg w_tgl, w_iss, w_rdy;
		reg [2:0] w_sy;
		reg a_tgl, a_v;
		reg [2:0] a_sy;
		reg [24:0] a_blk;
		reg [127:0] a_data;
		always @(posedge clk)
			if (rst) begin w_tgl <= 1'b0; w_iss <= 1'b0; w_rdy <= 1'b0; w_sy <= 3'd0; end
			else begin
				w_sy <= {w_sy[1:0], a_tgl};
				w_rdy <= 1'b0;
				if (s_wq_valid && !w_iss && !w_rdy) begin w_tgl <= ~w_tgl; w_iss <= 1'b1; end
				else if (w_iss && (w_sy[2] ^ w_sy[1])) begin w_iss <= 1'b0; w_rdy <= 1'b1; end
			end
		assign s_wq_ready = w_rdy;
		always @(posedge mclk)
			if (mrst) begin a_tgl <= 1'b0; a_v <= 1'b0; a_sy <= 3'd0; end
			else begin
				a_sy <= {a_sy[1:0], w_tgl};
				if (!a_v && (a_sy[2] ^ a_sy[1])) begin
					a_v <= 1'b1;
					a_blk <= s_wq_blk;
					a_data <= s_wq_data;
				end else if (a_v && wq_ready && !wq_dma) begin
					a_v <= 1'b0;
					a_tgl <= ~a_tgl;
				end
			end
		assign wq_valid = a_v;
		assign wq_blk = a_blk;
		assign wq_data = a_data;
	end else begin : g_wq1
		assign wq_valid = s_wq_valid;
		assign wq_blk = s_wq_blk;
		assign wq_data = s_wq_data;
		assign s_wq_ready = wq_ready && !wq_dma;
	end endgenerate

	mlm_sd sd (
		.clk(TWO_CLK ? clk : fclk), .rst(TWO_CLK ? rst : frst),
		.bus_req(rq_sd), .bus_we(TWO_CLK ? p_we : q_we), .bus_adr(TWO_CLK ? p_adr[5:2] : q_adr[5:2]),
		.bus_wdat(TWO_CLK ? p_wdat : q_wdat),
		.bus_ack(sd_ack), .bus_rdat(sd_rdat),
		.wq_valid(s_wq_valid), .wq_ready(s_wq_ready), .wq_blk(s_wq_blk), .wq_data(s_wq_data),
		.sd_sck(sd_sck), .sd_cmd_o(sd_cmd_o), .sd_cmd_oe(sd_cmd_oe), .sd_cmd_i(sd_cmd_i),
		.sd_dat_i(sd_dat_i)
	);

	mlm_uart #(.DIV_RESET((SYS_MHZ * 1000000 + BAUD / 2) / BAUD)) uart (
		.clk(clk), .rst(rst),
		.bus_req(rq_uart), .bus_we(p_we), .bus_adr(p_adr[3:2]), .bus_wdat(p_wdat),
		.bus_ack(uart_ack), .bus_rdat(uart_rdat),
		.tx(uart_tx), .rx(uart_rx)
	);

endmodule

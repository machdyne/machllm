/*
 * machllm
 * Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
 *
 * mlm_engine: the matrix-vector unit.
 *
 * For each of ROWS rows of a quantised matrix in main memory it computes
 *
 *   acc = sum_g m_g * sum_k q_gk * x[32g + k]          (exact, 64 bit)
 *
 * with x an int16 vector the CPU has written into the engine, and leaves
 * acc in an output memory for the CPU to read. The row format is the
 * .mlm format (fw/common/mlm.h, docs/format.md); the specification is
 * eng_dot_spec() in ref/mlm_run.c, and the testbench (sim/engine)
 * checks every call against it.
 *
 * -- CPU window (byte offsets) ------------------------------------
 *   0x0000-0x1fff  X     write only, int16 x[0..4095], 16/32-bit writes
 *   0x4000         ADDR    byte address of row 0 (16-byte aligned)
 *   0x4004         STRIDE  bytes between rows (multiple of 16)
 *   0x4008         ROWS    1..4096
 *   0x400c         GROUPS  groups of 32 per row, 1..128
 *   0x4010         QBITS   4 or 8
 *   0x4014         CTRL    write: start; read: [0] busy
 *   0x4018         CYCLES  cycles of the last run (read only)
 *   0x401c         WAITS   of those, cycles with no data to work on
 *   0x4020/0x4024  MAXABS0 largest |acc| written to bank 0 by its last run
 *   0x4028         SHIFT   for OUT32, 0..63
 *   0x402c         OBANK   the bank the next run writes (0 or 1)
 *   0x4030/0x4034  MAXABS1 the same for bank 1
 *   0x8000-0xffff  OUT bank 0, read only, row i at 0x8000 + 8i (lo, hi)
 *   0x18000-0x1ffff OUT bank 1
 *   0x10000-0x13fff OUT32 bank 0, read only, row i at 0x10000 + 4i:
 *                  (acc + 2^(SHIFT-1)) >> SHIFT, low 32 bits (the
 *                  firmware picks SHIFT from MAXABS so nothing is lost)
 *   0x14000-0x17fff OUT32 bank 1
 *   0x4038         GLU_N   rows for the GLU unit
 *   0x403c         GLU_P   [5:0] sg, [13:8] su, [23:16] sq (signed)
 *   0x4040         GLU_GO  write: start the GLU unit (CTRL bit 0 is busy)
 *   0x4044         GLU_RES [7:0] psh, [15:8] xsh (signed)
 *   0x4048         CO_GO   copy-out: n (1..2048) rows' OUT32 values into
 *                  the CPU's RAM from word CO_DST on (CTRL bit 1 busy)
 *   0x404c         CO_ROW  bank << 12 | first row
 *   0x4050         PK_GO   the packer: [12:0] n, [22:16] blocks of 32
 *                  (padn / 32): int32 v[0..n) from the RAM at word
 *                  PK_SRC, MX-packed into X (eng_mx_pack, eng_spec.h;
 *                  zero from n on); CTRL bit 2 busy
 *   0x4054         CO_DST  RAM word address of the copy-out's first row
 *   0x4058         PK_SRC  RAM word address of v[0]
 *   0x4060         PK_S    read: the packing's s (the input's exponent is
 *                  s - 15)
 *   X reads (0x0000-0x1fff) return the input vector, when idle (tests)
 *
 * The GLU unit (glu_spec() in fw/common/eng_spec.h) turns bank 0 (gate)
 * and bank 1 (up) into SiLU(gate) x up, quantised to int16, in X: three
 * passes over the rows, about 17 cycles a row, using bank 0 as its
 * scratch.
 *
 * Two output banks let the CPU work on one run's results while the
 * engine computes the next run into the other.
 *
 * -- memory stream -------------------------------------------------
 * A request is a 16-byte block address and a block count; the memory
 * answers with the blocks in order, one 128-bit block per sd_valid, and
 * requests are answered in the order made. The engine only asks for as
 * many blocks as its FIFO has room for, so sd_valid needs no ready.
 *
 * Throughput: one 64-bit word per cycle, which is what the memory gives
 * at its best (a block every second cycle).
 */

`default_nettype none

module mlm_engine #(
	parameter ADDR_BITS = 29,         // byte address bits of main memory
	parameter FIFO_LOG = 5,           // data FIFO: 2^FIFO_LOG blocks
	parameter RAM_AW = 15             // the CPU's RAM: word address bits
) (
	input wire clk,
	input wire rst,

	// CPU
	input wire bus_req_i,
	input wire bus_we_i,
	input wire [16:2] bus_adr_i,
	input wire [31:0] bus_wdat_i,
	input wire [3:0] bus_sel_i,
	output reg bus_ack,
	output reg [31:0] bus_rdat,

	// memory stream
	output reg sq_valid,
	input wire sq_ready,
	output reg [ADDR_BITS-5:0] sq_blk,
	output reg [8:0] sq_len,          // blocks, 1..256
	input wire sd_valid,
	input wire [127:0] sd_data,

	output wire busy_o,

	// the CPU's RAM, its second port (rtl/mlm_soc.v; word addresses): a
	// copy-out writes OUT32 values of consecutive rows into it, the
	// packer reads the int32 vector it MX-packs into X from it (the
	// data the third cycle after pb_ra)
	output reg pb_we,
	output reg [RAM_AW-1:0] pb_wa,
	output reg [31:0] pb_wd,
	output reg [RAM_AW-1:0] pb_ra,
	input wire [31:0] pb_q,
	output wire pb_busy              // a copy-out or the packer is using the port
);

	// The bus, registered on the way in (the decode below is otherwise a
	// long path from the clock crossing). The master drops its request
	// while bus_ack is high, so the copy falls the cycle after it.
	reg bus_req, bus_we;
	reg [16:2] bus_adr;
	reg [31:0] bus_wdat;
	reg [3:0] bus_sel;
	always @(posedge clk) begin
		bus_req <= bus_req_i && !rst;
		bus_we <= bus_we_i;
		bus_adr <= bus_adr_i;
		bus_wdat <= bus_wdat_i;
		bus_sel <= bus_sel_i;
	end


	// ---------------------------------------------------------------
	// registers

	reg [ADDR_BITS-1:0] r_addr;
	reg [ADDR_BITS-1:0] r_stride;
	reg [12:0] r_rows;
	reg [7:0] r_groups;
	reg r_q1;
	reg [5:0] r_shift;
	reg busy;
	reg start;
	reg [31:0] r_cycles, r_waits;
	assign busy_o = busy;

	// ---------------------------------------------------------------
	// X memory, MX-packed (eng_mx_pack in fw/common/eng_spec.h): 64 rows
	// of 64 int8 (512 bits, 16 banks of 32 bits; row r holds elements
	// 64r .. 64r + 63, bank b elements 64r + 4b .. + 3, lowest first),
	// and per row the two blocks' exponent offsets (4 bits each).
	//   CPU: 0x0000 + 4w   word w of the values (w = 16 row + bank)
	//        0x1000 + 4b   block b's exponent offset (bits 3:0)

	wire x_bus = bus_req && bus_we && (bus_adr[16:13] == 4'b0000);
	wire x_we_bus = x_bus && !bus_adr[12];
	wire xd_we_bus = x_bus && bus_adr[12];
	reg glu_xwe, glu_dwe;
	reg [3:0] glu_xbank;
	reg [5:0] glu_xentry;
	reg [31:0] glu_xdata;
	reg [6:0] glu_dblk;
	reg [3:0] glu_ddata;
	// the packer's writes (PK_GO): X word w (bank w[3:0], entry w[9:4])
	// and block b's offset
	reg ci_xwe, ci_dwe;
	reg [9:0] ci_xw;
	reg [6:0] ci_db;
	reg [31:0] pk_xd;
	reg [3:0] pk_dd;
	wire x_we = x_we_bus || glu_xwe || ci_xwe;
	wire [3:0] x_wbank = ci_xwe ? ci_xw[3:0] : glu_xwe ? glu_xbank : bus_adr[5:2];
	wire [5:0] x_wentry = ci_xwe ? ci_xw[9:4] : glu_xwe ? glu_xentry : bus_adr[11:6];
	wire [3:0] x_wsel = (ci_xwe || glu_xwe) ? 4'hf : bus_sel;
	wire [31:0] x_wdat = ci_xwe ? pk_xd : glu_xwe ? glu_xdata : bus_wdat;
	wire xd_we = xd_we_bus || glu_dwe || ci_dwe;
	wire [6:0] xd_wblk = ci_dwe ? ci_db : glu_dwe ? glu_dblk : bus_adr[8:2];
	wire [3:0] xd_wdat = ci_dwe ? pk_dd : glu_dwe ? glu_ddata : bus_wdat[3:0];
	reg [5:0] x_raddr;
	reg [5:0] xb_raddr;              // test read-back of X
	reg xb_rd;
	wire [5:0] x_ra = xb_rd ? xb_raddr : x_raddr;
	wire [511:0] x_rdata;

	genvar gb;
	generate for (gb = 0; gb < 16; gb = gb + 1) begin : xb
		reg [31:0] mem [0:63];
		reg [31:0] q;
		always @(posedge clk) begin
			if (x_we && x_wbank == gb) begin
				if (x_wsel[0]) mem[x_wentry][7:0] <= x_wdat[7:0];
				if (x_wsel[1]) mem[x_wentry][15:8] <= x_wdat[15:8];
				if (x_wsel[2]) mem[x_wentry][23:16] <= x_wdat[23:16];
				if (x_wsel[3]) mem[x_wentry][31:24] <= x_wdat[31:24];
			end
			q <= mem[x_ra];
		end
		assign x_rdata[32*gb +: 32] = q;
	end endgenerate
	// the exponent offsets: even and odd blocks of a row, read with it
	reg [3:0] xde [0:63];
	reg [3:0] xdo [0:63];
	reg [3:0] xde_q, xdo_q;
	always @(posedge clk) begin
		if (xd_we && !xd_wblk[0]) xde[xd_wblk[6:1]] <= xd_wdat;
		if (xd_we && xd_wblk[0]) xdo[xd_wblk[6:1]] <= xd_wdat;
		xde_q <= xde[x_ra];
		xdo_q <= xdo[x_ra];
	end

	// ---------------------------------------------------------------
	// OUT memory: 4096 x 64

	reg [63:0] out_mem [0:8191];
	reg out_we;
	reg [12:0] out_waddr;
	reg [63:0] out_wdata;
	reg [63:0] out_q;
	reg r_obank, run_bank;
	// CPU read address: OUT bank 0 0x08000, bank 1 0x18000 (8 bytes a
	// row); OUT32 bank 0 0x10000, bank 1 0x14000 (4 bytes a row)
	wire o32 = (bus_adr[16:15] == 2'b10);
	wire [12:0] out_raddr = o32 ? {bus_adr[14], bus_adr[13:2]} : {bus_adr[16], bus_adr[14:3]};
	reg glu_on;                      // the GLU unit owns the OUT RAM
	reg [12:0] glu_raddr, glu_waddr;
	reg glu_owe;
	reg [63:0] glu_wdata;
	wire o_we = out_we || glu_owe;
	wire [12:0] o_wa = glu_owe ? glu_waddr : out_waddr;
	wire [63:0] o_wd = glu_owe ? glu_wdata : out_wdata;
	// copy-out: CO_ROW (bank, first row), then CO_GO = n (1..2048) copies
	// rows' OUT32 values (SHIFT as set) into the mirror, one a cycle
	reg co_on;
	reg [12:0] co_ra;
	reg [10:0] co_left, co_k;
	reg [12:0] co_row;
	reg co_v1, co_v1b, co_v2;
	reg [10:0] co_k1, co_k1b, co_k2;
	reg [63:0] co_q, co_sum;
	wire [12:0] o_ra = co_on ? co_ra : glu_on ? glu_raddr : out_raddr;
	always @(posedge clk) begin
		if (o_we) out_mem[o_wa] <= o_wd;
		out_q <= out_mem[o_ra];
	end

	// ---------------------------------------------------------------
	// bus

	reg rd_outb;
	reg rd_out, rd_out_hi, rd_o32, rd_o32a, rd_o32b, rd_x1, rd_x2, rd_x2b, rd_x3;
	reg rd_rg;
	reg [5:0] rg_sel;               // {register page, GLU page, word}
	reg [31:0] rg_mux;
	always @(*) begin
		rg_mux = 32'd0;
		if (rg_sel[5] && rg_sel[4]) begin
			if (rg_sel[3:0] == 4'd1) rg_mux = {16'd0, glu_xsh, glu_psh};
			if (rg_sel[3:0] == 4'd8) rg_mux = {26'd0, pk_s};
		end else if (rg_sel[5]) case (rg_sel[3:0])
			4'd5: rg_mux = {29'd0, pk_busy,                                       // bit 2: the packer
			                co_go | co_on | co_v1 | co_v1b | co_v2 | pb_we,    // bit 1: copy-out
			                busy | glu_busy | glu_go};
			4'd6: rg_mux = r_cycles;
			4'd7: rg_mux = r_waits;
			4'd8: rg_mux = max0[31:0];
			4'd9: rg_mux = max0[63:32];
			4'd10: rg_mux = {26'd0, r_shift};
			4'd11: rg_mux = {31'd0, r_obank};
			4'd12: rg_mux = max1[31:0];
			4'd13: rg_mux = max1[63:32];
			default: ;
		endcase
	end
	reg [511:0] xb_hold;             // the X RAM's output, registered before the bank mux
	// ---------------------------------------------------------------
	// the packer: eng_mx_pack (fw/common/eng_spec.h) in hardware, one
	// element a cycle in each of two passes over v in the RAM.
	//   pass 1: per block of 32, mx = max |v[i]| (i < n), s_b =
	//           max(bitlen(mx) - 7, 0); s = max s_b
	//   pass 2: per block, sh = max(s - 15, s_b), offset s - sh; each
	//           value rshr32(v[i], sh) clamped to +-127 (0 from n on),
	//           four to an X word, the offset after the block's last
	// Reads: pb_ra at a cycle, its data the third cycle after (the RAM
	// registers its output, the SOC again); the tags travel alongside.
	reg ci_go;
	reg [12:0] pk_n;
	reg [6:0] pk_nb;
	reg [RAM_AW-1:0] pk_src, co_dst;
	reg [2:0] pk_ph;                  // 0 idle, 1 pass 1, 2 drain, 3 pass 2, 4 drain
	reg [11:0] pk_i;
	reg [3:0] pk_dr;
	reg [5:0] pk_s;
	reg [4:0] pk_bx [0:127];          // s_b per block
	reg [4:0] pk_sh;                  // pass 2: this block's shift and offset
	reg [3:0] pk_off;
	// tags: valid, in range, element of block, block
	reg t1_v, t1_in, t2_v, t2_in, t3_v, t3_in, a_v, a_in;
	reg [4:0] t1_k, t2_k, t3_k, a_k;
	reg [6:0] t1_b, t2_b, t3_b, a_b;
	reg [4:0] t1_sh, t2_sh, t3_sh;
	reg [3:0] t1_off, t2_off, t3_off, a_off;
	reg p2_t1, p2_t2, p2_t3;          // the tag is pass 2's
	reg [31:0] mx, mxb;               // pass 1: this block's max so far, a block's
	reg bd, bd2;
	reg [6:0] bd_b, bd2_b;
	reg [4:0] pk_sbr;
	wire [11:0] pk_last = {pk_nb, 5'd0} - 12'd1;
	// bit length as a balanced tree (a priority chain is a long path)
	function [2:0] pk_bl4;
		input [3:0] v;
		pk_bl4 = v[3] ? 3'd4 : v[2] ? 3'd3 : v[1] ? 3'd2 : v[0] ? 3'd1 : 3'd0;
	endfunction
	function [3:0] pk_bl8;
		input [7:0] v;
		pk_bl8 = (v[7:4] != 4'd0) ? {1'b0, pk_bl4(v[7:4])} + 4'd4 : {1'b0, pk_bl4(v[3:0])};
	endfunction
	function [4:0] pk_bl16;
		input [15:0] v;
		pk_bl16 = (v[15:8] != 8'd0) ? {1'b0, pk_bl8(v[15:8])} + 5'd8 : {1'b0, pk_bl8(v[7:0])};
	endfunction
	function [5:0] pk_bl32;
		input [31:0] v;
		pk_bl32 = (v[31:16] != 16'd0) ? {1'b0, pk_bl16(v[31:16])} + 6'd16 : {1'b0, pk_bl16(v[15:0])};
	endfunction
	// pass 2: s - 15 against s_b of the next block (pk_bn), read from the
	// table a block ahead into a register
	reg [6:0] pk_bn;
	reg [4:0] pk_bxn;
	reg signed [6:0] pk_s15;
	always @(posedge clk) begin
		pk_bxn <= pk_bx[pk_bn];
		pk_s15 <= $signed({1'b0, pk_s}) - 7'sd15;
	end
	function [4:0] pk_shof;
		input [4:0] sb; input signed [6:0] s15;
		pk_shof = (s15 > $signed({2'b00, sb})) ? s15[4:0] : sb;
	endfunction
	wire [4:0] pk_shn = pk_shof(pk_bxn, pk_s15);
	always @(posedge clk) begin
		t1_v <= 1'b0;
		if (rst) begin
			pk_ph <= 3'd0;
		end else case (pk_ph)
		3'd0: if (ci_go) begin
			pk_ph <= (pk_nb == 7'd0) ? 3'd0 : 3'd1;
			pk_i <= 12'd0;
			pk_s <= 6'd0;
		end
		3'd1, 3'd3: begin
			pb_ra <= pk_src + {{(RAM_AW-12){1'b0}}, pk_i};
			t1_v <= 1'b1;
			t1_in <= ({1'b0, pk_i} < pk_n);
			t1_k <= pk_i[4:0];
			t1_b <= pk_i[11:5];
			t1_sh <= pk_sh;
			t1_off <= pk_off;
			p2_t1 <= (pk_ph == 3'd3);
			if (pk_ph == 3'd3 && pk_i[4:0] == 5'd31) begin
				// the next block's shift (its s_b is final: pass 1 is done)
				pk_sh <= pk_shn;
				pk_off <= pk_s[3:0] - pk_shn[3:0];
				pk_bn <= pk_bn + 7'd1;
			end
			if (pk_i == pk_last) begin
				pk_ph <= pk_ph + 3'd1;
				pk_dr <= 4'd8;
				pk_bn <= 7'd0;             // pass 2's first block
			end else pk_i <= pk_i + 12'd1;
		end
		3'd2, 3'd4: if (pk_dr != 4'd0) pk_dr <= pk_dr - 4'd1;
		else if (pk_ph == 3'd2) begin
			pk_ph <= 3'd3;
			pk_i <= 12'd0;
			pk_sh <= pk_shn;           // block 0's (pk_bn has been 0 all the drain)
			pk_off <= pk_s[3:0] - pk_shn[3:0];
			pk_bn <= 7'd1;
		end else pk_ph <= 3'd0;
		default: pk_ph <= 3'd0;
		endcase
		// pass 1's end of a block: s_b into the table, s the largest
		if (bd2) begin
			pk_bx[bd2_b] <= pk_sbr;
			if ({1'b0, pk_sbr} > pk_s) pk_s <= {1'b0, pk_sbr};
		end
	end
	// the data: pb_q is t3's
	reg p2_a;
	reg [31:0] a_ab;                  // pass 1: |v|
	reg signed [31:0] a_r1;           // pass 2: v >>> (sh - 1), or v for sh 0
	reg a_sh0;
	always @(posedge clk) begin
		t2_v <= t1_v && !rst; t2_in <= t1_in; t2_k <= t1_k; t2_b <= t1_b;
		t2_sh <= t1_sh; t2_off <= t1_off; p2_t2 <= p2_t1;
		t3_v <= t2_v && !rst; t3_in <= t2_in; t3_k <= t2_k; t3_b <= t2_b;
		t3_sh <= t2_sh; t3_off <= t2_off; p2_t3 <= p2_t2;
		a_v <= t3_v && !rst; a_in <= t3_in; a_k <= t3_k; a_b <= t3_b; a_off <= t3_off; p2_a <= p2_t3;
		begin : pk_a
			reg signed [31:0] x;
			x = t3_in ? $signed(pb_q) : 32'sd0;
			a_ab <= x[31] ? (~x + 32'd1) : x;
			a_sh0 <= (t3_sh == 5'd0);
			a_r1 <= (t3_sh == 5'd0) ? x : (x >>> (t3_sh - 5'd1));
		end
	end
	// pass 1: the block's max, then s_b
	wire [31:0] mx_new = (a_k == 5'd0 || a_ab > mx) ? a_ab : mx;
	always @(posedge clk) begin
		bd <= 1'b0;
		bd2 <= 1'b0;
		if (a_v && !p2_a) begin
			mx <= mx_new;
			if (a_k == 5'd31) begin bd <= !rst; bd_b <= a_b; mxb <= mx_new; end
		end
		if (bd) begin
			bd2 <= !rst;
			bd2_b <= bd_b;
			pk_sbr <= (pk_bl32(mxb) > 6'd7) ? pk_bl32(mxb) - 6'd7 : 5'd0;
		end
	end
	// pass 2: round, then clamp, then four bytes to a word
	reg r_v, b_v;
	reg [4:0] r_k, b_k;
	reg [6:0] r_b, b_b;
	reg [3:0] r_off, b_off;
	reg signed [31:0] r_r;
	reg [7:0] b_byte;
	reg [23:0] wbuf;
	always @(posedge clk) begin
		r_v <= a_v && p2_a && !rst;
		r_k <= a_k; r_b <= a_b; r_off <= a_off;
		r_r <= a_sh0 ? a_r1 : ((a_r1 + 32'sd1) >>> 1);
		b_v <= r_v && !rst;
		b_k <= r_k; b_b <= r_b; b_off <= r_off;
		b_byte <= (r_r > 32'sd127) ? 8'd127 : (r_r < -32'sd127) ? 8'h81 : r_r[7:0];
		ci_xwe <= 1'b0;
		ci_dwe <= 1'b0;
		if (b_v) begin
			case (b_k[1:0])
			2'd0: wbuf[7:0] <= b_byte;
			2'd1: wbuf[15:8] <= b_byte;
			2'd2: wbuf[23:16] <= b_byte;
			default: begin
				ci_xwe <= 1'b1;
				ci_xw <= {b_b, b_k[4:2]};
				pk_xd <= {b_byte, wbuf};
			end
			endcase
			if (b_k == 5'd31) begin
				ci_dwe <= 1'b1;
				ci_db <= b_b;
				pk_dd <= b_off;
			end
		end
	end
	wire pk_busy = (pk_ph != 3'd0) || ci_go || t1_v || t2_v || t3_v || a_v || r_v || b_v || ci_xwe || ci_dwe;

	reg co_go;
	reg [11:0] co_n;
	assign pb_busy = pk_busy || co_go || co_on || co_v1 || co_v1b || co_v2 || pb_we;
	// one row a cycle: address, then out_q (+ rounding), then the shift
	wire [63:0] co_rnd = (r_shift == 6'd0) ? 64'd0 : (64'd1 << (r_shift - 6'd1));
	always @(posedge clk) begin
		pb_we <= 1'b0;
		co_v1 <= 1'b0;
		co_v1b <= 1'b0;
		co_v2 <= 1'b0;
		if (rst) begin
			co_on <= 1'b0;
		end else begin
			if (co_go && !co_on) begin
				co_on <= 1'b1;
				co_ra <= co_row;
				co_k <= 11'd0;
				co_left <= co_n[10:0] - 11'd1;
			end else if (co_on) begin
				co_v1 <= 1'b1;
				co_k1 <= co_k;
				co_ra <= co_ra + 13'd1;
				co_k <= co_k + 11'd1;
				co_left <= co_left - 11'd1;
				if (co_left == 11'd0) co_on <= 1'b0;
			end
			// stage 1: out_q is the row's acc (the RAM's output: registered
			// again before the adder, a long path otherwise)
			if (co_v1) begin
				co_v1b <= 1'b1;
				co_k1b <= co_k1;
				co_q <= out_q;
			end
			if (co_v1b) begin
				co_v2 <= 1'b1;
				co_k2 <= co_k1b;
				co_sum <= co_q + co_rnd;
			end
			if (co_v2) begin
				pb_we <= 1'b1;
				pb_wa <= co_dst + {{(RAM_AW-11){1'b0}}, co_k2};
				pb_wd <= ($signed(co_sum) >>> r_shift);
			end
		end
	end
	reg [63:0] o32_q;
	reg [31:0] xb_word;
	reg [3:0] xb_bank;
	reg xb_isd, xb_dodd;             // reading an exponent offset (0x1000 + 4b)
	reg [3:0] xb_dq;
	reg [12:0] glu_n;
	reg [5:0] glu_sg, glu_su;
	reg signed [7:0] glu_sq;
	reg glu_go;
	reg [7:0] glu_psh;
	reg signed [7:0] glu_xsh;
	reg glu_busy;
	reg [63:0] maxabs, max0, max1;
	// OUT32: the shifted value, a cycle after the RAM
	wire [63:0] rnd = (r_shift == 6'd0) ? 64'd0 : (64'd1 << (r_shift - 6'd1));
	reg [63:0] o32_sum;
	always @(posedge clk) begin
		bus_ack <= 1'b0;
		start <= 1'b0;
		co_go <= 1'b0;
		ci_go <= 1'b0;
		rd_out <= 1'b0;
		rd_outb <= 1'b0;
		rd_o32 <= 1'b0;
		rd_o32a <= 1'b0;
		rd_o32b <= 1'b0;
		glu_go <= 1'b0;
		rd_x1 <= 1'b0;
		rd_x2 <= 1'b0;
		rd_x2b <= 1'b0;
		rd_x3 <= 1'b0;
		rd_rg <= 1'b0;
		if (rd_x2) xb_rd <= 1'b0;
		if (rst) begin
			xb_rd <= 1'b0;
			r_addr <= 0; r_stride <= 0; r_rows <= 0; r_groups <= 0; r_q1 <= 1'b0;
			r_shift <= 6'd0;
			r_obank <= 1'b0;
		end else if (rd_out) begin
			o32_q <= out_q;
			rd_outb <= 1'b1;
		end else if (rd_outb) begin
			bus_ack <= 1'b1;
			bus_rdat <= rd_out_hi ? o32_q[63:32] : o32_q[31:0];
		end else if (rd_o32) begin
			o32_q <= out_q;              // the RAM's output, registered
			rd_o32a <= 1'b1;
		end else if (rd_o32a) begin
			o32_sum <= o32_q + rnd;
			rd_o32b <= 1'b1;
		end else if (rd_o32b) begin
			bus_ack <= 1'b1;
			bus_rdat <= $signed(o32_sum) >>> r_shift;
		end else if (rd_x1) begin
			rd_x2 <= 1'b1;              // the RAM reads xb_raddr
		end else if (rd_x2) begin
			rd_x2b <= 1'b1;
			xb_hold <= x_rdata;
			xb_dq <= xb_dodd ? xdo_q : xde_q;
		end else if (rd_x2b) begin
			rd_x3 <= 1'b1;
			xb_word <= xb_isd ? {28'd0, xb_dq} : xb_hold[32*xb_bank +: 32];
		end else if (rd_x3) begin
			bus_ack <= 1'b1;
			bus_rdat <= xb_word;
		end else if (rd_rg) begin
			bus_ack <= 1'b1;            // a register read, its select registered last cycle
			bus_rdat <= rg_mux;
		end else if (bus_req && !bus_ack && !rd_o32 && !rd_o32a && !rd_o32b && !rd_out && !rd_outb && !rd_x1 && !rd_x2 && !rd_x2b && !rd_x3 && !rd_rg) begin
			if (!bus_we && bus_adr[16:13] == 4'b0000) begin
				xb_rd <= 1'b1;
				xb_isd <= bus_adr[12];
				xb_dodd <= bus_adr[2];
				xb_raddr <= bus_adr[12] ? bus_adr[8:3] : bus_adr[11:6];
				xb_bank <= bus_adr[5:2];
				rd_x1 <= 1'b1;
			end else if (!bus_we && o32) begin
				rd_o32 <= 1'b1;          // RAM, then the shift
			end else if (!bus_we && bus_adr[15]) begin
				rd_out <= 1'b1;          // one cycle for the RAM
				rd_out_hi <= bus_adr[2];
			end else if (!bus_we) begin
				// registers: the select now, the value next cycle (one
				// 20-way mux from the decode otherwise)
				rd_rg <= 1'b1;
				rg_sel <= {bus_adr[16:14] == 3'b001, bus_adr[6], bus_adr[5:2]};
			end else begin
				bus_ack <= 1'b1;
				bus_rdat <= 32'd0;
				if (bus_adr[16:14] == 3'b001 && bus_adr[6]) begin
					// 0x4040 GLU_GO, 0x4044 GLU_RES, 0x4048 CO_GO, 0x404c CO_ROW
					// (0x4040-0x404c: bits 5:4 are 00, the registers
					// above 0x4050 must not alias them)
					if (bus_adr[5:4] == 2'b00) case (bus_adr[3:2])
					2'd0: if (bus_we) glu_go <= 1'b1;
					2'd2: if (bus_we) co_go <= 1'b1;
					2'd3: if (bus_we) co_row <= bus_wdat[12:0];
					default: ;
					endcase
					if (bus_we && bus_adr[5:2] == 4'd2) co_n <= bus_wdat[11:0];
					// 0x4050 PK_GO: n (bits 12:0), blocks (22:16);
					// 0x4054 CO_DST, 0x4058 PK_SRC
					if (bus_we && bus_adr[5:4] == 2'b01) case (bus_adr[3:2])
					2'd0: begin
						ci_go <= 1'b1;
						pk_n <= bus_wdat[12:0];
						pk_nb <= bus_wdat[22:16];
					end
					2'd1: co_dst <= bus_wdat[RAM_AW-1:0];
					2'd2: pk_src <= bus_wdat[RAM_AW-1:0];
					default: ;
					endcase
				end else if (bus_adr[16:14] == 3'b001) begin
					case (bus_adr[5:2])
					4'd0: if (bus_we) r_addr <= bus_wdat[ADDR_BITS-1:0];
					4'd1: if (bus_we) r_stride <= bus_wdat[ADDR_BITS-1:0];
					4'd2: if (bus_we) r_rows <= bus_wdat[12:0];
					4'd3: if (bus_we) r_groups <= bus_wdat[7:0];
					4'd4: if (bus_we) r_q1 <= (bus_wdat[3:0] == 4'd1);   // 1-bit (else the KV cache's 8-bit)
					4'd5: start <= 1'b1;
					4'd10: r_shift <= bus_wdat[5:0];
					4'd11: r_obank <= bus_wdat[0];
					4'd14: if (bus_we) glu_n <= bus_wdat[12:0];
					4'd15: if (bus_we) begin
						glu_sg <= bus_wdat[5:0];
						glu_su <= bus_wdat[13:8];
						glu_sq <= bus_wdat[23:16];
					end
					default: ;
					endcase
				end
			end
		end
	end

	// ---------------------------------------------------------------
	// row geometry

	// words per row = full * (1 + 4 wpg) + (rem ? 1 + rem wpg : 0)
	wire [5:0] g_full = r_groups[7:2];
	wire [1:0] g_rem = r_groups[1:0];
	// registered: QBITS and GROUPS are written at least a bus cycle
	// before CTRL starts a run, and the multiply-add is a long path
	reg [10:0] row_words;
	always @(posedge clk)
		row_words <= r_q1
			? ({2'd0, g_full, 3'd0} + {5'd0, g_full} + (g_rem != 2'd0 ? 11'd1 + {8'd0, g_rem, 1'b0} : 11'd0))
			: ({1'd0, g_full, 4'd0} + {5'd0, g_full} + (g_rem != 2'd0 ? 11'd1 + {7'd0, g_rem, 2'b00} : 11'd0));
	wire [9:0] row_blocks = row_words[10:1] + {9'd0, row_words[0]};
	reg [10:0] rw;                  // row_words, latched at start
	reg [9:0] rb;                   // row_blocks, latched at start

	// ---------------------------------------------------------------
	// request side: rows in pieces of up to PIECE blocks, credit-limited

	localparam FDEPTH = 1 << FIFO_LOG;
	localparam PIECE = FDEPTH / 2;
	reg [12:0] rq_row;
	reg [ADDR_BITS-1:0] rq_base, rq_ptr;
	reg [9:0] rq_rem;               // blocks of this row not yet asked for
	reg [FIFO_LOG:0] credits;       // blocks we may still ask for
	reg rq_active;
	// after a start or a handshake, one cycle registers the next
	// request's decisions (last row? piece size?) for the issue cycle
	reg rq_prep, rq_last;
	reg [9:0] piece_r;
	wire consume_block;             // a block leaves the head register
	wire [9:0] piece = (rq_rem > PIECE) ? PIECE[9:0] : rq_rem;
	wire hs = sq_valid && sq_ready;

	always @(posedge clk) begin
		if (rst) begin
			sq_valid <= 1'b0;
			rq_active <= 1'b0;
			rq_prep <= 1'b0;
			credits <= FDEPTH[FIFO_LOG:0];
		end else begin
			credits <= credits + {{FIFO_LOG{1'b0}}, consume_block}
			           - (hs ? sq_len[FIFO_LOG:0] : {(FIFO_LOG+1){1'b0}});
			if (start) begin
				rq_active <= 1'b1;
				rq_prep <= 1'b1;
				rq_row <= 0;
				rq_base <= r_addr;
				rq_ptr <= r_addr;
				rq_rem <= row_blocks;
				rw <= row_words;
				rb <= row_blocks;
			end else if (hs) begin
				sq_valid <= 1'b0;
				rq_prep <= 1'b1;
				if (rq_rem == {1'b0, sq_len}) begin
					rq_row <= rq_row + 1'b1;
					rq_base <= rq_base + r_stride;
					rq_ptr <= rq_base + r_stride;
					rq_rem <= rb;
				end else begin
					rq_rem <= rq_rem - {1'b0, sq_len};
					rq_ptr <= rq_ptr + {{(ADDR_BITS-13){1'b0}}, sq_len, 4'd0};
				end
			end else if (rq_prep) begin
				rq_prep <= 1'b0;
				rq_last <= (rq_row == r_rows);
				piece_r <= piece;
			end else if (rq_active && !sq_valid) begin
				if (rq_last)
					rq_active <= 1'b0;
				else if ({{(10-FIFO_LOG-1){1'b0}}, credits} >= piece_r) begin
					sq_valid <= 1'b1;
					sq_blk <= rq_ptr[ADDR_BITS-1:4];
					sq_len <= piece_r[8:0];
				end
			end
		end
	end

	// ---------------------------------------------------------------
	// data FIFO (distributed RAM) and its head register

	reg [127:0] fifo [0:FDEPTH-1];
	reg [FIFO_LOG:0] f_wp, f_rp;
	wire f_empty = (f_wp == f_rp);
	always @(posedge clk) begin
		if (sd_valid) fifo[f_wp[FIFO_LOG-1:0]] <= sd_data;
		if (rst) f_wp <= 0;
		else if (sd_valid) f_wp <= f_wp + 1'b1;
	end
	reg [127:0] f_head;
	reg f_head_v;

	// ---------------------------------------------------------------
	// stage 1: walk the row's words

	reg half;                       // 0: low word of the block, 1: high
	reg [10:0] wc;                  // word in row
	reg [4:0] wic;                  // word in chunk (0 = header)
	reg [5:0] chunk;
	reg [12:0] row;                 // row being consumed
	reg consuming;

	wire [4:0] chunk_words = r_q1 ? 5'd9 : 5'd17;
	wire take = consuming && f_head_v;
	// the row's last word: registered alongside wc (a compare in the
	// consume path otherwise, which fans out to the FIFO head)
	reg rlast;
	reg [10:0] rwm1;                // row words - 1
	assign consume_block = take && (half || rlast);

	always @(posedge clk) begin
		if (rst || start) begin
			f_rp <= rst ? 0 : f_rp;
			f_head_v <= rst ? 1'b0 : f_head_v;
		end
		if (!rst && (!f_head_v || consume_block)) begin
			if (!f_empty) begin
				f_head <= fifo[f_rp[FIFO_LOG-1:0]];
				f_head_v <= 1'b1;
				f_rp <= f_rp + 1'b1;
			end else
				f_head_v <= 1'b0;
		end
	end

	reg s1_v, s1_hdr, s1_glast, s1_rlast;
	reg [63:0] s1_word;
	reg [1:0] s1_gic;               // group in chunk
	reg [2:0] s1_sub;               // 8-bit: which 64 bits of the X row
	reg s1_odd;                     // 8-bit: the row's odd block
	reg [12:0] s1_row;

	wire is_hdr = (wic == 5'd0);
	wire [4:0] dj = wic - 5'd1;     // data word index in chunk
	wire [1:0] gic = r_q1 ? dj[2:1] : dj[3:2];
	wire [1:0] wig = r_q1 ? {1'b0, dj[0]} : dj[1:0];
	wire glast = r_q1 ? (dj[0] == 1'b1) : (dj[1:0] == 2'b11);
	// the X row of the word's first weight: 1-bit, a word is 64 weights
	// (a whole row); 8-bit, 8 weights (an eighth of one)
	wire [5:0] xrow = r_q1 ? {chunk[2:0], gic, wig[0]} : {chunk[4:0], gic[1]};

	always @(posedge clk) begin
		s1_v <= 1'b0;
		if (rst) begin
			consuming <= 1'b0;
		end else if (start) begin
			consuming <= 1'b1;
			half <= 1'b0; wc <= 0; wic <= 0; chunk <= 0; row <= 0;
			rwm1 <= row_words - 1'b1;
			rlast <= (row_words == 11'd1);
		end else if (take) begin
			s1_v <= 1'b1;
			s1_word <= half ? f_head[127:64] : f_head[63:0];
			s1_hdr <= is_hdr;
			s1_gic <= gic;
			s1_glast <= !is_hdr && glast;
			s1_rlast <= rlast;
			s1_sub <= {gic[0], wig};
			s1_odd <= gic[0];
			s1_row <= row;
			x_raddr <= xrow;
			if (rlast) begin
				half <= 1'b0;
				wc <= 0; wic <= 0; chunk <= 0;
				rlast <= (rwm1 == 11'd0);
				row <= row + 1'b1;
				if (row + 1'b1 == r_rows) consuming <= 1'b0;
			end else begin
				half <= ~half;
				wc <= wc + 1'b1;
				rlast <= (wc + 1'b1 == rwm1);
				if (wic == chunk_words - 1'b1) begin
					wic <= 0;
					chunk <= chunk + 1'b1;
				end else wic <= wic + 1'b1;
			end
		end
	end

	// ---------------------------------------------------------------
	// The word's control, delayed alongside its data: stage k of the
	// pipeline sees c_*[k]. Stage 2 is the cycle X arrives.
	localparam NST = 11;
	reg c_v [2:NST];
	reg c_hdr [2:NST];
	reg c_glast [2:NST];
	reg c_rlast [2:NST];
	reg [1:0] c_gic [2:NST];
	reg [12:0] c_row [2:NST];
	reg [3:0] c_da [3:NST];         // 1-bit: the row's two blocks' offsets; 8-bit: c_da is the block's
	reg [3:0] c_db [3:NST];
	integer ci;
	always @(posedge clk) begin
		c_v[2] <= s1_v && !rst;
		c_hdr[2] <= s1_hdr; c_glast[2] <= s1_glast; c_rlast[2] <= s1_rlast;
		c_gic[2] <= s1_gic; c_row[2] <= s1_row;
		for (ci = 3; ci <= NST; ci = ci + 1) begin
			c_v[ci] <= c_v[ci - 1] && !rst;
			c_hdr[ci] <= c_hdr[ci - 1]; c_glast[ci] <= c_glast[ci - 1]; c_rlast[ci] <= c_rlast[ci - 1];
			c_gic[ci] <= c_gic[ci - 1]; c_row[ci] <= c_row[ci - 1];
		end
		c_da[3] <= r_q1 ? xde_q : (s2_odd ? xdo_q : xde_q);
		c_db[3] <= xdo_q;
		for (ci = 4; ci <= NST; ci = ci + 1) begin
			c_da[ci] <= c_da[ci - 1];
			c_db[ci] <= c_db[ci - 1];
		end
	end
	reg [63:0] s2_word;
	reg [2:0] s2_sub, s3_sub;
	reg s2_odd;
	// the X row, registered before anything uses it (the RAM's output
	// to 512 loads is a long path otherwise): both modes start in stage 3
	reg [511:0] xr3;
	always @(posedge clk) begin
		s2_word <= s1_word; s2_sub <= s1_sub; s2_odd <= s1_odd;
		xr3 <= x_rdata; s3_sub <= s2_sub;
	end

	// ---------------------------------------------------------------
	// 1-bit: per block of 32, S = sum (bit ? q : ~q) + (number of 0
	// bits) = sum (bit ? q : -q). Stage 2 picks q or ~q; 3 and 4 add in
	// fours; 5 adds the last two and the count.

	reg [7:0] tq [0:63];             // stage 4
	reg [63:0] s3_word;
	integer k;
	always @(posedge clk) begin
		for (k = 0; k < 64; k = k + 1)
			tq[k] <= s3_word[k] ? xr3[8*k +: 8] : ~xr3[8*k +: 8];
		s3_word <= s2_word;
	end
	reg signed [9:0] u4 [0:15];
	// the zero count in two steps: per byte, then the four bytes of a block
	reg [3:0] zb3 [0:7];
	function [3:0] pop8;
		input [7:0] v;
		pop8 = {3'd0, v[0]} + {3'd0, v[1]} + {3'd0, v[2]} + {3'd0, v[3]} +
		       {3'd0, v[4]} + {3'd0, v[5]} + {3'd0, v[6]} + {3'd0, v[7]};
	endfunction
	always @(posedge clk) begin
		for (k = 0; k < 16; k = k + 1)
			u4[k] <= $signed(tq[4*k]) + $signed(tq[4*k+1]) + $signed(tq[4*k+2]) + $signed(tq[4*k+3]);
		for (k = 0; k < 8; k = k + 1)
			zb3[k] <= pop8(~s4_word[8*k +: 8]);
	end
	reg signed [11:0] u16 [0:3];
	reg [5:0] zc4a, zc4b;
	always @(posedge clk) begin
		for (k = 0; k < 4; k = k + 1)
			u16[k] <= u4[4*k] + u4[4*k+1] + u4[4*k+2] + u4[4*k+3];
		zc4a <= {2'd0, zb3[0]} + {2'd0, zb3[1]} + {2'd0, zb3[2]} + {2'd0, zb3[3]};
		zc4b <= {2'd0, zb3[4]} + {2'd0, zb3[5]} + {2'd0, zb3[6]} + {2'd0, zb3[7]};
	end
	reg signed [13:0] sa5, sb5;      // the two blocks' sums (valid in stage 7)
	always @(posedge clk) begin
		sa5 <= u16[0] + u16[1] + $signed({1'b0, zc4a});
		sb5 <= u16[2] + u16[3] + $signed({1'b0, zc4b});
	end

	// ---------------------------------------------------------------
	// 8-bit (attention): the word's 8 weights times their 8 x values
	// (stage 2 selects them; DSPs give the products in stage 5); stage
	// 6 sums them, the group's sum accumulates over its 4 words.

	reg signed [7:0] w8 [0:7];
	reg signed [7:0] q8 [0:7];
	wire [63:0] xsel8 = xr3[64*s3_sub +: 64];
	always @(posedge clk) begin
		for (k = 0; k < 8; k = k + 1) begin
			w8[k] <= s3_word[8*k +: 8];
			q8[k] <= xsel8[8*k +: 8];
		end
	end
	wire signed [35:0] p8 [0:7];
	genvar gm;
	generate for (gm = 0; gm < 8; gm = gm + 1) begin : m8
		mlm_mul18s mu (.clk(clk), .a({{10{w8[gm][7]}}, w8[gm]}), .b({{10{q8[gm][7]}}, q8[gm]}), .p(p8[gm]));
	end endgenerate
	reg signed [18:0] ps8a, ps8b;    // the products in two fours (stage 7)
	always @(posedge clk) begin
		ps8a <= $signed(p8[0][16:0]) + $signed(p8[1][16:0]) + $signed(p8[2][16:0]) + $signed(p8[3][16:0]);
		ps8b <= $signed(p8[4][16:0]) + $signed(p8[5][16:0]) + $signed(p8[6][16:0]) + $signed(p8[7][16:0]);
	end
	// the group's running sum (ps8a/b are stage 6's); g8tot, in stage
	// 7, is the total with this word's products
	reg signed [21:0] g8acc, g8tot;
	always @(posedge clk) begin
		if (rst || start) g8acc <= 0;
		else if (c_v[7] && !c_hdr[7] && !r_q1) begin
			if (c_glast[7]) g8acc <= 0;
			else g8acc <= g8acc + ps8a + ps8b;
		end
		g8tot <= g8acc + ps8a + ps8b;
	end

	// ---------------------------------------------------------------
	// 1-bit: S x m (DSPs: products in stage 8); the multipliers come
	// from each chunk's header word as it passes stage 5

	reg [63:0] mults;
	reg [15:0] m6;
	always @(posedge clk) begin
		if (c_v[6] && c_hdr[6]) mults <= s6_word;
		m6 <= mults[16*c_gic[6] +: 16];
	end
	reg [63:0] s4_word, s5_word, s6_word;
	always @(posedge clk) begin s4_word <= s3_word; s5_word <= s4_word; s6_word <= s5_word; end
	// stage 7: sa5, sb5 and m6; the products in stage 9
	wire signed [35:0] pa8, pb8;
	mlm_mul18s mua (.clk(clk), .a({{4{sa5[13]}}, sa5}), .b({2'b00, m6}), .p(pa8));
	mlm_mul18s mub (.clk(clk), .a({{4{sb5[13]}}, sb5}), .b({2'b00, m6}), .p(pb8));

	// ---------------------------------------------------------------
	// stage 8: each term shifted by 15 - d; stage 9: the word's total
	// (1-bit: both blocks; 8-bit: the group's sum on its last word)

	reg signed [21:0] g8t8;
	always @(posedge clk) g8t8 <= g8tot;
	reg signed [45:0] ta9, tb9;
	always @(posedge clk) begin
		if (r_q1) begin
			ta9 <= $signed(pa8[30:0]) <<< (4'd15 - c_da[9]);
			tb9 <= $signed(pb8[30:0]) <<< (4'd15 - c_db[9]);
		end else begin
			ta9 <= (c_glast[9]) ? ($signed(g8t8) <<< (4'd15 - c_da[9])) : 46'sd0;
			tb9 <= 46'sd0;
		end
	end
	reg signed [46:0] w10;
	always @(posedge clk) w10 <= ta9 + tb9;

	// stage 10 feeds the row accumulator (below) as s6 did
	wire s6_v = c_v[11] && !c_hdr[11];
	wire s6_rlast = c_rlast[11];
	wire [12:0] s6_row = c_row[11];
	wire signed [46:0] gprod = w10;

	reg signed [63:0] acc;
	reg [12:0] rows_done;
	wire signed [63:0] acc_next = acc + {{17{gprod[46]}}, gprod};
	// |result| of the row just written, for MAXABS: the absolute value a
	// cycle later, the comparison the cycle after (two long carry
	// chains otherwise in one cycle)
	wire [63:0] wabs = out_wdata[63] ? (~out_wdata + 64'd1) : out_wdata;
	reg [63:0] wabs_r;
	reg wabs_v;
	always @(posedge clk) begin
		wabs_v <= out_we && !rst;
		wabs_r <= wabs;
	end
	wire mx_pend = out_we || wabs_v;
	always @(posedge clk) begin
		out_we <= 1'b0;
		if (rst || start) begin
			acc <= 0;
			rows_done <= 0;
			maxabs <= 64'd0;
			if (start) run_bank <= r_obank;
		end else begin
			if (wabs_v && wabs_r > maxabs) maxabs <= wabs_r;
			if (s6_v) begin
				if (s6_rlast) begin
					out_we <= 1'b1;
					out_waddr <= {run_bank, s6_row[11:0]};
					out_wdata <= acc_next;
					acc <= 0;
					rows_done <= rows_done + 1'b1;
				end else
					acc <= acc_next;
			end
		end
	end

	// ---------------------------------------------------------------
	// busy, statistics

	always @(posedge clk) begin
		if (rst) begin
			busy <= 1'b0;
			max0 <= 64'd0;
			max1 <= 64'd0;
		end else if (start) begin
			busy <= 1'b1;
			r_cycles <= 0;
			r_waits <= 0;
		end else if (busy) begin
			r_cycles <= r_cycles + 1'b1;
			if (consuming && !f_head_v) r_waits <= r_waits + 1'b1;
			if (rows_done == r_rows && !out_we && !mx_pend) begin
				busy <= 1'b0;
				if (run_bank) max1 <= maxabs; else max0 <= maxabs;
			end
		end
	end


	// ---------------------------------------------------------------
	// GLU unit: glu_spec() in fw/common/eng_spec.h, one row at a time

	reg [15:0] sigtab [0:1024];
`include "mlm_sigtab.vh"
	reg [9:0] tj;
	reg [15:0] t0, t1;
	always @(posedge clk) begin
		t0 <= sigtab[tj];
		t1 <= sigtab[{1'b0, tj} + 11'd1];      // 11 bits: tj = 1023 reads entry 1024
	end

	// rshr32: x * 2^-s rounded half up; s <= 0 shifts left
	// (one expression, no local variables: a function's locals are shared
	// between its call sites)
	// signed in every branch, so >>> stays arithmetic
	function signed [31:0] rshr32;
		input signed [31:0] x;
		input signed [7:0] sh;
		rshr32 = (sh <= 8'sd0) ? (x <<< (-sh)) :
		         (sh > 8'sd31) ? 32'sd0 :
		         (((x >>> (sh - 8'sd1)) + 32'sd1) >>> 1);
	endfunction

	// bit length as a balanced tree (a loop over the bits becomes a
	// 32-deep priority chain)
	function [2:0] bl4;
		input [3:0] v;
		bl4 = v[3] ? 3'd4 : v[2] ? 3'd3 : v[1] ? 3'd2 : v[0] ? 3'd1 : 3'd0;
	endfunction
	function [3:0] bl8;
		input [7:0] v;
		bl8 = (v[7:4] != 4'd0) ? {1'b0, bl4(v[7:4])} + 4'd4 : {1'b0, bl4(v[3:0])};
	endfunction
	function [4:0] bl16;
		input [15:0] v;
		bl16 = (v[15:8] != 8'd0) ? {1'b0, bl8(v[15:8])} + 5'd8 : {1'b0, bl8(v[7:0])};
	endfunction
	function [5:0] bitlen32;
		input [31:0] v;
		bitlen32 = (v[31:16] != 16'd0) ? {1'b0, bl16(v[31:16])} + 6'd16 : {1'b0, bl16(v[15:0])};
	endfunction

	// Every step is registered: the unit is not throughput critical
	// (~25 cycles per element) but must close timing next to the engine.
	localparam G_IDLE = 5'd0,
	           G1_RD = 5'd1, G1_W = 5'd2, G1_SUM = 5'd3, G1_G = 5'd4, G1_ABS = 5'd5,
	           G1_A = 5'd6, G1_T = 5'd7, G1_TM = 5'd8, G1_SIG = 5'd9, G1_S = 5'd10,
	           G1_AB = 5'd11, G1_WR = 5'd12, G1_END = 5'd13,
	           G2_RD = 5'd14, G2_W = 5'd15, G2_W2 = 5'd16, G2_SUM = 5'd17, G2_G = 5'd18,
	           G2_S = 5'd19, G2_AB = 5'd20, G2_P = 5'd21, G2_END = 5'd22,
	           G3_RD = 5'd23, G3_Q = 5'd24, G3_P = 5'd25, G1_TR = 5'd26,
	           G1_SM = 5'd27, G2_SM = 5'd28, G2_END2 = 5'd29, G3_PA = 5'd30, G2_SB = 5'd31;
	reg [4:0] gst;
	reg [12:0] gi;
	reg signed [31:0] gg, gs, g15r, gsp, gu, pqr, pqa, gsp_p;
	reg [1:0] xs_mode;
	reg [7:0] xs_amt;
	// MX output (eng_mx_pack): each block's s_b from pass 2, the vector's
	// s, and per block in pass 3 the shift max(s - 15, s_b) and offset
	reg [31:0] bmx;
	reg [5:0] bmx_bl;
	reg [4:0] sb [0:127];
	reg [4:0] glu_s, sbk, sbr;       // sbr: the next block's s_b, fetched ahead
	reg [3:0] glu_dcur;
	reg [23:0] xw_acc;
	always @(posedge clk) bmx_bl <= bitlen32(bmx);
	// gi is the last element: registered (gi changes only at an element's
	// end, and every element takes several states before its end)
	reg g_last;
	always @(posedge clk) g_last <= (gi + 13'd1 == glu_n);
	reg [31:0] ga, gabs, sab, tm;
	reg [63:0] gacc;
	reg [9:0] gf;
	reg gneg;
	reg [16:0] gsig;
	reg [31:0] smx, pmx;
	reg [15:0] xlo;
	reg [1:0] gw;                     // wait counter for the RAM
	reg [63:0] g_rnd0, g_rnd1;
	reg [63:0] gq;                    // out_q, one cycle later (the RAM output is a long path)
	always @(posedge clk) begin
		gq <= out_q;
		g_rnd0 <= (glu_sg == 6'd0) ? 64'd0 : (64'd1 << (glu_sg - 6'd1));
		g_rnd1 <= (glu_su == 6'd0) ? 64'd0 : (64'd1 << (glu_su - 6'd1));
	end
	wire signed [63:0] gacc_s = gacc;
	wire signed [63:0] g_sh0 = gacc_s >>> glu_sg;
	wire signed [63:0] g_sh1 = gacc_s >>> glu_su;
	// a = |g| in Q16
	// a = |g| scaled by sq into Q16, saturating at 16: the shift amounts
	// decoded from sq once per run, the shifts (G1_A's first cycle) and
	// the choice (its second) apart
	reg sq_pos, sq_big;
	reg [4:0] sq_l, sq_thr;
	reg signed [7:0] sq_neg;
	always @(posedge clk) begin
		sq_pos <= (glu_sq >= 8'sd0);
		sq_big <= (glu_sq > 8'sd20);
		sq_l <= glu_sq[4:0];
		sq_thr <= 5'd20 - glu_sq[4:0];
		sq_neg <= -glu_sq;
		sq_negm1 <= -glu_sq - 8'sd1;
		sq_nbig <= (-glu_sq > 8'sd31);
	end
	reg [31:0] a_l, a_rp;
	reg a_ovf;
	reg signed [7:0] sq_negm1;
	reg sq_nbig;
	// rshr32(|g|, -sq) = ((|g| >> (-sq - 1)) + 1) >> 1: the shift in the
	// first cycle, the rounding in the second
	wire [31:0] a_r = (a_rp + 32'd1) >> 1;
	wire [31:0] a_val = sq_pos ? ((sq_big || a_ovf) ? (32'd16 << 16) : a_l) : a_r;
	reg [15:0] t0r, t1r;              // the table's output, registered
	// the two products, computed every cycle from registers; a state
	// waits a cycle and then takes them (the multipliers are kept out
	// of the state machine's muxes)
	// Only the low 32 bits of each product are used: they come from
	// three 16x16 products and one sum, each registered (a 32x32
	// multiplier cascade is too slow for 100 MHz), two cycles after the
	// operands
	reg signed [31:0] pm1, pm2;
	// the six 16 x 16 products, each in a DSP block with its own input
	// and output registers (mlm_mul16r: two cycles); the S states wait
	// three cycles for pm1/pm2. u1b = {15'd0, gsig}: its high half is
	// gsig[16] alone.
	wire [31:0] m1a, m1b, m1c, m2a, m2b, m2c;
	reg gend2;
	// pass 1's g >> 15, carried to stage 14 (the pipelined GLU, below)
	reg signed [31:0] g15c [6:14];
	wire signed [31:0] g15c14 = g15c[14];
	mlm_mul16r mu1a (.clk(clk), .a(g15c14[15:0]),  .b(gsig[15:0]),          .p(m1a));
	mlm_mul16r mu1b (.clk(clk), .a(g15c14[31:16]), .b(gsig[15:0]),          .p(m1b));
	mlm_mul16r mu1c (.clk(clk), .a(g15c14[15:0]),  .b({15'd0, gsig[16]}),   .p(m1c));
	mlm_mul16r mu2a (.clk(clk), .a(gsp[15:0]),   .b(gu[15:0]),            .p(m2a));
	mlm_mul16r mu2b (.clk(clk), .a(gsp[31:16]),  .b(gu[15:0]),            .p(m2b));
	mlm_mul16r mu2c (.clk(clk), .a(gsp[15:0]),   .b(gu[31:16]),           .p(m2c));
	always @(posedge clk) begin
		pm1 <= m1a + ((m1b + m1c) << 16);
		pm2 <= m2a + ((m2b + m2c) << 16);
	end
	// t1 - t0 as 17-bit signed: the product mod 2^32 is the same as
	// with 32-bit operands, and fits one 18x18 multiplier
	wire signed [16:0] t_diff17 = $signed({1'b0, t1r}) - $signed({1'b0, t0r});
	reg signed [16:0] tdr;
	wire signed [31:0] tm_p = tdr * $signed({1'b0, gf});
	wire [31:0] t_int = {16'd0, t0r} + (tm >> 10);
	// sigmoid in two steps: gsv = the table value (or 1.0 past 16),
	// then the sign and the rounding in one adder
	reg ga_big;
	reg [31:0] gsv;
	wire [31:0] sig_r = gneg ? (32'd65537 - gsv) : (gsv + 32'd1);
	// bit lengths, registered (a priority encoder: a long path); the END
	// states wait a cycle for them
	reg [5:0] smx_bl, pmx_bl;
	reg gend;
	always @(posedge clk) begin
		smx_bl <= bitlen32(smx);
		pmx_bl <= bitlen32(pmx);
	end
	wire signed [7:0] pq_c = (pqr > 127) ? 8'sd127 : (pqr < -127) ? -8'sd127 : pqr[7:0];

	// ---------------------------------------------------------------
	// The GLU unit, pipelined (glu_spec() in eng_spec.h): pass 1 takes an
	// element a cycle through 17 stages, pass 2 one every two cycles
	// (both banks through the OUT RAM's one read port), pass 3 one a
	// cycle. Stage k of an element is the cycle k after its read was
	// issued (glu_raddr set at k = 0, its data in gq at k = 3). Each stage
	// is one step of the former state machine; values a later stage needs
	// travel with the element.
	localparam GP_IDLE = 3'd0, GP_1 = 3'd1, GP_1E = 3'd2, GP_2 = 3'd3, GP_2E = 3'd4,
	           GP_3 = 3'd5, GP_3E = 3'd6;
	reg [2:0] gph;
	reg [12:0] gie;                   // the next element to issue
	reg gtog;                         // pass 2: its second read
	reg [5:0] gdr;                    // drain count
	// pass 1: valid and element per stage (3..19)
	reg [19:1] p1v;
	reg [11:0] p1e [1:19];
	reg gneg6 [6:13];
	reg [31:0] ga8 [8:10];
	reg [9:0] gf8 [8:11];
	reg [15:0] t0c [10:12];
	reg ga_big11 [11:12];
	reg signed [31:0] gs19;
	// pass 2: valid and element per stage (1..13)
	reg [13:1] p2v;
	reg [11:0] p2e [1:13];
	reg signed [31:0] gs2;
	reg p2be12;                       // a block ended (its max is in bmx)
	reg [6:0] p2bk12;
	// pass 3
	reg [5:1] p3v;
	reg [11:0] p3e [1:5];
	reg [1:0] p3m [1:4];
	reg [7:0] p3a [1:3];
	reg [3:0] p3d [1:5];
	reg [1:0] cur_m;
	reg [7:0] cur_a;
	reg [3:0] cur_d;

	// pass 1's arithmetic, on the stage's own values
	wire signed [16:0] p1_tdiff = $signed({1'b0, t1r}) - $signed({1'b0, t0r});
	wire signed [31:0] p1_tm = tdr * $signed({1'b0, gf8[11]});
	wire [31:0] p1_tint = {16'd0, t0c[12]} + (tm >> 10);
	wire [31:0] p1_sig = gneg6[13] ? (32'd65537 - gsv) : (gsv + 32'd1);
	// pass 3: the block's shift and offset at its first element
	wire [4:0] p3_sbk = (glu_s > 5'd15 && glu_s - 5'd15 > sbr) ? (glu_s - 5'd15) : sbr;

	integer gk;
	always @(posedge clk) begin
		glu_owe <= 1'b0;
		glu_xwe <= 1'b0;
		glu_dwe <= 1'b0;
		p1v <= {p1v[18:1], 1'b0};
		p2v <= {p2v[12:1], 1'b0};
		p3v <= {p3v[4:1], 1'b0};
		for (gk = 2; gk <= 19; gk = gk + 1) p1e[gk] <= p1e[gk - 1];
		for (gk = 2; gk <= 13; gk = gk + 1) p2e[gk] <= p2e[gk - 1];
		for (gk = 2; gk <= 5; gk = gk + 1) begin p3e[gk] <= p3e[gk - 1]; p3d[gk] <= p3d[gk - 1]; end
		for (gk = 2; gk <= 4; gk = gk + 1) p3m[gk] <= p3m[gk - 1];
		for (gk = 2; gk <= 3; gk = gk + 1) p3a[gk] <= p3a[gk - 1];
		if (rst) begin
			gph <= GP_IDLE;
			glu_busy <= 1'b0;
			glu_on <= 1'b0;
			p1v <= 0; p2v <= 0; p3v <= 0;
		end else begin
			// -- issue -------------------------------------------------
			case (gph)
			GP_IDLE: if (glu_go) begin
				glu_busy <= 1'b1;
				glu_on <= 1'b1;
				gie <= 13'd0;
				smx <= 32'd0;
				pmx <= 32'd0;
				gph <= GP_1;
			end
			GP_1: begin
				glu_raddr <= {1'b0, gie[11:0]};
				p1v[1] <= 1'b1;
				p1e[1] <= gie[11:0];
				if (gie + 13'd1 == glu_n) begin gph <= GP_1E; gdr <= 6'd24; end
				gie <= gie + 13'd1;
			end
			GP_1E: if (gdr != 6'd0) gdr <= gdr - 6'd1; else begin
				glu_psh <= (smx_bl > 6'd15) ? {2'd0, smx_bl - 6'd15} : 8'd0;
				gie <= 13'd0; gtog <= 1'b0;
				gph <= GP_2;
			end
			GP_2: begin
				// the element's s (bank 0), then its up (bank 1)
				glu_raddr <= {gtog, gie[11:0]};
				gtog <= ~gtog;
				if (!gtog) begin
					p2v[1] <= 1'b1;
					p2e[1] <= gie[11:0];
				end else begin
					if (gie + 13'd1 == glu_n) begin gph <= GP_2E; gdr <= 6'd18; end
					gie <= gie + 13'd1;
				end
			end
			GP_2E: if (gdr != 6'd0) gdr <= gdr - 6'd1; else begin
				glu_s <= (pmx_bl > 6'd7) ? (pmx_bl[4:0] - 5'd7) : 5'd0;
				glu_xsh <= ((pmx_bl > 6'd7) ? ($signed({2'b00, pmx_bl}) - 8'sd7) : 8'sd0) - 8'sd15;
				sbr <= sb[7'd0];
				gie <= 13'd0;
				gph <= GP_3;
			end
			GP_3: begin
				glu_raddr <= {1'b0, gie[11:0]};
				p3v[1] <= 1'b1;
				p3e[1] <= gie[11:0];
				if (gie[4:0] == 5'd0) begin
					// a block starts: its shift max(s - 15, s_b), offset s - shift
					cur_m <= (p3_sbk == 5'd0) ? 2'd0 : 2'd2;
					cur_a <= (p3_sbk == 5'd0) ? 8'd0 : {3'd0, p3_sbk - 5'd1};
					cur_d <= glu_s - p3_sbk;
					p3m[1] <= (p3_sbk == 5'd0) ? 2'd0 : 2'd2;
					p3a[1] <= (p3_sbk == 5'd0) ? 8'd0 : {3'd0, p3_sbk - 5'd1};
					p3d[1] <= glu_s - p3_sbk;
				end else begin
					p3m[1] <= cur_m; p3a[1] <= cur_a; p3d[1] <= cur_d;
				end
				if (gie[4:0] == 5'd31) sbr <= sb[gie[11:5] + 7'd1];   // the next block's s_b
				if (gie + 13'd1 == glu_n) begin gph <= GP_3E; gdr <= 6'd8; end
				gie <= gie + 13'd1;
			end
			GP_3E: if (gdr != 6'd0) gdr <= gdr - 6'd1; else begin
				glu_busy <= 1'b0;
				glu_on <= 1'b0;
				gph <= GP_IDLE;
			end
			default: gph <= GP_IDLE;
			endcase

			// -- pass 1: g, sigmoid, s = SiLU, into bank 0 -------------
			if (p1v[3]) gacc <= gq + g_rnd0;                       // stage 3
			if (p1v[4]) gg <= g_sh0[31:0];                         // 4
			if (p1v[5]) begin                                      // 5
				gabs <= gg[31] ? (~gg + 32'd1) : gg;
				gneg6[6] <= gg[31];
				g15c[6] <= rshr32(gg, 8'sd15);
			end
			for (gk = 7; gk <= 13; gk = gk + 1) gneg6[gk] <= gneg6[gk - 1];
			for (gk = 7; gk <= 14; gk = gk + 1) g15c[gk] <= g15c[gk - 1];
			if (p1v[6]) begin                                      // 6
				a_l <= gabs << sq_l;
				a_ovf <= (gabs >> sq_thr) != 32'd0;
				a_rp <= sq_nbig ? 32'd0 : (gabs >> sq_negm1[4:0]);
			end
			if (p1v[7]) begin                                      // 7
				ga8[8] <= a_val;
				tj <= a_val[19:10];
				gf8[8] <= a_val[9:0];
			end
			for (gk = 9; gk <= 10; gk = gk + 1) ga8[gk] <= ga8[gk - 1];
			for (gk = 9; gk <= 11; gk = gk + 1) gf8[gk] <= gf8[gk - 1];
			// 8: the table reads tj (t0, t1 in 9)
			if (p1v[9]) begin t0r <= t0; t1r <= t1; end            // 9
			if (p1v[10]) begin                                     // 10
				tdr <= p1_tdiff;
				ga_big11[11] <= (ga8[10] >= (32'd16 << 16));
				t0c[11] <= t0r;
			end
			t0c[12] <= t0c[11];
			ga_big11[12] <= ga_big11[11];
			if (p1v[11]) tm <= p1_tm;                              // 11
			if (p1v[12]) gsv <= ga_big11[12] ? 32'd65535 : p1_tint;   // 12
			if (p1v[13]) gsig <= p1_sig >> 1;                      // 13
			// 14: u1 = g15c[14], gsig into the DSPs; pm1 in 17
			if (p1v[17]) gs19 <= pm1;                              // 17
			if (p1v[18]) begin                                     // 18
				sab <= gs19[31] ? (~gs19 + 32'd1) : gs19;
				gs <= gs19;
			end
			if (p1v[19]) begin                                     // 19
				glu_owe <= 1'b1;
				glu_waddr <= {1'b0, p1e[19]};
				glu_wdata <= {{32{gs[31]}}, gs};
				if (sab > smx) smx <= sab;
			end

			// -- pass 2: p = (s >> psh) x up, into bank 0 --------------
			if (p2v[3]) gs2 <= gq[31:0];                           // 3: s
			if (p2v[4]) begin                                      // 4: up
				gacc <= gq + g_rnd1;
				gsp_p <= (glu_psh == 8'd0) ? gs2 : (gs2 >>> (glu_psh[4:0] - 5'd1));
			end
			if (p2v[5]) begin                                      // 5
				gu <= g_sh1[31:0];
				gsp <= (glu_psh == 8'd0) ? gsp_p : ((gsp_p + 32'sd1) >>> 1);
			end
			// 6: gsp, gu into the DSPs; pm2 in 9
			if (p2v[9]) gs <= pm2;                                 // 9
			if (p2v[10]) sab <= gs[31] ? (~gs + 32'd1) : gs;       // 10
			if (p2v[11]) begin                                     // 11
				glu_owe <= 1'b1;
				glu_waddr <= {1'b0, p2e[11]};
				glu_wdata <= {{32{gs[31]}}, gs};
				if (sab > pmx) pmx <= sab;
				if (p2e[11][4:0] == 5'd0 || sab > bmx) bmx <= sab;
			end
			p2be12 <= p2v[11] && (p2e[11][4:0] == 5'd31 || {1'b0, p2e[11]} + 13'd1 == glu_n);
			p2bk12 <= p2e[11][11:5];
			// 13: the block's s_b (bmx_bl has the block's max now)
			if (p2be12_d) sb[p2bk13] <= (bmx_bl > 6'd7) ? (bmx_bl[4:0] - 5'd7) : 5'd0;

			// -- pass 3: int8, four to an X word; offsets per block ----
			if (p3v[3])                                            // 3
				pqa <= (p3m[3] == 2'd0) ? ($signed(gq[31:0]) <<< p3a[3][5:0]) :
				       ($signed(gq[31:0]) >>> p3a[3][5:0]);
			if (p3v[4]) pqr <= (p3m[4] == 2'd2) ? ((pqa + 32'sd1) >>> 1) : pqa;   // 4
			if (p3v[5]) begin                                      // 5
				if (p3e[5][1:0] != 2'd3) begin
					xw_acc[8*p3e[5][1:0] +: 8] <= pq_c;
				end else begin
					glu_xwe <= 1'b1;
					glu_xbank <= p3e[5][5:2];
					glu_xentry <= p3e[5][11:6];
					glu_xdata <= {pq_c, xw_acc};
				end
				if (p3e[5][4:0] == 5'd31) begin
					glu_dwe <= 1'b1;
					glu_dblk <= p3e[5][11:5];
					glu_ddata <= p3d[5];
				end
			end
		end
	end
	// pass 2's block end, a cycle later (bmx_bl is bitlen(bmx) registered)
	reg p2be12_d;
	reg [6:0] p2bk13;
	always @(posedge clk) begin
		p2be12_d <= p2be12 && !rst;
		p2bk13 <= p2bk12;
	end

endmodule

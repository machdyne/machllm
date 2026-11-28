/*
 * machllm -- vector unit: the model's per-element loops on the CPU's
 * own block RAM (docs/vu.md).
 *
 * The CPU writes the operands, then OP; that write is not acknowledged
 * until the operation is done, so the CPU is stalled (it cannot even
 * fetch) and the RAM's single port belongs to the unit meanwhile.
 * The specification is the v_* loops in fw/common/mlm_model.c; every
 * operation must give exactly their result.
 *
 * Registers (byte offsets):
 *   0x00 A, 0x04 B, 0x08 D   byte addresses in the RAM (word aligned)
 *   0x0c N                   elements
 *   0x10 P0, 0x14 P1         parameters (signed)
 *   0x18 OP                  write: run operation OP (ack when done)
 *   0x1c R0                  max of MAXABS, RESID, ATTSC, EXPCOEF
 *   0x20 SUMLO, 0x24 SUMHI   the sum of SQMUL, EXPCOEF
 *   0x28 ID                  "VU01"
 *
 * Operations (MLM_VU_* in fw/common/mlm_model.h):
 *   1 MAXABS   R0 = max |A[i]|
 *   2 SQMUL    u = A[i] >> P0, D[i] = u g[i] (g: int16 at B), SUM += u^2
 *   3 SCALE15  D[i] = (A[i] >> 15) P0
 *   4 RSHR     D[i] = A[i] >> P0 (P0 < 0: left)
 *   5 RESID    A[i] = (A[i] >> P0) + (B[i] >> P1); R0 = max |A[i]|
 *   6 MULSHR30 D[i] = (A[i] P0) >> 30 (64-bit product)
 *   8 KVQ8     int8 ((A[i] >> P0) P1) >> 16, clamped, four per D word
 *   9 ROPE     rotate pairs of A (P0 != 0: (i, i + N/2), else (2i, 2i+1))
 *              by B word i = {sin, cos} Q15
 *  10 ATTSC    A[t] = sat30(A[t] x mant(B[t]) << (P0 + exp(B[t]))), R0 = max
 *  15 EXPCOEF  e = exp(A[t] - P0) Q16 (table + interpolation), SUM += e,
 *              A[t] = (e x mant(B[t])) >> (P1 - exp(B[t])), R0 = max
 * (7, 11-14 were QPACK, EXPSUM, COEF, QPACKU, MXPACK: gone.)
 * ">>" is rshr32: rounded half up, 0 beyond 31; products wrap to 32
 * bits except MULSHR30's and KVQ8's.
 *
 * Every operation but MAXABS (a read a cycle) is streamed: reads at
 * fixed phases of a short period, each result written a fixed number
 * of cycles after its reads, so the port is busy every cycle (S_FS: 2
 * or 3 cycles an element; S_AS: 3; S_P5: ROPE 2.5, KVQ8 1.25).
 */
`default_nettype none
module mlm_vu #(
	parameter AW = 15                 // RAM word address bits
) (
	input wire clk,
	input wire rst,
	// bus (the SOC's target protocol: req held until ack)
	input wire bus_req,
	input wire bus_we,
	input wire [5:2] bus_adr,
	input wire [31:0] bus_wdat,
	output reg bus_ack,
	output reg [31:0] bus_rdat,
	// the RAM's port, while busy
	output wire ram_own,
	output reg [AW-1:0] ram_a,
	output reg ram_we,
	output reg [31:0] ram_wd,
	input wire [31:0] ram_q
);
	localparam OP_MAXABS = 4'd1, OP_SQMUL = 4'd2, OP_SCALE15 = 4'd3, OP_RSHR = 4'd4,
	           OP_RESID = 4'd5, OP_MULSHR30 = 4'd6, OP_KVQ8 = 4'd8,
	           OP_ROPE = 4'd9, OP_ATTSC = 4'd10, OP_EXPCOEF = 4'd15;

	// 2^(-i/256) in Q16 for EXPCOEF (fw/common/mlm_exp2tab.h)
	reg [16:0] exp2tab [0:256];
`include "mlm_exp2tab.vh"

	function signed [31:0] rshr32;
		input signed [31:0] x; input signed [7:0] sh;
		rshr32 = (sh <= 8'sd0) ? (x <<< (-sh)) : (sh > 8'sd31) ? 32'sd0 :
		         (((x >>> (sh - 8'sd1)) + 32'sd1) >>> 1);
	endfunction
	function [63:0] uabs64;
		input [63:0] x;
		uabs64 = x[63] ? (~x + 64'd1) : x;
	endfunction
	// bit length as a balanced tree
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
	function [5:0] bl32;
		input [31:0] v;
		bl32 = (v[31:16] != 16'd0) ? {1'b0, bl16(v[31:16])} + 6'd16 : {1'b0, bl16(v[15:0])};
	endfunction
	function [6:0] bitlen64;
		input [63:0] x;
		bitlen64 = (x[63:32] != 32'd0) ? {1'b0, bl32(x[63:32])} + 7'd32 : {1'b0, bl32(x[31:0])};
	endfunction
	function [31:0] uabs;
		input [31:0] x;
		uabs = x[31] ? (~x + 32'd1) : x;
	endfunction

	reg [31:0] r_a, r_b, r_d, r_n;
	reg signed [31:0] r_p0, r_p1;
	// shift parameters, clamped once (they are small; anything beyond
	// +-40 behaves the same)
	reg signed [7:0] s0, s1;
	reg [3:0] op;
	reg [31:0] mx;
	reg [63:0] sum;

	localparam S_IDLE = 4'd0, S_MX = 4'd1, S_DONE = 4'd7,
	           S_FS = 4'd12,                                         // SCALE15, RSHR, MULSHR30, SQMUL
	           S_AS = 4'd13,                                         // ATTSC, EXPCOEF, RESID
	           S_P5 = 4'd14;                                         // ROPE, KVQ8
	reg [3:0] st;
	reg [31:0] rq;                    // the RAM's output, registered (a long path otherwise)
	always @(posedge clk) rq <= ram_q;
	assign ram_own = (st != S_IDLE);

	reg [15:0] i, cnt;                // element (pair) index; how many
	wire [AW-1:0] aw = r_a[AW+1:2], bw = r_b[AW+1:2], dw = r_d[AW+1:2];
	wire split = (r_p0 != 32'sd0);
	wire [15:0] half = r_n[16:1];

	reg signed [33:0] ma, mb;
	reg signed [67:0] mp;
	reg [3:0] mv;                     // MAXABS: which pipeline steps hold an element
	reg signed [31:0] smx;            // ATTSC's (signed) maximum
	reg [31:0] ma0;

	// one product a cycle, registered (S_FS): operands set at a cycle
	// appear in mp two cycles later
	always @(posedge clk) mp <= ma * mb;

	// Streamed: SCALE15, RSHR, MULSHR30 an element every two cycles
	// (read A[j], then a write), SQMUL every three (A[j], B[j/2], a
	// write). Element j is read at R; its data are in rq at R + 2 (u),
	// its product(s) set up at R + 3 (and R + 4) and in mp two later,
	// the result written at R + 7 (R + 8 for SQMUL): with reads at
	// phase 0 (and 1) the write always falls on a free cycle.
	wire fs_op = (op == OP_SCALE15 || op == OP_RSHR || op == OP_MULSHR30 || op == OP_SQMUL);
	wire fs_sq = (op == OP_SQMUL);
	reg [8:1] fv;                     // fv[s]: an element read s cycles ago
	reg [8:1] fpar;                   // ... and its index's low bit
	reg [1:0] fph;                    // the period's phase (0 .. 1 or 2)
	reg [15:0] fji, fjw;              // elements read, written
	reg signed [31:0] fu3, fu4, fu5, fg4, fr6, fr7, fr8;
	wire fs_iss = (fph == 2'd0) && (fji < cnt);
	wire fs_wr = fs_sq ? fv[8] : fv[7];
	wire signed [63:0] fs_r30 = (mp[63:0] + 64'sd536870912) >>> 30;
	always @(posedge clk) if (st != S_FS) begin
		fv <= 8'd0; fph <= 2'd0; fji <= 16'd0; fjw <= 16'd0;
	end else begin
		fv <= {fv[7:1], fs_iss};
		fpar <= {fpar[7:1], fji[0]};
		fph <= (fph == (fs_sq ? 2'd2 : 2'd1)) ? 2'd0 : fph + 2'd1;
		if (fs_iss) fji <= fji + 16'd1;
		if (fv[2]) fu3 <= (op == OP_SCALE15) ? rshr32(rq, 8'sd15) : (op == OP_MULSHR30) ? rq : rshr32(rq, s0);
		if (fv[3]) begin
			fu4 <= fu3;
			fg4 <= fpar[3] ? {{16{rq[31]}}, rq[31:16]} : {{16{rq[15]}}, rq[15:0]};
		end
		if (fv[4]) fu5 <= fu4;
		if (fv[5]) begin
			fr6 <= (op == OP_SCALE15) ? mp[31:0] : (op == OP_MULSHR30) ? fs_r30[31:0] : fu5;
		end
		if (fv[6]) fr7 <= fs_sq ? mp[31:0] : fr6;
		if (fv[7]) fr8 <= fr7;
		if (fs_wr) fjw <= fjw + 16'd1;
	end

	// ATTSC, EXPCOEF and RESID streamed (S_AS): an element every three cycles,
	// A[j] read at phase 0 (cycle R), B[j] at phase 1, the result written
	// at R + 14 (phase 2 again). A's data is in rq at R + 2, B's at R + 3;
	// below, a register named for a cycle holds its value in that cycle.
	reg [14:1] av;                    // av[s]: an element read s cycles ago
	reg [1:0] aph;
	reg [15:0] aji, ajw;
	wire as_iss = (aph == 2'd0) && (aji < cnt);
	wire as_wr = av[14];
	wire as_e = (op == OP_EXPCOEF);
	wire as_rs = (op == OP_RESID);
	// RESID: (A >> P0) + (B >> P1), joining ATTSC's result chain at 8
	reg signed [31:0] rs3, rs4, rs5, rs6, rs7;
	// ATTSC (v_attsc): v = A mant(B), k = P0 + exp(B), then the shift
	reg signed [31:0] at3, ak4, ak5, ak6, ak7;
	reg signed [63:0] av4, av5, av6, av7, ash5, ash6, ash7, arn5, ashr6, ashr7;
	reg [63:0] aab5;
	reg [6:0] abl6;
	reg aov7;
	reg signed [31:0] ar8, ar9, ar10, ar11, ar12, ar13, ar14;
	// EXPCOEF: e = exp_neg_q16(P0 - A) as EXPSUM, then COEF's product
	reg [31:0] ex3;
	reg [63:0] em4;
	reg [31:0] ew4, ew5, ew6, ew7, ew8, ew9;
	reg [8:0] era5, erb5;
	reg [16:0] ek5, ek6, ek7, ek8;
	reg [7:0] er5, er6;
	reg [16:0] eta6, etb6, eta7;
	reg [31:0] em7, ev8, ee9;
	reg signed [31:0] ed10, ed11;
	reg [47:0] em10;
	reg [31:0] ec11, ec12, ec13, ec14;
	always @(posedge clk) begin
		eta6 <= exp2tab[era5];
		etb6 <= exp2tab[erb5];
	end
	always @(posedge clk) if (st != S_AS) begin
		av <= 14'd0; aph <= 2'd0; aji <= 16'd0; ajw <= 16'd0;
	end else begin
		av <= {av[13:1], as_iss};
		aph <= (aph == 2'd2) ? 2'd0 : aph + 2'd1;
		if (as_iss) aji <= aji + 16'd1;
		if (as_wr) ajw <= ajw + 16'd1;
		// cycle 2: A
		at3 <= rq;
		rs3 <= rshr32(rq, s0);
		ex3 <= r_p0 - rq;
		// cycle 3: B, and the first products
		av4 <= at3 * $signed({16'd0, rq[15:0]});
		ak4 <= r_p0 + {{16{rq[31]}}, rq[31:16]};
		em4 <= {32'd0, ex3} * 64'd94548;
		ew4 <= rq;
		rs4 <= rs3 + rshr32(rq, s1);
		rs5 <= rs4; rs6 <= rs5; rs7 <= rs6;
		// ATTSC: |v|, the left shift, the rounding add of the right shift
		aab5 <= uabs64(av4);
		ash5 <= av4 <<< ak4[4:0];
		arn5 <= av4 + (64'sd1 <<< ((-ak4) - 32'sd1));
		av5 <= av4; ak5 <= ak4;
		abl6 <= bitlen64(aab5);
		ashr6 <= ((-ak5) > 32'sd62) ? (av5 < 0 ? -64'sd1 : 64'sd0) : (arn5 >>> (-ak5));
		ash6 <= ash5; av6 <= av5; ak6 <= ak5;
		aov7 <= (ak6 <= 32'sd30) && ({1'b0, abl6} + ak6[6:0] > 7'd62);
		ash7 <= ash6; ashr7 <= ashr6; av7 <= av6; ak7 <= ak6;
		begin : as_rb
			reg signed [63:0] sc;
			sc = (ak7 >= 0) ? ((ak7 > 32'sd30 || aov7) ? (av7 < 0 ? -64'sd1099511627776 : 64'sd1099511627776) : ash7)
			                : ashr7;
			ar8 <= as_rs ? rs7 :
			       (sc > 64'sd1073741824) ? 32'sd1073741824 : (sc < -64'sd1073741824) ? -32'sd1073741824 : sc[31:0];
		end
		ar9 <= ar8; ar10 <= ar9; ar11 <= ar10; ar12 <= ar11; ar13 <= ar12; ar14 <= ar13;
		// EXPCOEF: the table index (k = y >> 16, i, r) from y = (x 94548) >> 16
		era5 <= {1'b0, em4[31:24]};
		erb5 <= {1'b0, em4[31:24]} + 9'd1;
		ek5 <= em4[48:32];
		er5 <= em4[23:16];
		ew5 <= ew4;
		ek6 <= ek5; er6 <= er5; ew6 <= ew5;
		em7 <= {15'd0, eta6 - etb6} * {24'd0, er6};
		eta7 <= eta6; ek7 <= ek6; ew7 <= ew6;
		ev8 <= {15'd0, eta7} - ((em7 + 32'd128) >> 8);
		ek8 <= ek7; ew8 <= ew7;
		ee9 <= (ek8 >= 17'd17) ? 32'd0 : (ek8 == 17'd0) ? ev8 :
		       ((ev8 + (32'd1 << (ek8 - 17'd1))) >> ek8);
		ew9 <= ew8;
		em10 <= {16'd0, ee9} * {32'd0, ew9[15:0]};
		ed10 <= r_p1 - {{16{ew9[31]}}, ew9[31:16]};
		ec11 <= (ed10 > 32'sd63) ? 32'd0 : (em10 >> ed10[5:0]);
		ec12 <= ec11; ec13 <= ec12; ec14 <= ec13;
	end

	// ROPE and KVQ8 streamed (S_P5), on a period of five cycles:
	//   ROPE  pair p: A[ia] read at phase 0 (cycle R), A[ib] at 1, its rt
	//         word B[p] at 2; the two results written at R + 8 (phase 3)
	//         and R + 9 (phase 4)
	//   KVQ8  elements 4w .. 4w + 3 read at phases 0 .. 3, the word D[w]
	//         written at phase 4 of the next period (six cycles after
	//         the last read)
	reg [9:1] pv;                     // ROPE: a pair issued s cycles ago
	reg [6:1] kq;                     // KVQ8: a word's last element read s cycles ago
	reg [2:0] p5;
	reg [15:0] pji, pjw;              // pairs (words) issued, written
	reg [15:0] kji;                   // KVQ8: elements issued
	wire rp_iss = (op == OP_ROPE) && (p5 == 3'd0) && (pji < cnt);
	wire kq_iss = (op == OP_KVQ8) && (p5 != 3'd4) && (kji < cnt);
	wire [15:0] pia = split ? pji : {pji[14:0], 1'b0};
	wire [15:0] pwa = split ? pjw : {pjw[14:0], 1'b0};
	wire [15:0] pwb = split ? pjw + half : {pjw[14:0], 1'b1};
	wire [15:0] pib1 = split ? (pji - 16'd1) + half : {pji[14:0] - 15'd1, 1'b1};   // ib of the pair issued last cycle
	wire rp_wa = pv[8], rp_wb = pv[9];
	wire kq_wr = kq[6];
	reg signed [31:0] pa3, pb4, pa4, pa5, pb5, pc5, ps5;
	reg signed [31:0] pac6, pbs6, pas6, pbc6, pwa7, pwb7;
	reg signed [31:0] ku3;
	reg signed [63:0] kp4;
	reg [7:0] kb5;
	reg [1:0] kl3, kl4, kl5;          // KVQ8: the element's byte lane
	reg [31:0] kw6;
	reg kv3, kv4, kv5;
	always @(posedge clk) if (st != S_P5) begin
		pv <= 9'd0; kq <= 6'd0; p5 <= 3'd0; pji <= 16'd0; pjw <= 16'd0; kji <= 16'd0;
		kv3 <= 1'b0; kv4 <= 1'b0; kv5 <= 1'b0;
	end else begin
		p5 <= (p5 == 3'd4) ? 3'd0 : p5 + 3'd1;
		pv <= {pv[8:1], rp_iss};
		kq <= {kq[5:1], kq_iss && (kji[1:0] == 2'd3)};
		if (rp_iss) pji <= pji + 16'd1;
		if (kq_iss) kji <= kji + 16'd1;
		if (rp_wb || kq_wr) pjw <= pjw + 16'd1;
		// ROPE: a and b (>> 15), then the four products, then the sums
		if (pv[2]) pa3 <= rshr32(rq, 8'sd15);
		if (pv[3]) begin pb4 <= rshr32(rq, 8'sd15); pa4 <= pa3; end
		if (pv[4]) begin
			pa5 <= pa4; pb5 <= pb4;
			pc5 <= {{16{rq[15]}}, rq[15:0]};
			ps5 <= {{16{rq[31]}}, rq[31:16]};
		end
		pac6 <= pa5 * pc5; pbs6 <= pb5 * ps5;
		pas6 <= pa5 * ps5; pbc6 <= pb5 * pc5;
		pwa7 <= pac6 - pbs6;
		pwb7 <= pas6 + pbc6;
		// KVQ8: ((a >> s) r) >> 16 rounded, clamped, into its lane
		kv3 <= kq_iss_d2; kl3 <= kl_d2;
		ku3 <= rshr32(rq, s0);
		kv4 <= kv3; kl4 <= kl3;
		kp4 <= ku3 * r_p1;
		kv5 <= kv4; kl5 <= kl4;
		begin : kq_r
			reg signed [63:0] t;
			t = (kp4 + 64'sd32768) >>> 16;
			kb5 <= (t > 64'sd127) ? 8'd127 : (t < -64'sd127) ? 8'h81 : t[7:0];
		end
		if (kv5) kw6[8*kl5 +: 8] <= kb5;
	end
	// KVQ8: an element's read two cycles back (its data is in rq now)
	reg kq_iss_d1, kq_iss_d2;
	reg [1:0] kl_d1, kl_d2;
	always @(posedge clk) begin
		kq_iss_d1 <= (st == S_P5) && kq_iss; kl_d1 <= kji[1:0];
		kq_iss_d2 <= kq_iss_d1; kl_d2 <= kl_d1;
	end

	// the RAM port: address and write for the current step
	always @(*) begin
		ram_a = aw + i[AW-1:0];
		ram_we = 1'b0;
		ram_wd = 32'd0;
		case (st)
		S_P5: begin
			if (op == OP_ROPE) begin
				if (rp_wa) begin
					ram_we = 1'b1; ram_a = aw + pwa[AW-1:0]; ram_wd = pwa7;
				end else if (rp_wb) begin
					ram_we = 1'b1; ram_a = aw + pwb[AW-1:0]; ram_wd = pwb7;
				end else if (p5 == 3'd0) ram_a = aw + pia[AW-1:0];
				else if (p5 == 3'd1) ram_a = aw + pib1[AW-1:0];
				else ram_a = bw + pji[AW-1:0] - {{(AW-1){1'b0}}, 1'b1};    // rt of the pair issued two cycles ago
			end else begin
				if (kq_wr) begin
					ram_we = 1'b1; ram_a = dw + pjw[AW-1:0]; ram_wd = kw6;
				end else ram_a = aw + kji[AW-1:0];
			end
		end
		S_AS: begin
			if (as_wr) begin
				ram_we = 1'b1;
				ram_a = aw + ajw[AW-1:0];
				ram_wd = as_e ? ec14 : ar14;
			end else if (as_iss) ram_a = aw + aji[AW-1:0];
			else ram_a = bw + aji[AW-1:0] - {{(AW-1){1'b0}}, 1'b1};   // B of the element issued last cycle
		end
		S_FS: begin
			if (fs_wr) begin
				ram_we = 1'b1;
				ram_a = dw + fjw[AW-1:0];
				ram_wd = fs_sq ? fr8 : fr7;
			end else if (fs_iss) ram_a = aw + fji[AW-1:0];
			else ram_a = bw + {1'b0, fji[AW-1:1] - {{(AW-2){1'b0}}, ~fji[0]}};   // SQMUL: B[(fji - 1) / 2]
		end
		default: ;
		endcase
	end

	always @(posedge clk) begin
		bus_ack <= 1'b0;
		if (rst) begin
			st <= S_IDLE;
		end else case (st)
		S_IDLE: if (bus_req && !bus_ack) begin
			if (bus_we) begin
				case (bus_adr)
				4'h0: begin r_a <= bus_wdat; bus_ack <= 1'b1; end
				4'h1: begin r_b <= bus_wdat; bus_ack <= 1'b1; end
				4'h2: begin r_d <= bus_wdat; bus_ack <= 1'b1; end
				4'h3: begin r_n <= bus_wdat; bus_ack <= 1'b1; end
				4'h4: begin
					r_p0 <= bus_wdat;
					s0 <= ($signed(bus_wdat) > 32'sd40) ? 8'sd40 : ($signed(bus_wdat) < -32'sd40) ? -8'sd40 : bus_wdat[7:0];
					bus_ack <= 1'b1;
				end
				4'h5: begin
					r_p1 <= bus_wdat;
					s1 <= ($signed(bus_wdat) > 32'sd40) ? 8'sd40 : ($signed(bus_wdat) < -32'sd40) ? -8'sd40 : bus_wdat[7:0];
					bus_ack <= 1'b1;
				end
				4'h6: begin
					// run; the ack comes when it is done
					op <= bus_wdat[3:0];
					i <= 16'd0;
					mx <= 32'd0;
					sum <= 64'd0;
					cnt <= (bus_wdat[3:0] == OP_ROPE) ? r_n[16:1] : r_n[15:0];
					mv <= 4'd0;
					smx <= -32'sd2147483647;
					// (7, 11-14 are no longer operations: done at once)
					st <= (bus_wdat[3:0] == OP_MAXABS) ? S_MX :
					      (bus_wdat[3:0] == OP_ATTSC || bus_wdat[3:0] == OP_EXPCOEF ||
					       bus_wdat[3:0] == OP_RESID) ? S_AS :
					      (bus_wdat[3:0] == OP_ROPE || bus_wdat[3:0] == OP_KVQ8) ? S_P5 :
					      (bus_wdat[3:0] == OP_SCALE15 || bus_wdat[3:0] == OP_RSHR ||
					       bus_wdat[3:0] == OP_MULSHR30 || bus_wdat[3:0] == OP_SQMUL) ? S_FS : S_DONE;
				end
				default: bus_ack <= 1'b1;
				endcase
			end else begin
				bus_ack <= 1'b1;
				case (bus_adr)
				4'h7: bus_rdat <= (op == OP_ATTSC) ? smx : mx;
				4'h8: bus_rdat <= sum[31:0];
				4'h9: bus_rdat <= sum[63:32];
				4'ha: bus_rdat <= 32'h56553031;       // "VU01"
				default: bus_rdat <= 32'd0;
				endcase
			end
		end

		// MAXABS: a read every cycle; its data (registered) two cycles later,
		// its absolute value three, compared four
		S_MX: begin
			mv <= {mv[2:0], (i < cnt)};
			if (mv[1]) ma0 <= uabs(rq);
			if (mv[2] && ma0 > mx) mx <= ma0;
			if (i < cnt) i <= i + 16'd1;
			else if (mv[1:0] == 2'b00) st <= S_DONE;   // the last compare is this cycle's
		end

		S_P5: if (pjw == ((op == OP_KVQ8) ? {2'b00, cnt[15:2]} : cnt)) st <= S_DONE;
		S_AS: begin
			// the maxima and the sum, from the pipeline above
			if (av[8] && !as_e && !as_rs && ar8 > smx) smx <= ar8;
			if (av[8] && as_rs && uabs(ar8) > mx) mx <= uabs(ar8);
			if (av[9] && as_e) sum <= sum + {32'd0, ee9};
			if (av[12] && as_e && ec12 > mx) mx <= ec12;
			if (ajw == cnt) st <= S_DONE;
		end
		S_FS: begin
			if (fv[3]) begin ma <= fu3; mb <= fs_sq ? fu3 : r_p0; end
			if (fv[4] && fs_sq) begin ma <= fu4; mb <= fg4; end
			if (fv[5] && fs_sq) sum <= sum + {32'd0, mp[31:0]};
			if (fjw == cnt || (fs_wr && fjw + 16'd1 == cnt)) st <= S_DONE;
		end
		S_DONE: begin
			bus_ack <= 1'b1;
			st <= S_IDLE;
		end
		default: st <= S_IDLE;
		endcase
	end

endmodule
`default_nettype wire

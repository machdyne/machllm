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
 *   0x1c R0                  max |.| of MAXABS / RESID
 *   0x20 SUMLO, 0x24 SUMHI   SQMUL's sum of squares
 *   0x28 ID                  "VU01"
 *
 * Operations (MLM_VU_* in fw/common/mlm_model.h):
 *   1 MAXABS   R0 = max |A[i]|                         (1 cycle a element)
 *   2 SQMUL    u = A[i] >> P0, D[i] = u g[i] (g: int16 at B), SUM += u^2
 *   3 SCALE15  D[i] = (A[i] >> 15) P0
 *   4 RSHR     D[i] = A[i] >> P0 (P0 < 0: left)
 *   5 RESID    A[i] = (A[i] >> P0) + (B[i] >> P1); R0 = max |A[i]|
 *   6 MULSHR30 D[i] = (A[i] P0) >> 30 (64-bit product)
 *   7 QPACK    D[j] = int16 pair (A[2j], A[2j+1]) >> P0, clamped,
 *              zero from N on; P1/2 words
 *   8 KVQ8     int8 ((A[i] >> P0) P1) >> 16, clamped, four per D word
 *   9 ROPE     rotate pairs of A (P0 != 0: (i, i + N/2), else (2i, 2i+1))
 *              by B word i = {sin, cos} Q15
 *  10 ATTSC    A[t] = sat30(A[t] x mant(B[t]) << (P0 + exp(B[t]))), R0 = max
 *  11 EXPSUM   A[t] = exp(A[t] - P0) Q16 (table + interpolation), SUM
 *  12 COEF     A[t] = (A[t] x mant(B[t])) >> (P0 - exp(B[t])), R0 = max
 *  13 QPACKU   QPACK for unsigned values (64-bit rounding shift)
 * ">>" is rshr32: rounded half up, 0 beyond 31; products wrap to 32
 * bits except MULSHR30's and KVQ8's.
 *
 * Each element: read(s), shifts, up to four products through one
 * registered multiplier, write(s): 3-14 cycles an element (the next
 * element's reads overlapped); SCALE15, RSHR, MULSHR30 and SQMUL
 * streamed instead (S_FS, 2-3 cycles), against ~20-30 for the CPU.
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
	           OP_RESID = 4'd5, OP_MULSHR30 = 4'd6, OP_QPACK = 4'd7, OP_KVQ8 = 4'd8,
	           OP_ROPE = 4'd9, OP_ATTSC = 4'd10, OP_EXPSUM = 4'd11, OP_COEF = 4'd12,
	           OP_QPACKU = 4'd13, OP_MXPACK = 4'd14;

	// 2^(-i/256) in Q16 for EXPSUM (fw/common/mlm_exp2tab.h)
	reg [16:0] exp2tab [0:256];
`include "mlm_exp2tab.vh"
	reg [8:0] rom_a;
	reg [16:0] rom_q;
	always @(posedge clk) rom_q <= exp2tab[rom_a];

	function signed [31:0] rshr32;
		input signed [31:0] x; input signed [7:0] sh;
		rshr32 = (sh <= 8'sd0) ? (x <<< (-sh)) : (sh > 8'sd31) ? 32'sd0 :
		         (((x >>> (sh - 8'sd1)) + 32'sd1) >>> 1);
	endfunction
	function [63:0] uabs64;
		input [63:0] x;
		uabs64 = x[63] ? (~x + 64'd1) : x;
	endfunction
	// rshr64 of the model code for any s (s <= 0 left)
	function signed [63:0] rshr64;
		input signed [63:0] v; input signed [31:0] s;
		rshr64 = (s <= 0) ? ((s < -62) ? (v <<< 62) : (v <<< (-s))) :
		         (s > 62) ? (v < 0 ? -64'sd1 : 64'sd0) :
		         ((v + (64'sd1 <<< (s - 1))) >>> s);
	endfunction
	function [63:0] rshr64u;           // the same for a non-negative 64-bit value, 8-bit s
		input [63:0] v; input signed [7:0] s;
		rshr64u = (s <= 8'sd0) ? (v << (-s)) : (s > 8'sd62) ? 64'd0 :
		          ((v + (64'd1 << (s - 8'sd1))) >> s);
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

	localparam S_IDLE = 4'd0, S_MX = 4'd1, S_MXL = 4'd2, S_NEXT = 4'd3, S_RD = 4'd4,
	           S_CMP = 4'd5, S_WR = 4'd6, S_DONE = 4'd7,
	           S_X1 = 4'd8, S_XB = 4'd9, S_XR = 4'd10, S_XW = 4'd11,   // MXPACK
	           S_FS = 4'd12;                                         // streamed ops
	reg [3:0] st;
	reg [2:0] rk;                     // read step: reads at 0.., their data two later
	reg [31:0] rq;                    // the RAM's output, registered (a long path otherwise)
	always @(posedge clk) rq <= ram_q;
	reg [3:0] k;                      // compute step
	reg wk;                           // write step
	assign ram_own = (st != S_IDLE);

	reg [15:0] i, cnt;                // element (pair, word) index; how many
	wire [AW-1:0] aw = r_a[AW+1:2], bw = r_b[AW+1:2], dw = r_d[AW+1:2];
	wire split = (r_p0 != 32'sd0);
	wire [15:0] half = r_n[16:1];
	wire pk = (op == OP_QPACK || op == OP_QPACKU);
	wire [15:0] ia = (op == OP_ROPE) ? (split ? i : {i[14:0], 1'b0}) :
	                 pk ? {i[14:0], 1'b0} : i;
	wire [15:0] ib = (op == OP_ROPE) ? (split ? i + half : {i[14:0], 1'b1}) :
	                 pk ? {i[14:0], 1'b1} : i;
	wire in_a = !pk || (ia < r_n[15:0]);
	wire in_b = !pk || (ib < r_n[15:0]);
	// written back into A (else D)
	wire in_place = (op == OP_ROPE || op == OP_RESID || op == OP_ATTSC || op == OP_EXPSUM || op == OP_COEF);
	// the second read is B[i] (else A[ib]); SQMUL's is B[i/2]
	wire rd_bi = (op == OP_RESID || op == OP_ATTSC || op == OP_COEF);
	// reads per element: ROPE 3 (A[ia], A[ib], rt), SQMUL/RESID/QPACK 2, else 1
	wire [1:0] nrd = (op == OP_ROPE) ? 2'd3 :
	                 (op == OP_SQMUL || rd_bi || pk) ? 2'd2 : 2'd1;
	// the compute step after which the results are ready
	reg [3:0] kdone;
	always @(*) case (op)
		OP_ROPE: kdone = 4'd8;
		OP_SQMUL: kdone = 4'd6;
		OP_SCALE15, OP_MULSHR30, OP_KVQ8: kdone = 4'd5;
		OP_ATTSC: kdone = 4'd9;
		OP_EXPSUM: kdone = 4'd14;
		OP_COEF: kdone = 4'd6;
		OP_QPACKU: kdone = 4'd4;
		default: kdone = 4'd3;
	endcase

	reg signed [31:0] xa, xb, xc, ta, tb;
	reg signed [33:0] ma, mb;         // 34 bits: EXPSUM's x reaches 2^31
	reg signed [67:0] mp;
	reg signed [63:0] p0, p1, p2, p3;
	reg signed [31:0] r1, r2;
	reg [31:0] wacc;
	reg [3:0] mv;                     // MAXABS: which pipeline steps hold an element
	reg signed [31:0] smx;            // ATTSC's (signed) maximum
	reg signed [31:0] ke;             // ATTSC: the shift; COEF: d
	reg signed [63:0] v64, shl_r, shr_r, ua64, ub64;
	reg ovf_r;
	reg [63:0] vabs;
	reg signed [63:0] rnd_r;
	reg [6:0] vbl;
	reg signed [31:0] r1q, r2q;     // the results, registered
	reg [31:0] ex_x;
	reg [31:0] ex_y;
	reg [16:0] ex_a, ex_b;
	reg [31:0] ex_v;
	reg [31:0] ma0;

	// one product a cycle, registered: operands set at step k appear in
	// mp at step k + 2
	always @(posedge clk) mp <= ma * mb;

	wire signed [31:0] rt_c = {{16{xc[15]}}, xc[15:0]};
	wire signed [31:0] rt_s = {{16{xc[31]}}, xc[31:16]};
	wire signed [31:0] g16 = i[0] ? {{16{xb[31]}}, xb[31:16]} : {{16{xb[15]}}, xb[15:0]};
	wire signed [63:0] p0_r16 = (p0 + 64'sd32768) >>> 16;
	wire signed [63:0] p0_r30 = (p0 + 64'sd536870912) >>> 30;
	wire signed [31:0] kv_t = (p0_r16 > 64'sd127) ? 32'sd127 : (p0_r16 < -64'sd127) ? -32'sd127 : p0_r16[31:0];
	wire signed [31:0] qa = (ta > 32'sd32767) ? 32'sd32767 : (ta < -32'sd32767) ? -32'sd32767 : ta;
	wire signed [31:0] qb = (tb > 32'sd32767) ? 32'sd32767 : (tb < -32'sd32767) ? -32'sd32767 : tb;
	wire last_word = (i[1:0] == 2'd3) || (i + 16'd1 == cnt);
	// ATTSC's result: saturated beyond k = 30 or on overflow, else the
	// shifted product, clamped to +-2^30
	wire signed [63:0] att_sc = (ke >= 0) ? ((ke > 32'sd30 || ovf_r) ? (v64 < 0 ? -64'sd1099511627776 : 64'sd1099511627776) : shl_r)
	                                      : shr_r;
	wire signed [31:0] att_r = (att_sc > 64'sd1073741824) ? 32'sd1073741824 :
	                           (att_sc < -64'sd1073741824) ? -32'sd1073741824 : att_sc[31:0];
	wire [63:0] coef_sh = p0[63:0] >> ke[5:0];
	// QPACKU: clamped to 32767 as a signed 64-bit value, as the C does
	wire [31:0] qu_a = (!in_a) ? 32'd0 : ($signed(ua64) > 64'sd32767) ? 32'd32767 : ua64[31:0];
	wire [31:0] qu_b = (!in_b) ? 32'd0 : ($signed(ub64) > 64'sd32767) ? 32'd32767 : ub64[31:0];
	wire [31:0] wacc_n = wacc | ({24'd0, r1q[7:0]} << {i[1:0], 3'b000});

	// MXPACK (eng_mx_pack): pass 1 reads A once (a read a cycle) for each
	// block of 32's s_b = max(0, bitlen(max |a|) - 7) and s = max s_b;
	// pass 2 per block writes d = s - sh (sh = max(s - 15, s_b)) to B[b],
	// then int8 clamp(rshr32(a, sh)) four to a D word (7 cycles a word)
	reg [4:0] xsb [0:127];
	reg [4:0] xs;
	reg [3:0] xv;                     // pass 1: which pipeline steps hold an element
	reg [15:0] xi1, xi2;
	reg [31:0] xbm;
	reg xbe;                          // the block just ended (its max is in xbm)
	reg [6:0] xbk, xbn;
	reg [2:0] xw;
	reg [1:0] xj;
	reg [4:0] xsh;
	reg [31:0] wpk;
	reg [2:0] xdr;
	reg xp1, xp2, xin1, xin2;         // pass 2: a read in flight, inside n
	reg [1:0] xj1, xj2;
	wire [5:0] xbl = bl32(xbm);
	wire [4:0] xsbv = (xbl > 6'd7) ? (xbl[4:0] - 5'd7) : 5'd0;
	wire [4:0] xshc = (xs > 5'd15 && xs - 5'd15 > xsb[xbn]) ? (xs - 5'd15) : xsb[xbn];
	wire [15:0] xel = {2'b00, xbn, xw, xj};          // pass 2: the element read now
	wire signed [31:0] xq = rshr32(rq, {3'b000, xsh});
	wire [7:0] xq8 = (!xin2) ? 8'd0 : (xq > 32'sd127) ? 8'd127 : (xq < -32'sd127) ? 8'h81 : xq[7:0];

	// Prefetch: for ops whose reads are A[i], B[i] or B[i/2], the next
	// element's reads go out during this one's compute steps (the port
	// is free then): k = 1 reads A[i+1], k = 2 the second; their data are
	// in rq at k = 3 and k = 4 (RESID, kdone 3: in S_WR). S_WR then goes
	// straight to S_CMP.
	wire pf_op = (op == OP_SQMUL || op == OP_SCALE15 || op == OP_RSHR || op == OP_RESID ||
	              op == OP_MULSHR30 || op == OP_KVQ8 || op == OP_ATTSC || op == OP_EXPSUM ||
	              op == OP_COEF);
	wire [15:0] inx = i + 16'd1;
	wire pf_go = pf_op && (inx < cnt);
	reg signed [31:0] pfa, pfb;

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

	// the RAM port: address and write for the current step
	always @(*) begin
		ram_a = aw + i[AW-1:0];
		ram_we = 1'b0;
		ram_wd = r1q;
		case (st)
		S_RD: case (rk)
			3'd0: ram_a = aw + ia[AW-1:0];
			3'd1: ram_a = (op == OP_SQMUL) ? bw + {1'b0, i[AW-1:1]} :
			              rd_bi ? bw + i[AW-1:0] : aw + ib[AW-1:0];
			default: ram_a = bw + i[AW-1:0];          // ROPE's rt word
			endcase
		S_CMP: if (pf_go) begin
			if (k == 4'd1) ram_a = aw + inx[AW-1:0];
			else if (k == 4'd2 && nrd == 2'd2)
				ram_a = (op == OP_SQMUL) ? bw + {1'b0, inx[AW-1:1]} : bw + inx[AW-1:0];
		end
		S_X1: ram_a = aw + i[AW-1:0];
		S_XB: begin
			ram_we = 1'b1;
			ram_a = bw + {{(AW-7){1'b0}}, xbn};
			ram_wd = {27'd0, xs - xshc};
		end
		S_XR: ram_a = aw + xel[AW-1:0];
		S_FS: begin
			if (fs_wr) begin
				ram_we = 1'b1;
				ram_a = dw + fjw[AW-1:0];
				ram_wd = fs_sq ? fr8 : fr7;
			end else if (fs_iss) ram_a = aw + fji[AW-1:0];
			else ram_a = bw + {1'b0, fji[AW-1:1] - {{(AW-2){1'b0}}, ~fji[0]}};   // SQMUL: B[(fji - 1) / 2]
		end
		S_XW: begin
			ram_we = (xdr == 3'd0);
			ram_a = dw + {{(AW-10){1'b0}}, xbn, xw};
			ram_wd = wpk;
		end
		S_WR: begin
			if (!wk) begin
				ram_we = (op != OP_KVQ8) || last_word;
				ram_a = in_place ? aw + ia[AW-1:0] :
				        (op == OP_KVQ8) ? dw + {2'b0, i[AW-1:2]} : dw + i[AW-1:0];
				ram_wd = (op == OP_KVQ8) ? wacc_n : r1q;
			end else begin
				ram_we = 1'b1;                        // ROPE's second
				ram_a = aw + ib[AW-1:0];
				ram_wd = r2q;
			end
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
					wacc <= 32'd0;
					cnt <= (bus_wdat[3:0] == OP_QPACK || bus_wdat[3:0] == OP_QPACKU) ? r_p1[16:1] :
					       (bus_wdat[3:0] == OP_ROPE) ? r_n[16:1] : r_n[15:0];
					mv <= 4'd0;
					smx <= -32'sd2147483647;
					xv <= 4'd0; xs <= 5'd0; xbe <= 1'b0;
					if (bus_wdat[3:0] == OP_MXPACK) cnt <= r_p1[15:0];
					st <= (bus_wdat[3:0] == OP_MAXABS) ? S_MX :
					      (bus_wdat[3:0] == OP_MXPACK) ? S_X1 :
					      (bus_wdat[3:0] == OP_SCALE15 || bus_wdat[3:0] == OP_RSHR ||
					       bus_wdat[3:0] == OP_MULSHR30 || bus_wdat[3:0] == OP_SQMUL) ? S_FS : S_NEXT;
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

		S_NEXT: begin
			rk <= 3'd0;
			st <= (i == cnt) ? S_DONE : S_RD;
		end

		// MXPACK pass 1: a read a cycle; its data in rq two later
		S_X1: begin
			xv <= {xv[2:0], (i < cnt)};
			xi1 <= i; xi2 <= xi1;
			if (i < cnt) i <= i + 16'd1;
			// the element in rq: the block's running max
			if (xv[1]) begin
				if (xi2[4:0] == 5'd0 || ((xi2 < r_n[15:0]) ? uabs(rq) : 32'd0) > xbm)
					xbm <= (xi2 < r_n[15:0]) ? uabs(rq) : 32'd0;
			end
			xbe <= xv[1] && (xi2[4:0] == 5'd31);
			xbk <= xi2[11:5];
			// a block ended last cycle: its s_b
			if (xbe) begin
				xsb[xbk] <= xsbv;
				if (xsbv > xs) xs <= xsbv;
			end
			if (i == cnt && xv[2:0] == 3'd0 && !xbe) begin
				xbn <= 7'd0;
				if (cnt == 16'd0) begin mx <= 32'd0; st <= S_DONE; end
				else st <= S_XB;
			end
		end
		// pass 2: the block's shift (and d written to B[b] this cycle)
		S_XB: begin
			xsh <= xshc;
			xw <= 3'd0;
			xj <= 2'd0;
			st <= S_XR;
		end
		// four reads; their int8 into wpk as they arrive
		S_XR: begin
			xj <= xj + 2'd1;
			if (xj == 2'd3) begin
				xdr <= 3'd2;
				st <= S_XW;
			end
		end
		S_XW: begin
			if (xdr != 3'd0) xdr <= xdr - 3'd1;
			else begin
				// the word is written this cycle
				if (xw == 3'd7) begin
					if ({xbn, 5'd0} + 12'd32 >= cnt[11:0]) begin
						mx <= {27'd0, xs};
						st <= S_DONE;
					end else begin
						xbn <= xbn + 7'd1;
						st <= S_XB;
					end
				end else begin
					xw <= xw + 3'd1;
					xj <= 2'd0;
					st <= S_XR;
				end
			end
		end
		// reads at rk = 0 .. nrd-1; the data of the read at rk is in rq at
		// rk + 2
		S_RD: begin
			if (rk == 3'd2) xa <= in_a ? rq : 32'sd0;
			if (rk == 3'd3) xb <= (nrd == 2'd2 && !in_b) ? 32'sd0 : rq;
			if (rk == 3'd4) xc <= rq;
			if (rk == {1'b0, nrd} + 3'd1) begin
				st <= S_CMP;
				k <= 4'd1;
			end else rk <= rk + 3'd1;
		end
		S_CMP: begin
			k <= k + 4'd1;
			if (pf_go && k == 4'd3) pfa <= rq;
			if (pf_go && k == 4'd4) pfb <= rq;
			case (k)
			4'd1: begin
				case (op)
				OP_ATTSC: ke <= r_p0 + {{16{xb[31]}}, xb[31:16]};
				OP_EXPSUM: ex_x <= r_p0 - xa;
				OP_COEF: ke <= r_p0 - {{16{xb[31]}}, xb[31:16]};
				OP_QPACKU: begin
					ua64 <= rshr64u({32'd0, xa}, s0);
					ub64 <= rshr64u({32'd0, xb}, s0);
				end
				OP_SCALE15, OP_ROPE: begin ta <= rshr32(xa, 8'sd15); tb <= rshr32(xb, 8'sd15); end
				OP_MULSHR30: ta <= xa;
				OP_RESID: begin ta <= rshr32(xa, s0); tb <= rshr32(xb, s1); end
				default: begin ta <= rshr32(xa, s0); tb <= rshr32(xb, s0); end
				endcase
			end
			// products: operands at k = 2.., results at k = 4..
			4'd2: case (op)
				OP_ATTSC: begin ma <= xa; mb <= {18'd0, xb[15:0]}; end
				OP_EXPSUM: begin ma <= {2'b00, ex_x}; mb <= 34'sd94548; end
				OP_COEF: begin ma <= {2'b00, xa}; mb <= {18'd0, xb[15:0]}; end
				OP_SQMUL: begin ma <= ta; mb <= ta; end
				OP_SCALE15, OP_MULSHR30: begin ma <= ta; mb <= r_p0; end
				OP_KVQ8: begin ma <= ta; mb <= r_p1; end
				OP_ROPE: begin ma <= ta; mb <= rt_c; end          // a c
				default: ;
				endcase
			4'd3: case (op)
				OP_SQMUL: begin ma <= ta; mb <= g16; end
				OP_ROPE: begin ma <= tb; mb <= rt_s; end          // b s
				default: ;
				endcase
			4'd4: begin
				p0 <= mp[63:0];
				v64 <= mp[63:0];
				ex_y <= mp[47:16];                     // (x 94548) >> 16
				if (op == OP_ROPE) begin ma <= ta; mb <= rt_s; end   // a s
			end
			4'd5: begin
				p1 <= mp[63:0];
				// ATTSC: |v|, the left shift, the rounding add of the right shift
				vabs <= uabs64(v64);
				shl_r <= v64 <<< ke[4:0];
				rnd_r <= v64 + (64'sd1 <<< ((-ke) - 32'sd1));
				// EXPSUM: the table index
				rom_a <= {1'b0, ex_y[15:8]};
				if (op == OP_ROPE) begin ma <= tb; mb <= rt_c; end   // b c
			end
			4'd6: begin
				p2 <= mp[63:0];
				vbl <= bitlen64(vabs);
				shr_r <= ((-ke) > 32'sd62) ? (v64 < 0 ? -64'sd1 : 64'sd0) : (rnd_r >>> (-ke));
				rom_a <= {1'b0, ex_y[15:8]} + 9'd1;
			end
			4'd7: begin
				p3 <= mp[63:0];
				ex_a <= rom_q;
			end
			4'd8: begin
				ex_b <= rom_q;
				// ATTSC: overflow when bitlen |v| > 62 - k
				ovf_r <= (ke <= 32'sd30) && ({1'b0, vbl} + ke[6:0] > 7'd62);
			end
			4'd9: begin ma <= {17'd0, ex_a - ex_b}; mb <= {26'd0, ex_y[7:0]}; end
			4'd11: ex_v <= {15'd0, ex_a} - ((mp[31:0] + 32'd128) >> 8);
			4'd12: ex_v <= (ex_y[31:16] >= 16'd17) ? 32'd0 :
			               (ex_y[31:16] == 16'd0) ? ex_v :
			               ((ex_v + (32'd1 << (ex_y[31:16] - 16'd1))) >> ex_y[31:16]);
			default: ;
			endcase
			// done once the last product is in (k = 3 + npr), or at k = 2
			// without products; then the results
			if (k == kdone) begin
				r1q <= r1;
				r2q <= r2;
				st <= S_WR;
				wk <= 1'b0;
			end
		end
		S_WR: begin
			if (!wk) begin
				// the accumulations that need this element's result
				if (op == OP_SQMUL) sum <= sum + {32'd0, p0[31:0]};
				if (op == OP_RESID && uabs(r1q) > mx) mx <= uabs(r1q);
				if (op == OP_ATTSC && r1q > smx) smx <= r1q;
				if (op == OP_EXPSUM) sum <= sum + {32'd0, r1q};
				if (op == OP_COEF && r1q > mx) mx <= r1q;
				if (op == OP_KVQ8) wacc <= last_word ? 32'd0 : wacc_n;
			end
			if (!wk && op == OP_ROPE) wk <= 1'b1;
			else if (pf_go) begin
				// the next element's operands are here (RESID's second
				// arrives this cycle)
				i <= inx;
				xa <= pfa;
				xb <= (nrd == 2'd2) ? ((kdone == 4'd3) ? rq : pfb) : xb;
				k <= 4'd1;
				st <= S_CMP;
			end else begin
				i <= i + 16'd1;
				st <= S_NEXT;
			end
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

	// MXPACK pass 2: reads issued in S_XR, their bytes two cycles later
	always @(posedge clk) begin
		xp1 <= (st == S_XR);
		xj1 <= xj;
		xin1 <= (xel < r_n[15:0]);
		xp2 <= xp1; xj2 <= xj1; xin2 <= xin1;
		if (xp2) wpk[8*xj2 +: 8] <= xq8;
	end

	// results, from the products and shifts above (valid in S_WR)
	always @(*) begin
		r1 = 32'sd0;
		r2 = 32'sd0;
		case (op)
		OP_SQMUL: r1 = p1[31:0];
		OP_SCALE15: r1 = p0[31:0];
		OP_RSHR: r1 = ta;
		OP_RESID: r1 = ta + tb;
		OP_MULSHR30: r1 = p0_r30[31:0];
		OP_QPACK: r1 = {qb[15:0], qa[15:0]};
		OP_ROPE: begin r1 = p0[31:0] - p1[31:0]; r2 = p2[31:0] + p3[31:0]; end
		OP_ATTSC: r1 = att_r;
		OP_KVQ8: r1 = kv_t;
		OP_EXPSUM: r1 = ex_v;
		OP_COEF: r1 = (ke > 32'sd63) ? 32'd0 : coef_sh[31:0];
		OP_QPACKU: r1 = qu_a | (qu_b << 16);
		default: ;
		endcase
	end

endmodule

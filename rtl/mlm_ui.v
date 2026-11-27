/*
 * machllm
 * Copyright (c) 2026 Lone Dynamics Corporation. All rights reserved.
 *
 * mlm_ui: the screen and the keyboard (optional: `make UI=1`, the
 * default for the two-clock builds).
 *
 * Screen: 640x480 at 60 Hz over DVI (the DDMI connector), 80 x 25
 * characters of 8 x 16 pixels (400 lines, centred: 40 blank lines
 * above and below). A cell is 9 bits, one block RAM for all 2,000:
 * {colour[1:0], char[6:0]}; characters 32..127 come from the 8x16 font
 * (rtl/mlm_font8x16.hex, in logic, not block RAM; 127 is a solid block,
 * the cursor). Colours: 0 text, 1 the user's words, 2 accent, 3 dim.
 * A row can have the bar background instead of the page's (ROWBG).
 *
 * Keyboard: nand2mario's usb_hid_host (Zeitlos's rtl/ext/usb_hid_host;
 * a low-speed keyboard on the first USB host port, no hub) on 12 MHz.
 * Each report's modifier byte and three keys are handed to the CPU's
 * clock with a toggle; the firmware turns key codes into characters.
 *
 * CPU window (clk; answers the next cycle, as the RAM):
 *   0x0000-0x1fff  cell i at 4i: {colour[8:7], char[6:0]}, read/write
 *   0x2000  ID "UI01"
 *   0x2004  ROWBG: bit r = row r on the bar background
 *   0x2008  KEYS: {key3, key2, key1, modifiers}, the last report
 *   0x200c  KSTAT: {report count[15:0], 13'b0, error, type[1:0]}
 *           (type 1: a keyboard)
 *
 * Video pipeline (pclk): the pixel counters, then the cell read (block
 * RAM), the glyph row (logic), the pixel and its colour: four stages;
 * the syncs and the blanking are delayed by as many.
 */
`default_nettype none
`ifndef MLM_FONT_FILE
`define MLM_FONT_FILE "rtl/mlm_font8x16.hex"
`endif
module mlm_ui #(
	parameter FONT_FILE = `MLM_FONT_FILE
) (
	input wire clk,                 // the CPU's
	input wire rst,
	input wire [13:0] adr,          // byte address in the window
	input wire [31:0] wdat,
	input wire we,
	input wire stb,                 // an access this cycle
	output reg [31:0] rdat,
	output reg ack,

	input wire pclk,                // 25.2 MHz
	input wire bclk,                // 126 MHz, 5 x pclk
	output wire [1:0] tmds_clk,     // two bits a bclk, for ODDRX1F (D0 first)
	output wire [1:0] tmds_r,
	output wire [1:0] tmds_g,
	output wire [1:0] tmds_b,

	input wire clk12,
	inout wire usb_dp,
	inout wire usb_dm
);
	// ---------------------------------------------------------------
	// cells: port A the CPU's (clk), port B the video's (pclk)
	reg [8:0] cells [0:2047];
	reg [8:0] cq;
	wire c_acc = stb && !adr[13];
	always @(posedge clk) begin
		if (c_acc && we) cells[adr[12:2]] <= wdat[8:0];
		cq <= cells[adr[12:2]];
	end

	reg [24:0] rowbg;
	reg [31:0] keys_c, kstat_c;
	reg r_cell;
	reg [1:0] r_sel;
	always @(posedge clk) begin
		ack <= stb && !rst;
		r_cell <= !adr[13];
		r_sel <= adr[3:2];
		if (rst) rowbg <= 25'd0;
		else if (stb && we && adr[13] && adr[3:2] == 2'd1) rowbg <= wdat[24:0];
	end
	always @(*) begin
		if (r_cell) rdat = {23'd0, cq};
		else case (r_sel)
			2'd0: rdat = 32'h55493031;          // "UI01"
			2'd1: rdat = {7'd0, rowbg};
			2'd2: rdat = keys_c;
			default: rdat = kstat_c;
		endcase
	end

	// ---------------------------------------------------------------
	// video timing: 640x480, h 800 (640 16 96 48), v 525 (480 10 2 33),
	// both syncs negative
	reg [9:0] hc = 0, vc = 0;
	always @(posedge pclk) begin
		if (hc == 10'd799) begin
			hc <= 0;
			vc <= (vc == 10'd524) ? 10'd0 : vc + 10'd1;
		end else hc <= hc + 10'd1;
	end
	reg [24:0] rowbg_m, rowbg_p;            // ROWBG into pclk (it changes rarely)
	always @(posedge pclk) begin rowbg_m <= rowbg; rowbg_p <= rowbg_m; end
	// stage 1: which cell, which glyph row
	wire [9:0] ty = vc - 10'd40;
	wire in_text = (vc >= 10'd40) && (vc < 10'd440) && (hc < 10'd640);
	wire [4:0] trow = ty[8:4];
	reg [10:0] c_adr;
	reg [3:0] s1_line;
	reg [2:0] s1_x;
	reg [4:0] s1_row;
	reg s1_text;
	always @(posedge pclk) begin
		c_adr <= {trow, 6'd0} + {2'd0, trow, 4'd0} + {4'd0, hc[9:3]};   // row x 80 + column
		s1_line <= ty[3:0];
		s1_x <= hc[2:0];
		s1_row <= trow;
		s1_text <= in_text;
	end
	// stage 2: the cell
	reg [8:0] vq;
	reg [3:0] s2_line;
	reg [2:0] s2_x;
	reg [4:0] s2_row;
	reg s2_text;
	always @(posedge pclk) begin
		vq <= cells[c_adr];
		s2_line <= s1_line; s2_x <= s1_x; s2_row <= s1_row; s2_text <= s1_text;
	end
	// stage 3: the glyph's row (logic ROM: 96 x 16 bytes)
	(* rom_style = "logic" *) reg [7:0] font [0:1535];
	initial $readmemh(FONT_FILE, font);
	wire [6:0] ch = vq[6:0];
	wire printable = ch >= 7'd32;
	wire [6:0] gi = ch - 7'd32;
	reg [7:0] s3_bits;
	reg [1:0] s3_col;
	reg [2:0] s3_x;
	reg s3_text, s3_bar;
	always @(posedge pclk) begin
		s3_bits <= printable ? font[{gi, s2_line}] : 8'd0;
		s3_col <= vq[8:7];
		s3_x <= s2_x;
		s3_text <= s2_text;
		s3_bar <= s2_text && rowbg_p[s2_row];
	end
	// stage 4: the pixel's colour
	reg [7:0] pr, pg, pb;
	always @(posedge pclk) begin
		if (s3_text && s3_bits[3'd7 - s3_x]) begin
			case (s3_col)
				2'd0: {pr, pg, pb} <= 24'he8e6df;    // text
				2'd1: {pr, pg, pb} <= 24'h7fc8f8;    // the user's words
				2'd2: {pr, pg, pb} <= 24'hf2b84b;    // accent
				default: {pr, pg, pb} <= 24'h8a8f98; // dim
			endcase
		end else if (s3_bar) {pr, pg, pb} <= 24'h1c222b;
		else {pr, pg, pb} <= 24'h0e1116;
	end
	// syncs and blanking, delayed four stages to meet the pixels
	wire hs0 = !(hc >= 10'd656 && hc < 10'd752);
	wire vs0 = !(vc >= 10'd490 && vc < 10'd492);
	wire de0 = (hc < 10'd640) && (vc < 10'd480);
	reg [3:0] hs_d, vs_d, de_d;
	always @(posedge pclk) begin
		hs_d <= {hs_d[2:0], hs0};
		vs_d <= {vs_d[2:0], vs0};
		de_d <= {de_d[2:0], de0};
	end

	// DVI: TMDS per channel (blue carries the syncs), then two bits a bclk
	wire [9:0] t_r, t_g, t_b;
	mlm_tmds enc_r (.clk(pclk), .de(de_d[3]), .d(pr), .c(2'b00), .q(t_r));
	mlm_tmds enc_g (.clk(pclk), .de(de_d[3]), .d(pg), .c(2'b00), .q(t_g));
	mlm_tmds enc_b (.clk(pclk), .de(de_d[3]), .d(pb), .c({vs_d[3], hs_d[3]}), .q(t_b));
	reg [2:0] sm = 0;
	reg load = 0;
	reg [9:0] sr_r = 0, sr_g = 0, sr_b = 0, sr_c = 0;
	always @(posedge bclk) begin
		sm <= (sm == 3'd4) ? 3'd0 : sm + 3'd1;
		load <= (sm == 3'd4);
		if (load) begin
			sr_r <= t_r; sr_g <= t_g; sr_b <= t_b; sr_c <= 10'b0000011111;
		end else begin
			sr_r <= {2'b00, sr_r[9:2]}; sr_g <= {2'b00, sr_g[9:2]};
			sr_b <= {2'b00, sr_b[9:2]}; sr_c <= {2'b00, sr_c[9:2]};
		end
	end
	assign tmds_r = sr_r[1:0];
	assign tmds_g = sr_g[1:0];
	assign tmds_b = sr_b[1:0];
	assign tmds_clk = sr_c[1:0];

	// ---------------------------------------------------------------
	// keyboard
	reg [2:0] ur = 3'b111;                   // reset into clk12
	always @(posedge clk12) ur <= {ur[1:0], rst};
	wire [1:0] typ;
	wire report, conerr;
	wire [7:0] kmod, k1, k2, k3, k4;
	usb_hid_host hid (
		.usbclk(clk12), .usbrst_n(!ur[2]),
		.usb_dm(usb_dm), .usb_dp(usb_dp),
		.typ(typ), .report(report), .conerr(conerr),
		.key_modifiers(kmod), .key1(k1), .key2(k2), .key3(k3), .key4(k4),
		.mouse_btn(), .mouse_dx(), .mouse_dy(),
		.game_l(), .game_r(), .game_u(), .game_d(),
		.game_a(), .game_b(), .game_x(), .game_y(), .game_sel(), .game_sta(),
		.dbg_hid_report()
	);
	// a report: hold it and toggle; the CPU's side takes it when the
	// toggle arrives (reports are milliseconds apart)
	reg [31:0] k_hold = 0;
	reg k_tgl = 0;
	reg [2:0] k_stat12 = 0;
	always @(posedge clk12) begin
		k_stat12 <= {conerr, typ};
		if (report && typ == 2'd1) begin
			k_hold <= {k3, k2, k1, kmod};
			k_tgl <= ~k_tgl;
		end
	end
	reg [2:0] k_sy;
	reg [2:0] st_m, st_c;
	reg [15:0] k_cnt;
	always @(posedge clk) begin
		k_sy <= {k_sy[1:0], k_tgl};
		st_m <= k_stat12; st_c <= st_m;
		if (rst) begin keys_c <= 32'd0; k_cnt <= 16'd0; end
		else if (k_sy[2] != k_sy[1]) begin keys_c <= k_hold; k_cnt <= k_cnt + 16'd1; end
		kstat_c <= {k_cnt, 13'd0, st_c};
	end
endmodule

/* a DVI TMDS encoder (8b/10b with DC balance; control symbols when
 * de is low) */
module mlm_tmds (
	input wire clk,
	input wire de,
	input wire [7:0] d,
	input wire [1:0] c,
	output reg [9:0] q
);
	function [3:0] ones8(input [7:0] v);
		ones8 = v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7];
	endfunction
	wire [3:0] n1d = ones8(d);
	wire xnor_ = (n1d > 4'd4) || (n1d == 4'd4 && !d[0]);
	wire [8:0] qm;
	assign qm[0] = d[0];
	genvar i;
	generate for (i = 1; i < 8; i = i + 1) begin : g
		assign qm[i] = xnor_ ? ~(qm[i-1] ^ d[i]) : (qm[i-1] ^ d[i]);
	end endgenerate
	assign qm[8] = !xnor_;
	wire [3:0] n1 = ones8(qm[7:0]);
	wire signed [4:0] diff = $signed({1'b0, n1}) - $signed(5'd4);    // (ones - zeros) / 2
	reg signed [4:0] cnt = 0;                                        // running disparity / 2
	always @(posedge clk) begin
		if (!de) begin
			cnt <= 0;
			case (c)
				2'b00: q <= 10'b1101010100;
				2'b01: q <= 10'b0010101011;
				2'b10: q <= 10'b0101010100;
				default: q <= 10'b1010101011;
			endcase
		end else if (cnt == 0 || n1 == 4'd4) begin
			q <= {~qm[8], qm[8], qm[8] ? qm[7:0] : ~qm[7:0]};
			cnt <= qm[8] ? cnt + diff : cnt - diff;
		end else if ((cnt > 0 && n1 > 4'd4) || (cnt < 0 && n1 < 4'd4)) begin
			q <= {1'b1, qm[8], ~qm[7:0]};
			cnt <= cnt + (qm[8] ? 5'sd1 : 5'sd0) - diff;
		end else begin
			q <= {1'b0, qm[8], qm[7:0]};
			cnt <= cnt - (qm[8] ? 5'sd0 : 5'sd1) + diff;
		end
	end
endmodule
`default_nettype wire

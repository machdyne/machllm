/*
 * machllm
 * Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
 *
 * mlm_dma: moves data between main memory (DDR3) and the CPU's RAM on
 * the memory clock, through the RAM's second port (rtl/mlm_soc.v), so
 * the CPU does not copy word by word across the clock crossing.
 *
 * Registers (CPU window 0x8100_0000, byte offsets):
 *   0x00 SRC      RD: main-memory offset (16-byte aligned);
 *                 WBLK, COL: RAM byte address
 *   0x04 DST      RD: RAM byte address (word aligned);
 *                 WBLK: main-memory offset (16-byte aligned); COL: offset
 *   0x08 N        RD: words; WBLK: blocks a row; COL: bytes a column
 *   0x0c STRIDE   COL: bytes between a column's bytes in main memory
 *   0x10 OUTER    WBLK, COL: rows / columns (>= 1)
 *   0x14 OSTRIDE  WBLK, COL: bytes between rows / columns in main memory
 *   0x18 GO       write: start (1 RD, 2 WBLK, 3 COL); read: [0] busy
 *
 *   RD    N words from main memory into the RAM: stream requests (the
 *         engine's port; the engine must be idle), at most 16 blocks
 *         outstanding, a word a cycle into the RAM
 *   WBLK  OUTER rows of N 16-byte blocks, read from the RAM one after
 *         the other, written whole through the block-write port
 *   COL   OUTER columns of N bytes, read from the RAM one after the
 *         other, byte k of column o to DST + o OSTRIDE + k STRIDE: byte
 *         stores through the controller's word port (read-modify-write
 *         there; the DRAM's masked writes are not used)
 *
 * The RAM's second port is the engine's while it copies out or packs;
 * the firmware never overlaps those with a DMA operation (rtl/mlm_soc.v
 * checks it in simulation).
 */

`default_nettype none

module mlm_dma #(
	parameter ADDR_BITS = 29,
	parameter RAM_AW = 15
) (
	input wire clk,
	input wire rst,

	input wire bus_req,
	input wire bus_we,
	input wire [4:2] bus_adr,
	input wire [31:0] bus_wdat,
	output reg bus_ack,
	output reg [31:0] bus_rdat,

	// the RAM's second port: read data the third cycle after pb_a
	output reg pb_we,
	output reg [RAM_AW-1:0] pb_a,
	output reg [31:0] pb_wd,
	input wire [31:0] pb_q,

	// stream reads (RD); rd_act: the stream's data is this unit's
	output reg sq_valid,
	input wire sq_ready,
	output reg [ADDR_BITS-5:0] sq_blk,
	output reg [8:0] sq_len,
	input wire sd_valid,
	input wire [127:0] sd_data,
	output wire rd_act,

	// block writes (WBLK)
	output reg wq_valid,
	input wire wq_ready,
	output reg [ADDR_BITS-5:0] wq_blk,
	output reg [127:0] wq_data,

	// byte stores (COL): wb_want while the operation runs, wb_own when
	// the SOC has given it the port
	output wire wb_want,
	input wire wb_own,
	output reg wb_req,
	output reg [ADDR_BITS-1:2] wb_adr,
	output reg [3:0] wb_sel,
	output reg [31:0] wb_dat,
	input wire wb_ack,

	output wire busy
);
	reg [31:0] r_src, r_dst, r_n, r_stride, r_outer, r_ostride;
	reg [1:0] op;                     // running: 0 none
	reg go;
	reg [1:0] go_op;
	assign busy = (op != 2'd0) || go;

	always @(posedge clk) begin
		bus_ack <= 1'b0;
		go <= 1'b0;
		if (rst) begin
			bus_rdat <= 32'd0;
		end else if (bus_req && !bus_ack) begin
			bus_ack <= 1'b1;
			bus_rdat <= {31'd0, busy};
			if (bus_we) case (bus_adr)
			3'd0: r_src <= bus_wdat;
			3'd1: r_dst <= bus_wdat;
			3'd2: r_n <= bus_wdat;
			3'd3: r_stride <= bus_wdat;
			3'd4: r_outer <= bus_wdat;
			3'd5: r_ostride <= bus_wdat;
			3'd6: begin go <= 1'b1; go_op <= bus_wdat[1:0]; end
			default: ;
			endcase
		end
	end

	// ---------------------------------------------------------------
	// RD: requests, a 16-block FIFO, the words into the RAM

	reg [127:0] fifo [0:15];
	reg [4:0] f_wp, f_rp;
	wire f_empty = (f_wp == f_rp);
	reg [16:0] rq_left;               // blocks still to request
	reg [16:0] rx_left;               // blocks still to arrive
	reg [ADDR_BITS-5:0] rq_ptr;
	reg [4:0] space;                  // FIFO slots not yet promised
	reg [RAM_AW-1:0] rd_wa;           // next RAM word
	reg [17:0] rd_wleft;              // words still to write
	reg [1:0] rd_w;                   // word of the head block
	assign rd_act = (op == 2'd1) && (rx_left != 17'd0);
	// the next request's size, registered (rq_left changes only as a
	// request goes out, and the next can go out two cycles later at the
	// earliest)
	reg [4:0] piece;
	reg rq_more;
	always @(posedge clk) begin
		piece <= (rq_left > 17'd16) ? 5'd16 : rq_left[4:0];
		rq_more <= (rq_left != 17'd0);
	end
	reg rd_new;                       // the first cycle of an RD: piece not yet valid
	wire rd_iss = (op == 2'd1) && !sq_valid && rq_more && (space >= piece) && !rd_new;
	wire pop = (op == 2'd1) && !f_empty && (rd_w == 2'd3 || rd_wleft == 18'd1);
	always @(posedge clk) begin
		if (sd_valid && rd_act) fifo[f_wp[3:0]] <= sd_data;
	end
	wire [127:0] f_head = fifo[f_rp[3:0]];

	// ---------------------------------------------------------------
	// WBLK and COL: the RAM read side

	reg [RAM_AW-1:0] rs_a;            // next RAM word to read
	reg [31:0] k_in, o_out;           // position in the row / column, row / column
	reg [ADDR_BITS-1:0] o_base;       // main-memory offset of the row / column
	reg [ADDR_BITS-1:0] c_adr;        // COL: this byte's offset
	reg [1:0] rk;                     // WBLK: words of the block read
	reg [2:0] wst;                    // 0 read, 1-2 wait, 3 present / store, 4 gap
	reg [1:0] c_byte;                 // COL: the byte of the RAM word
	reg [31:0] c_srcb;                // COL: RAM byte address of this byte
	reg pv1, pv2, pv3;                // a RAM read in flight (data with pv3)
	reg [1:0] pk1, pk2, pk3;
	assign wb_want = (op == 2'd3);

	always @(posedge clk) begin
		pb_we <= 1'b0;
		pv1 <= 1'b0;
		rd_new <= go && op == 2'd0;
		pv2 <= pv1;
		pv3 <= pv2;
		pk2 <= pk1;
		pk3 <= pk2;
		if (rst) begin
			op <= 2'd0;
			sq_valid <= 1'b0;
			wq_valid <= 1'b0;
			wb_req <= 1'b0;
			f_wp <= 5'd0; f_rp <= 5'd0;
			pv2 <= 1'b0;
			pv3 <= 1'b0;
		end else begin
			if (go && op == 2'd0) begin
				op <= go_op;
				// RD
				rq_left <= (r_n[16:0] + 17'd3) >> 2;
				rx_left <= (r_n[16:0] + 17'd3) >> 2;
				rq_ptr <= r_src[ADDR_BITS-1:4];
				space <= 5'd16;
				rd_wa <= r_dst[RAM_AW+1:2];
				rd_wleft <= r_n[17:0];
				rd_w <= 2'd0;
				f_wp <= 5'd0; f_rp <= 5'd0;
				// WBLK, COL
				rs_a <= r_src[RAM_AW+1:2];
				c_srcb <= r_src;
				k_in <= 32'd0;
				o_out <= 32'd0;
				o_base <= r_dst[ADDR_BITS-1:0];
				c_adr <= r_dst[ADDR_BITS-1:0];
				rk <= 2'd0;
				wst <= 3'd0;
				if (go_op == 2'd0 || r_n == 32'd0 || ((go_op != 2'd1) && r_outer == 32'd0)) op <= 2'd0;
			end
			case (op)
			2'd1: begin
				// requests: a piece when the FIFO will have room for it
				if (sq_valid && sq_ready) begin
					sq_valid <= 1'b0;
				end else if (rd_iss) begin
					sq_valid <= 1'b1;
					sq_blk <= rq_ptr;
					sq_len <= {4'd0, piece};
					rq_ptr <= rq_ptr + piece;
					rq_left <= rq_left - piece;
				end
				space <= space - (rd_iss ? piece : 5'd0) + (pop ? 5'd1 : 5'd0);
				if (sd_valid && rd_act) begin
					f_wp <= f_wp + 5'd1;
					rx_left <= rx_left - 17'd1;
				end
				// drain: a word a cycle
				if (!f_empty && rd_wleft != 18'd0) begin
					pb_we <= 1'b1;
					pb_a <= rd_wa;
					pb_wd <= f_head[32*rd_w +: 32];
					rd_wa <= rd_wa + 1'b1;
					rd_wleft <= rd_wleft - 18'd1;
					rd_w <= rd_w + 2'd1;
					if (pop) begin
						f_rp <= f_rp + 5'd1;
						rd_w <= 2'd0;
					end
					if (rd_wleft == 18'd1) op <= 2'd0;
				end
			end
			2'd2: case (wst)
				// read the block's four words, then present it
				3'd0: begin
					pb_a <= rs_a;
					rs_a <= rs_a + 1'b1;
					pv1 <= 1'b1;
					pk1 <= rk;
					rk <= rk + 2'd1;
					if (rk == 2'd3) wst <= 3'd1;
				end
				3'd1: wst <= 3'd2;
				3'd2: wst <= 3'd3;
				3'd3: if (!wq_valid) begin
					wq_valid <= 1'b1;
					wq_blk <= o_base[ADDR_BITS-1:4] + k_in[ADDR_BITS-5:0];
				end else if (wq_ready) begin
					wq_valid <= 1'b0;
					wst <= 3'd0;
					if (k_in + 32'd1 == r_n) begin
						k_in <= 32'd0;
						o_base <= o_base + r_ostride[ADDR_BITS-1:0];
						o_out <= o_out + 32'd1;
						if (o_out + 32'd1 == r_outer) op <= 2'd0;
					end else k_in <= k_in + 32'd1;
				end
				default: wst <= 3'd0;
				endcase
			2'd3: case (wst)
				// read the byte's word, then store the byte
				3'd0: if (wb_own) begin
					pb_a <= c_srcb[RAM_AW+1:2];
					c_byte <= c_srcb[1:0];
					pv1 <= 1'b1;
					wst <= 3'd1;
				end
				3'd1: wst <= 3'd2;
				3'd2: wst <= 3'd3;     // pv3's data is in wb_dat as wb_req rises
				3'd3: if (!wb_req) begin
					wb_req <= 1'b1;
					wb_adr <= c_adr[ADDR_BITS-1:2];
					wb_sel <= 4'd1 << c_adr[1:0];
				end else if (wb_ack) begin
					wb_req <= 1'b0;
					wst <= 3'd4;
					c_srcb <= c_srcb + 32'd1;
					if (k_in + 32'd1 == r_n) begin
						k_in <= 32'd0;
						o_base <= o_base + r_ostride[ADDR_BITS-1:0];
						c_adr <= o_base + r_ostride[ADDR_BITS-1:0];
						o_out <= o_out + 32'd1;
						if (o_out + 32'd1 == r_outer) op <= 2'd0;
					end else begin
						k_in <= k_in + 32'd1;
						c_adr <= c_adr + r_stride[ADDR_BITS-1:0];
					end
				end
				default: wst <= 3'd0; // one idle cycle after each store
				endcase
			default: ;
			endcase
		end
	end

	// the RAM's data: a WBLK word into its block, a COL byte into all
	// four lanes of the store
	always @(posedge clk) begin
		if (pv3 && op == 2'd2) wq_data[32*pk3 +: 32] <= pb_q;
		if (pv3 && op == 2'd3) wb_dat <= {4{pb_q[8*c_byte +: 8]}};
	end
endmodule

`default_nettype wire

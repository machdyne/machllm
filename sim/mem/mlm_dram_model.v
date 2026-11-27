/*
 * machllm
 * Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
 *
 * DDR3 model at the controller/PHY boundary -- SIMULATION ONLY.
 *
 * Takes the controller's PHY-side command and data outputs (one command
 * per controller cycle, on the first of its two DRAM clocks; write data
 * whole with the WRITE; read data whole, phy_rvalid a fixed latency
 * after the READ, as rtl/mem/ddr3_phy_ecp5.v delivers it) and checks
 * the DDR3 rules a controller can break, failing the run on the first:
 *
 *   bank state (no ACT to an active bank, no READ/WRITE to an idle one),
 *   tRCD, tRAS, tRP, tRC, tRRD, tFAW, tCCD, tRTP, tWR (auto-precharge),
 *   tWTR, tRFC (against every command), REFRESH only with all banks
 *   idle, and the refresh interval.
 *
 * Unlike Zeitlos's rtl/tb/ddr3_model.v it allows any number of reads in
 * flight, which the stream needs; it does not model DQS (that is the
 * PHY's business and is covered by Zeitlos's model and the hardware).
 *
 * Storage is the testbench's: DPI calls with the block index the
 * controller's address map gives ({row, column block, bank}).
 * Timings are the MT41K256M16TW-107's, in picoseconds.
 */

`default_nettype none

module mlm_dram_model #(
	parameter TCK_PS = 10417,       // DRAM clock (96 MHz)
	parameter RD_LAT = 14,          // controller cycles, READ to rvalid
	parameter CL = 6
) (
	input wire clk,
	input wire cs_n, ras_n, cas_n, we_n,
	input wire [15:0] a,
	input wire [2:0] ba,
	input wire wren,
	input wire [127:0] wdata,
	input wire [15:0] wmask,
	input wire rden,
	output reg rvalid,
	output reg [127:0] rdata
);

	import "DPI-C" function int dram_rd32(input int blk, input int word);
	import "DPI-C" function void dram_wr32(input int blk, input int word, input int val, input int mask);
	import "DPI-C" function void dram_fail(input int code, input longint t_ps);

	localparam integer TRCD = 13910, TRP = 13910, TRAS = 34000, TRC = 47910;
	localparam integer TRRD_NS = 7500, TFAW = 35000, TRTP_NS = 7500, TWR = 15000;
	localparam integer TWTR_NS = 7500, TRFC = 260000, TREFI = 7800000;
	localparam integer CK = TCK_PS;
	function [63:0] mx(input [63:0] x, input [63:0] y); mx = (x > y) ? x : y; endfunction
	localparam integer TRRD = (4 * CK > TRRD_NS) ? 4 * CK : TRRD_NS;
	localparam integer TRTP = (4 * CK > TRTP_NS) ? 4 * CK : TRTP_NS;
	localparam integer TWTR = (4 * CK > TWTR_NS) ? 4 * CK : TWTR_NS;
	localparam integer TCCD = 4 * CK;

	// failure codes, for the testbench's message
	localparam F_ACT_ACTIVE = 1, F_ACT_TRP = 2, F_ACT_TRC = 3, F_TRRD = 4, F_TFAW = 5,
	           F_RD_IDLE = 6, F_TRCD = 7, F_TCCD = 8, F_WR_IDLE = 9, F_REF_ACTIVE = 10,
	           F_TRFC = 11, F_REFI = 12, F_TWTR = 13, F_NOINIT = 14, F_PRE = 15;

	reg [63:0] t;                   // ps, time of this cycle's command
	reg active [0:7];
	reg [14:0] row [0:7];
	reg [63:0] t_act [0:7];
	reg [63:0] t_idle [0:7];        // precharge complete
	reg [63:0] t_act4 [0:3];        // last four ACTIVATEs
	reg [63:0] t_last_act, t_last_rd, t_last_wr_end, t_ref, t_ref_end;
	reg [3:0] mrs_count;
	reg init_ok;
	integer i, w;
	reg [3:0] cmd;

	// read return pipeline
	reg [RD_LAT:0] rpipe;
	reg [24:0] rblk [0:RD_LAT];

	initial begin
		t = 0;
		for (i = 0; i < 8; i = i + 1) begin
			active[i] = 1'b0; t_act[i] = 0; t_idle[i] = 0;
		end
		for (i = 0; i < 4; i = i + 1) t_act4[i] = 0;
		t_last_act = 0; t_last_rd = 0; t_last_wr_end = 0; t_ref = 0; t_ref_end = 0;
		mrs_count = 0; init_ok = 1'b0;
		rpipe = 0; rvalid = 1'b0;
	end

	function [24:0] blk_of(input [2:0] b, input [14:0] r, input [9:0] c);
		blk_of = {r, c[9:3], b};
	endfunction

	always @(posedge clk) begin
		t <= t + 2 * CK;
		cmd = {cs_n, ras_n, cas_n, we_n};

		// read returns
		rpipe <= {rpipe[RD_LAT-1:0], 1'b0};
		for (i = RD_LAT; i > 0; i = i - 1) rblk[i] <= rblk[i - 1];
		rvalid <= rpipe[RD_LAT - 1];
		if (rpipe[RD_LAT - 1])
			for (w = 0; w < 4; w = w + 1)
				rdata[32*w +: 32] <= dram_rd32({7'd0, rblk[RD_LAT - 1]}, w);

		if (!cs_n && cmd != 4'b0111) begin
			if (t_ref_end > t && cmd != 4'b1111) dram_fail(F_TRFC, t);
			case (cmd)
			4'b0000: begin                       // MRS
				mrs_count <= mrs_count + 1'b1;
			end
			4'b0110: begin                       // ZQCL
				if (mrs_count >= 4) init_ok <= 1'b1;
			end
			4'b0011: begin                       // ACTIVATE
				if (!init_ok) dram_fail(F_NOINIT, t);
				if (active[ba]) dram_fail(F_ACT_ACTIVE, t);
				if (t < t_idle[ba]) begin
					$display("ACT bank %0d row %0d at %0d ps; idle at %0d; last act %0d", ba, a[14:0], t, t_idle[ba], t_act[ba]);
					dram_fail(F_ACT_TRP, t);
				end
				if (t_act[ba] != 0 && t - t_act[ba] < TRC) dram_fail(F_ACT_TRC, t);
				if (t_last_act != 0 && t - t_last_act < TRRD) dram_fail(F_TRRD, t);
				if (t_act4[3] != 0 && t - t_act4[3] < TFAW) dram_fail(F_TFAW, t);
				active[ba] <= 1'b1;
				row[ba] <= a[14:0];
				t_act[ba] <= t;
				t_last_act <= t;
				t_act4[0] <= t; t_act4[1] <= t_act4[0];
				t_act4[2] <= t_act4[1]; t_act4[3] <= t_act4[2];
			end
			4'b0101: begin                       // READ
				if (!active[ba]) dram_fail(F_RD_IDLE, t);
				if (t - t_act[ba] < TRCD) dram_fail(F_TRCD, t);
				if (t_last_rd != 0 && t - t_last_rd < TCCD) dram_fail(F_TCCD, t);
				if (t_last_wr_end != 0 && t < t_last_wr_end + TWTR) dram_fail(F_TWTR, t);
				if (!rden) dram_fail(F_RD_IDLE, t);
				t_last_rd <= t;
				rpipe[0] <= 1'b1;
				rblk[0] <= blk_of(ba, row[ba], a[9:0]);
				if (a[10]) begin
					active[ba] <= 1'b0;
					t_idle[ba] <= mx(t + TRTP, t_act[ba] + TRAS) + TRP;
				end
			end
			4'b0100: begin                       // WRITE
				if (!active[ba]) dram_fail(F_WR_IDLE, t);
				if (t - t_act[ba] < TRCD) dram_fail(F_TRCD, t);
				if (!wren) dram_fail(F_WR_IDLE, t);
				for (w = 0; w < 4; w = w + 1)
					dram_wr32({7'd0, blk_of(ba, row[ba], a[9:0])}, w, wdata[32*w +: 32],
					          {28'd0, wmask[4*w +: 4]});
				t_last_wr_end <= t + (CL + 4) * CK;
				if (a[10]) begin
					active[ba] <= 1'b0;
					// WR from MR0 is 6 clocks (rtl: WR_CK)
					t_idle[ba] <= mx(t + (6 + 4 + 6) * CK, t_act[ba] + TRAS) + TRP;
				end
			end
			4'b0001: begin                       // REFRESH
				for (i = 0; i < 8; i = i + 1)
					if (active[i] || t < t_idle[i]) dram_fail(F_REF_ACTIVE, t);
				if (t_ref != 0 && t - t_ref > 9 * TREFI) dram_fail(F_REFI, t);
				t_ref <= t;
				t_ref_end <= t + TRFC;
			end
			4'b0010: dram_fail(F_PRE, t);        // PRECHARGE: never issued
			default: ;
			endcase
		end
		if (init_ok && t_ref != 0 && t - t_ref > 9 * TREFI) dram_fail(F_REFI, t);
	end

endmodule

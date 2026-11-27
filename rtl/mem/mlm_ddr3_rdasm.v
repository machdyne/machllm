/*
 * machllm's copy of Zeitlos's ddr3_rdasm (ext/zeitlos/rtl/mem/ddr3_rdasm.v),
 * same module name and ports, used in its place by the builds.
 *
 * The one change: the history indices off + 7 - b are registered. The
 * offsets are training settings, written by the CPU long before the
 * reads that use them, so a cycle's delay is invisible; the adders were
 * on the path into the 128 read-data flops (the DDR3-200 build's
 * critical path at one point). Read latency is unchanged.
 * sim/rdasm checks this against the original.
 */

`default_nettype none

module ddr3_rdasm #(
	parameter HIST = 32
) (
	input wire clk_i,
	input wire rst_i,

	input wire cap_i,

	input wire [15:0] q0_i,
	input wire [15:0] q1_i,
	input wire [15:0] q2_i,
	input wire [15:0] q3_i,

	input wire [3:0] off0_i,
	input wire [3:0] off1_i,

	output reg [127:0] rdata_o,
	output reg rvalid_o
);

	reg [7:0] hist0 [0:HIST-1];
	reg [7:0] hist1 [0:HIST-1];

	integer h;
	always @(posedge clk_i) begin
		for (h = HIST - 1; h >= 4; h = h - 1) begin
			hist0[h] <= hist0[h - 4];
			hist1[h] <= hist1[h - 4];
		end
		hist0[3] <= q0_i[7:0];   hist1[3] <= q0_i[15:8];
		hist0[2] <= q1_i[7:0];   hist1[2] <= q1_i[15:8];
		hist0[1] <= q2_i[7:0];   hist1[1] <= q2_i[15:8];
		hist0[0] <= q3_i[7:0];   hist1[0] <= q3_i[15:8];
	end

	// the indices, registered (off + 7 - b is at most 22: five bits)
	reg [4:0] ix0 [0:7];
	reg [4:0] ix1 [0:7];
	integer k;
	always @(posedge clk_i)
		for (k = 0; k < 8; k = k + 1) begin
			ix0[k] <= {1'b0, off0_i} + 5'd7 - k[4:0];
			ix1[k] <= {1'b0, off1_i} + 5'd7 - k[4:0];
		end

	integer b;
	always @(posedge clk_i) begin
		rvalid_o <= 1'b0;
		if (rst_i) begin
			rvalid_o <= 1'b0;
		end else if (cap_i) begin
			for (b = 0; b < 8; b = b + 1) begin
				rdata_o[b*16 +: 8]     <= hist0[ix0[b]];
				rdata_o[b*16 + 8 +: 8] <= hist1[ix1[b]];
			end
			rvalid_o <= 1'b1;
		end
	end

endmodule

`default_nettype wire

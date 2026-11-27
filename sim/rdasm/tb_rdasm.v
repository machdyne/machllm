module tb_rdasm (
	input wire clk, input wire rst, input wire cap,
	input wire [15:0] q0, input wire [15:0] q1, input wire [15:0] q2, input wire [15:0] q3,
	input wire [3:0] off0, input wire [3:0] off1,
	output wire [127:0] ra, output wire va, output wire [127:0] rb, output wire vb
);
	ddr3_rdasm a (.clk_i(clk), .rst_i(rst), .cap_i(cap), .q0_i(q0), .q1_i(q1), .q2_i(q2), .q3_i(q3),
		.off0_i(off0), .off1_i(off1), .rdata_o(ra), .rvalid_o(va));
	ddr3_rdasm_zeitlos b (.clk_i(clk), .rst_i(rst), .cap_i(cap), .q0_i(q0), .q1_i(q1), .q2_i(q2), .q3_i(q3),
		.off0_i(off0), .off1_i(off1), .rdata_o(rb), .rvalid_o(vb));
endmodule

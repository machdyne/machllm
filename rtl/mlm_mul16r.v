/*
 * machllm -- a 16 x 16 unsigned multiply, p = a * b two cycles later
 * (operands and product registered in the fabric; Yosys puts the
 * multiply in a DSP block).
 */
`default_nettype none
module mlm_mul16r (
	input wire clk,
	input wire [15:0] a,
	input wire [15:0] b,
	output wire [31:0] p
);
	// Plain Verilog on purpose: Yosys maps the multiply to a DSP block
	// with the registers in the fabric. Instantiating MULT18X18D with
	// the block's own registers gave all-zero products on the board
	// (output register: not routed; input registers: did not load).
	reg [15:0] ar, br;
	reg [31:0] pr;
	always @(posedge clk) begin
		ar <= a;
		br <= b;
		pr <= ar * br;
	end
	assign p = pr;
endmodule
`default_nettype wire

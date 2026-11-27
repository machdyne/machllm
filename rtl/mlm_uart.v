/*
 * machllm
 * Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
 *
 * mlm_uart: 8N1 serial port, 16-byte receive queue.
 *
 *   0  DATA    write: send a byte (when TX ready); read: next received
 *              byte, [8] set if the queue was empty
 *   1  STATUS  [0] TX ready  [1] RX has data  [2] RX overrun (sticky)
 *   2  DIV     clk cycles per bit (reset: the SOC's BAUD, 48 at 48 MHz for 1M)
 */

`default_nettype none

module mlm_uart #(
	parameter [15:0] DIV_RESET = 16'd48
) (
	input wire clk,
	input wire rst,
	input wire bus_req,
	input wire bus_we,
	input wire [3:2] bus_adr,
	input wire [31:0] bus_wdat,
	output reg bus_ack,
	output reg [31:0] bus_rdat,
	output reg tx = 1'b1,
	input wire rx
);
	reg [15:0] div;

	// transmit
	reg [9:0] tsh;
	reg [3:0] tbits;
	reg [15:0] tcnt;
	// ready once the stop bit has had its full time, not when it starts
	// (it used to be the latter: back-to-back bytes then had a stop bit
	// of a clock or two, which a real receiver reads as garbage)
	wire tx_ready = (tbits == 4'd0) && (tcnt == 16'd0);
	always @(posedge clk) begin
		if (rst) begin
			tx <= 1'b1; tbits <= 4'd0; tcnt <= 16'd0;
		end else if (bus_req && !bus_ack && bus_we && bus_adr == 2'd0 && tx_ready) begin
			tsh <= {1'b1, bus_wdat[7:0], 1'b0};
			tbits <= 4'd10;
			tcnt <= 16'd0;
		end else if (tbits != 4'd0 || tcnt != 16'd0) begin
			if (tcnt == 16'd0) begin
				tx <= tsh[0];
				tsh <= {1'b1, tsh[9:1]};
				tbits <= tbits - 4'd1;
				tcnt <= div - 16'd1;
			end else tcnt <= tcnt - 16'd1;
		end
	end

	// receive
	reg [2:0] rs;
	always @(posedge clk) rs <= {rs[1:0], rx};
	wire rxs = rs[2];
	reg [3:0] rbits;
	reg [15:0] rcnt;
	reg [7:0] rsh;
	reg [7:0] q [0:15];
	reg [4:0] qw, qr;
	reg ovr;
	wire q_empty = (qw == qr);
	wire q_full = (qw - qr) == 5'd16;
	reg pop;
	always @(posedge clk) begin
		if (rst) begin
			rbits <= 4'd0; qw <= 5'd0; ovr <= 1'b0;
		end else begin
			if (rbits == 4'd0) begin
				if (!rxs) begin rbits <= 4'd10; rcnt <= {1'b0, div[15:1]}; end
			end else if (rcnt == 16'd0) begin
				rcnt <= div - 16'd1;
				rbits <= rbits - 4'd1;
				if (rbits == 4'd10) begin
					if (rxs) rbits <= 4'd0;              // false start
				end else if (rbits == 4'd1) begin
					if (rxs) begin
						if (q_full) ovr <= 1'b1;
						else begin q[qw[3:0]] <= rsh; qw <= qw + 5'd1; end
					end
				end else
					rsh <= {rxs, rsh[7:1]};
			end else rcnt <= rcnt - 16'd1;
		end
	end

	always @(posedge clk) begin
		bus_ack <= 1'b0;
		pop <= 1'b0;
		if (rst) begin
			div <= DIV_RESET;
			qr <= 5'd0;
		end else if (bus_req && !bus_ack) begin
			if (bus_adr == 2'd0 && bus_we && !tx_ready) begin
				// wait until the transmitter takes it
			end else begin
				bus_ack <= 1'b1;
				bus_rdat <= 32'd0;
				case (bus_adr)
				2'd0: if (!bus_we) begin
					if (q_empty) bus_rdat <= 32'h100;
					else begin bus_rdat <= {24'd0, q[qr[3:0]]}; qr <= qr + 5'd1; end
				end
				2'd1: bus_rdat <= {29'd0, ovr, !q_empty, tx_ready};
				2'd2: if (bus_we) div <= bus_wdat[15:0]; else bus_rdat <= {16'd0, div};
				default: ;
				endcase
			end
		end
	end
endmodule

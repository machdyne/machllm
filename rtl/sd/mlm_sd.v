/*
 * machllm
 * Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
 *
 * mlm_sd: SD card host, native mode, 1- or 4-bit data bus, reading
 * straight into main memory.
 *
 * Two engines driven by the CPU (fw/common/sd.c has the protocol):
 *
 *   CMD  sends a 48-bit command with its CRC7, receives a 48- or
 *        136-bit response (CRC7 checked for 48), optionally waits for
 *        the card's busy (DAT0 low) to end.
 *   DAT  receives N 512-byte blocks on DAT0 or DAT0-3, checks each
 *        line's CRC16, and writes the data to main memory sixteen bytes
 *        at a time through a block-write port (wq_*). Armed before the
 *        command that starts the transfer (CMD17/18), so no edge is
 *        missed.
 *
 * SCK runs only while an engine needs it, and stops while the write
 * queue is full (the card waits for the clock). SCK = clk / (2 (DIV+1)):
 * DIV 59 gives 400 kHz at 48 MHz for identification, DIV 0 gives 24 MHz.
 * The host changes CMD after a falling edge and samples CMD and DAT at
 * rising edges, as default-speed mode wants.
 *
 * -- registers (word offsets) ------------------------------------
 *   0  DIV       SCK half period - 1, in clk cycles
 *   1  ARG       command argument
 *   2  CMD       write starts: [5:0] index, [9:8] response 0 none,
 *                1 48-bit, 2 136-bit, [10] wait busy after the response,
 *                [11] no CRC check (R3)
 *   3  STATUS    [0] cmd busy [1] cmd timeout [2] cmd crc error
 *                [3] dat busy [4] dat crc error [5] dat timeout
 *                [6] DAT0 [7] write queue not empty
 *                errors clear when the engine is started again
 *   4-7 RESP0-3  response bits 31:0 .. 127:96 (48-bit: RESP0 is the
 *                32-bit field, RESP1[5:0] the index)
 *   8  DADDR     memory byte address for the data (16-byte aligned)
 *   9  DBLOCKS   write arms the DAT engine: [15:0] blocks, [16] 4-bit
 *   10 DDONE     blocks received
 *   11 IDLE      write N: N clocks with CMD high (power-up: 74+)
 */

`default_nettype none

module mlm_sd #(
	parameter BLK_BITS = 25          // memory block (16-byte) address bits
) (
	input wire clk,
	input wire rst,

	input wire bus_req,
	input wire bus_we,
	input wire [5:2] bus_adr,
	input wire [31:0] bus_wdat,
	output reg bus_ack,
	output reg [31:0] bus_rdat,

	// block writes to memory
	output wire wq_valid,
	input wire wq_ready,
	output wire [BLK_BITS-1:0] wq_blk,
	output wire [127:0] wq_data,

	// pins
	output reg sd_sck,
	output reg sd_cmd_o,
	output reg sd_cmd_oe,
	input wire sd_cmd_i,
	input wire [3:0] sd_dat_i
);

	// ---------------------------------------------------------------
	// registers

	reg [7:0] r_div;
	reg [31:0] r_arg;
	reg [11:0] r_cmd;
	reg [127:0] resp;
	reg [31:0] r_daddr;
	reg [15:0] r_dblocks;
	reg r_wide;
	reg [15:0] r_ddone;
	reg [15:0] r_idle;

	reg cmd_busy, cmd_tmo, cmd_crc;
	reg dat_busy, dat_crc, dat_tmo;
	reg cmd_go, dat_go, idle_go;

	wire q_nonempty;

	always @(posedge clk) begin
		bus_ack <= 1'b0;
		cmd_go <= 1'b0;
		dat_go <= 1'b0;
		idle_go <= 1'b0;
		if (rst) begin
			r_div <= 8'd59;
		end else if (bus_req && !bus_ack) begin
			bus_ack <= 1'b1;
			bus_rdat <= 32'd0;
			case (bus_adr)
			4'd0: if (bus_we) r_div <= bus_wdat[7:0]; else bus_rdat <= {24'd0, r_div};
			4'd1: if (bus_we) r_arg <= bus_wdat; else bus_rdat <= r_arg;
			4'd2: if (bus_we) begin r_cmd <= bus_wdat[11:0]; cmd_go <= 1'b1; end
			4'd3: bus_rdat <= {24'd0, q_nonempty, din_q[0], dat_tmo, dat_crc, dat_busy,
			                   cmd_crc, cmd_tmo, cmd_busy};
			4'd4: bus_rdat <= resp[31:0];
			4'd5: bus_rdat <= resp[63:32];
			4'd6: bus_rdat <= resp[95:64];
			4'd7: bus_rdat <= resp[127:96];
			4'd8: if (bus_we) r_daddr <= bus_wdat; else bus_rdat <= r_daddr;
			4'd9: if (bus_we) begin r_dblocks <= bus_wdat[15:0]; r_wide <= bus_wdat[16]; dat_go <= 1'b1; end
			4'd10: bus_rdat <= {16'd0, r_ddone};
			4'd11: if (bus_we) begin r_idle <= bus_wdat[15:0]; idle_go <= 1'b1; end
			default: ;
			endcase
		end
	end

	// ---------------------------------------------------------------
	// SCK

	reg [15:0] idle_left;
	wire q_full;
	// dtrail: clocks after a read's last block (its end bit, then the
	// eight a card needs before it takes the next command; without them
	// a real card misses that command, the simulation model did not)
	reg [3:0] dtrail;
	wire need_clk = cmd_busy || (idle_left != 16'd0) || (dat_busy && !q_full) || (dtrail != 4'd0);
	reg [7:0] divc;
	reg rise, fall;                 // this cycle SCK goes high / low

	always @(posedge clk) begin
		rise <= 1'b0;
		fall <= 1'b0;
		if (rst) begin
			sd_sck <= 1'b0;
			divc <= 8'd0;
		end else if (divc != 8'd0) begin
			divc <= divc - 8'd1;
		end else if (sd_sck) begin
			sd_sck <= 1'b0;
			fall <= 1'b1;
			divc <= {1'b0, r_div[6:0]};
		end else if (need_clk) begin
			sd_sck <= 1'b1;
			rise <= 1'b1;
			// r_div[7]: the high half one cycle shorter, for an odd
			// period (3 cycles: 20 MHz from 60). The low half stays
			// long: the card's output settles in it, and the host's
			// CMD changes after the falling edge
			divc <= {1'b0, r_div[6:0]} - {7'd0, r_div[7]};
		end
	end

	// inputs, registered; sampled the cycle after a rising edge
	reg [3:0] din_q;
	reg cin_q;
	reg rs;                         // "rise" delayed: inputs as at the edge
	always @(posedge clk) begin
		din_q <= sd_dat_i;
		cin_q <= sd_cmd_i;
		rs <= rise;
	end

	// ---------------------------------------------------------------
	// idle clocks

	always @(posedge clk) begin
		if (rst) idle_left <= 16'd0;
		else if (idle_go) idle_left <= r_idle;
		else if (rise && idle_left != 16'd0) idle_left <= idle_left - 16'd1;
	end

	// ---------------------------------------------------------------
	// CRC7

	function [6:0] crc7_bit(input [6:0] c, input b);
		reg f;
		begin
			f = c[6] ^ b;
			crc7_bit = {c[5:3], c[2] ^ f, c[1:0], f};
		end
	endfunction

	// ---------------------------------------------------------------
	// CMD engine

	localparam C_IDLE = 3'd0, C_SEND = 3'd1, C_WAITR = 3'd2, C_RESP = 3'd3, C_BUSY = 3'd4,
	           C_GAP = 3'd5;
	reg [2:0] cst;
	reg [47:0] csh;                 // bits to send, MSB first
	reg [5:0] ccnt;
	reg [7:0] rcnt;                 // response bits left / timeout
	reg [6:0] ccrc;
	reg [135:0] rsh;
	reg [7:0] rlen;
	reg [23:0] btmo;

	wire [39:0] cmd_head = {2'b01, r_cmd[5:0], r_arg};
	reg [6:0] head_crc;
	integer k;
	always @(*) begin
		head_crc = 7'd0;
		for (k = 39; k >= 0; k = k - 1) head_crc = crc7_bit(head_crc, cmd_head[k]);
	end

	always @(posedge clk) begin
		if (rst) begin
			cst <= C_IDLE;
			cmd_busy <= 1'b0;
			sd_cmd_o <= 1'b1;
			sd_cmd_oe <= 1'b1;
		end else case (cst)
		C_IDLE: begin
			sd_cmd_oe <= 1'b1;
			sd_cmd_o <= 1'b1;
			if (cmd_go) begin
				csh <= {cmd_head, head_crc, 1'b1};
				ccnt <= 6'd48;
				cmd_busy <= 1'b1;
				cmd_tmo <= 1'b0;
				cmd_crc <= 1'b0;
				cst <= C_SEND;
			end
		end
		default: ;
		endcase

		// the send/receive steps, on clock edges
		if (!rst) begin
			if (cst == C_SEND && fall) begin
				if (ccnt == 6'd0) begin
					sd_cmd_oe <= 1'b0;          // release CMD
					sd_cmd_o <= 1'b1;
					if (r_cmd[9:8] == 2'd0) begin
						cst <= r_cmd[10] ? C_BUSY : C_GAP;
						rcnt <= 8'd8;
						btmo <= 24'hffffff;
					end else begin
						rcnt <= 8'd64;              // NCR: response within 64 clocks
						rlen <= (r_cmd[9:8] == 2'd2) ? 8'd136 : 8'd48;
						cst <= C_WAITR;
					end
				end else begin
					sd_cmd_o <= csh[47];
					csh <= {csh[46:0], 1'b1};
					ccnt <= ccnt - 6'd1;
				end
			end
			if (cst == C_WAITR && rs) begin
				if (!cin_q) begin
					rsh <= 136'd0;
					rlen <= rlen - 8'd1;
					ccrc <= 7'd0;
					cst <= C_RESP;
				end else if (rcnt == 8'd0) begin
					cmd_tmo <= 1'b1;
					cmd_busy <= 1'b0;
					cst <= C_IDLE;
				end else
					rcnt <= rcnt - 8'd1;
			end
			if (cst == C_RESP && rs) begin
				rsh <= {rsh[134:0], cin_q};
				// CRC7 over the 40 bits after the start bit, 48-bit only
				if (rlen > 8'd8 && r_cmd[9:8] == 2'd1)
					ccrc <= crc7_bit(ccrc, cin_q);
				rlen <= rlen - 8'd1;
				if (rlen == 8'd1) begin
					// rsh now holds the response without its start bit
					if (r_cmd[9:8] == 2'd1) begin
						// rsh[45:0]: transmission bit, index, argument, CRC7
						resp <= {90'd0, rsh[44:39], rsh[38:7]};
						if (!r_cmd[11] && ccrc != rsh[6:0]) cmd_crc <= 1'b1;
					end else
						// R2: CID/CSD bits 127:1 and the end bit
						resp <= {rsh[126:0], cin_q};
					cst <= r_cmd[10] ? C_BUSY : C_GAP;
					rcnt <= 8'd8;
					btmo <= 24'hffffff;
				end
			end
			if (cst == C_BUSY && rs) begin
				// two clocks before looking (NCR for busy), then until DAT0 high
				if (rcnt > 8'd6) rcnt <= rcnt - 8'd1;
				else if (din_q[0]) begin cst <= C_GAP; rcnt <= 8'd8; end
				else if (btmo == 24'd0) begin cmd_tmo <= 1'b1; cst <= C_GAP; end
				else btmo <= btmo - 24'd1;
			end
			if (cst == C_GAP && rs) begin
				// NRC: eight clocks before the next command
				if (rcnt == 8'd0) begin
					cmd_busy <= 1'b0;
					cst <= C_IDLE;
				end else rcnt <= rcnt - 8'd1;
			end
		end
	end

	// ---------------------------------------------------------------
	// DAT engine

	localparam D_IDLE = 3'd0, D_START = 3'd1, D_DATA = 3'd2, D_CRC = 3'd3, D_END = 3'd4;
	reg [2:0] dst;
	reg [12:0] dcnt;                // nibbles / bits left in the block
	reg [4:0] ccnt16;
	reg [15:0] crc0, crc1, crc2, crc3;
	reg [15:0] rc0, rc1, rc2, rc3;
	reg [127:0] dbuf;
	reg [4:0] dpos;                 // nibbles (4-bit) in dbuf; bits/4 in 1-bit
	reg [7:0] bbuf;
	reg [2:0] bcnt;
	reg [24:0] dtmo;
	reg [BLK_BITS-1:0] dblk;

	// write queue: 4 entries
	reg [127:0] q_data [0:3];
	reg [BLK_BITS-1:0] q_blk [0:3];
	reg [2:0] q_wp, q_rp;
	assign q_full = (q_wp - q_rp) >= 3'd3;      // one spare for the cycle in flight
	assign q_nonempty = (q_wp != q_rp);
	assign wq_valid = q_nonempty;
	assign wq_blk = q_blk[q_rp[1:0]];
	assign wq_data = q_data[q_rp[1:0]];
	reg q_push;
	reg [127:0] q_pd;
	reg [BLK_BITS-1:0] q_pb;
	always @(posedge clk) begin
		if (rst) begin
			q_wp <= 3'd0; q_rp <= 3'd0;
		end else begin
			if (q_push) begin
				q_data[q_wp[1:0]] <= q_pd;
				q_blk[q_wp[1:0]] <= q_pb;
				q_wp <= q_wp + 3'd1;
			end
			if (wq_valid && wq_ready) q_rp <= q_rp + 3'd1;
		end
	end

	function [15:0] crc16_bit(input [15:0] c, input b);
		reg f;
		begin
			f = c[15] ^ b;
			crc16_bit = {c[14:12], c[11] ^ f, c[10:5], c[4] ^ f, c[3:0], f};
		end
	endfunction

	always @(posedge clk) begin
		q_push <= 1'b0;
		if (rise && dtrail != 4'd0) dtrail <= dtrail - 4'd1;
		if (rst) begin
			dst <= D_IDLE;
			dat_busy <= 1'b0;
			r_ddone <= 16'd0;
			dtrail <= 4'd0;
		end else if (dat_go) begin
			dst <= (r_dblocks != 16'd0) ? D_START : D_IDLE;
			dat_busy <= (r_dblocks != 16'd0);
			dat_crc <= 1'b0;
			dat_tmo <= 1'b0;
			r_ddone <= 16'd0;
			dblk <= r_daddr[BLK_BITS+3:4];
			dtmo <= 25'h1ffffff;
		end else if (rs) case (dst)
			D_START: begin
				if (r_wide ? (din_q == 4'd0) : !din_q[0]) begin
					dst <= D_DATA;
					dcnt <= r_wide ? 13'd1024 : 13'd4096;
					crc0 <= 16'd0; crc1 <= 16'd0; crc2 <= 16'd0; crc3 <= 16'd0;
					dpos <= 5'd0;
					bcnt <= 3'd0;
				end else if (dtmo == 25'd0) begin
					dat_tmo <= 1'b1;
					dat_busy <= 1'b0;
					dst <= D_IDLE;
				end else
					dtmo <= dtmo - 25'd1;
			end
			D_DATA: begin
				crc0 <= crc16_bit(crc0, din_q[0]);
				if (r_wide) begin
					crc1 <= crc16_bit(crc1, din_q[1]);
					crc2 <= crc16_bit(crc2, din_q[2]);
					crc3 <= crc16_bit(crc3, din_q[3]);
					// high nibble first; byte n at bits 8n+7:8n
					if (!dpos[0]) dbuf[8*dpos[4:1] + 4 +: 4] <= din_q;
					else dbuf[8*dpos[4:1] +: 4] <= din_q;
					dpos <= dpos + 5'd1;
					if (dpos == 5'd31) begin
						q_push <= 1'b1;
						q_pd <= {dbuf[127:124], din_q, dbuf[119:0]};
						q_pb <= dblk;
						dblk <= dblk + 1'b1;
					end
				end else begin
					bbuf <= {bbuf[6:0], din_q[0]};
					bcnt <= bcnt + 3'd1;
					if (bcnt == 3'd7) begin
						dbuf[8*dpos[3:0] +: 8] <= {bbuf[6:0], din_q[0]};
						dpos <= dpos + 5'd1;
						if (dpos[3:0] == 4'd15) begin
							q_push <= 1'b1;
							q_pd <= {bbuf[6:0], din_q[0], dbuf[119:0]};
							q_pb <= dblk;
							dblk <= dblk + 1'b1;
							dpos <= 5'd0;
						end
					end
				end
				dcnt <= dcnt - 13'd1;
				if (dcnt == 13'd1) begin
					dst <= D_CRC;
					ccnt16 <= 5'd16;
				end
			end
			D_CRC: begin
				rc0 <= {rc0[14:0], din_q[0]};
				rc1 <= {rc1[14:0], din_q[1]};
				rc2 <= {rc2[14:0], din_q[2]};
				rc3 <= {rc3[14:0], din_q[3]};
				ccnt16 <= ccnt16 - 5'd1;
				if (ccnt16 == 5'd1) dst <= D_END;
			end
			D_END: begin
				if (rc0 != crc0 || (r_wide && (rc1 != crc1 || rc2 != crc2 || rc3 != crc3)))
					dat_crc <= 1'b1;
				r_ddone <= r_ddone + 16'd1;
				if (r_ddone + 16'd1 == r_dblocks) begin
					dat_busy <= 1'b0;
					dtrail <= 4'd10;
					dst <= D_IDLE;
				end else begin
					dst <= D_START;
					dtmo <= 25'h1ffffff;
				end
			end
			default: ;
		endcase
	end

endmodule

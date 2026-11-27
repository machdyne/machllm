/*
 * machllm: usb_hid_host's program ROM (nand2mario's usb_hid_host,
 * Apache-2.0, in ext/zeitlos/rtl/ext/usb_hid_host) in logic rather than
 * a block RAM: 620 x 4 bits. Replaces that project's usb_hid_host_rom.v
 * (same module name and ports); the hex file is its own.
 */
`ifndef MLM_HID_ROM
`define MLM_HID_ROM "ext/zeitlos/rtl/ext/usb_hid_host/src/usb_hid_host_rom.hex"
`endif
module usb_hid_host_rom(clk, adr, data);
    parameter ROM_FILE = `MLM_HID_ROM;
    input clk;
    input [13:0] adr;
    output [3:0] data;
    reg [3:0] data;
    (* rom_style = "logic" *) reg [3:0] mem [0:619];
    initial $readmemh(ROM_FILE, mem);
    always @(posedge clk) data <= mem[adr];
endmodule

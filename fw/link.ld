/* machllm firmware: everything in the block RAM at 0 */
MEMORY { RAM (rwx) : ORIGIN = 0, LENGTH = 96K }
SECTIONS {
	.text : { *(.text.start) *(.text*) *(.rodata*) *(.srodata*) } > RAM
	.data : { *(.data*) *(.sdata*) } > RAM
	.bss (NOLOAD) : { _bss_start = .; *(.bss*) *(.sbss*) *(COMMON) . = ALIGN(4); _bss_end = .; } > RAM
	_stack_top = ORIGIN(RAM) + LENGTH(RAM);
}

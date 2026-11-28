W,H=1400,1040
o=[]
def r(x,y,w,h,fill="#fff",stroke="#7c8594",sw=1.5,rx=8,dash=None):
    d=f' stroke-dasharray="{dash}"' if dash else ''
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
def t(x,y,s,size=13,fill="#4a5262",anchor="start",bold=False):
    b=' font-weight="700"' if bold else ''
    s=s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    o.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}"{b}>{s}</text>')
def ln(x1,y1,x2,y2,c="#3a4250",w=2,end=True,start=False):
    m=(' marker-end="url(#ah)"' if end else '')+(' marker-start="url(#ah)"' if start else '')
    o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="{w}"{m}/>')
DSP="#c2561a"; INK="#1b1f27"
o.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="-apple-system, \'Segoe UI\', Helvetica, Arial, sans-serif">')
o.append('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#3a4250"/></marker></defs>')
r(0,0,W,H,fill="#ffffff",stroke="none",rx=18)
t(40,52,"machllm",28,INK,bold=True); t(168,52,"a 1-bit language model computer on one ECP5 FPGA",20,"#4a5262")
t(40,80,"Machdyne Sergei ML2 · Lattice LFE5U-45F · 512 MB DDR3L · open gateware and firmware · runs Bonsai 1.7B offline",15,"#6a7282")
# legend
r(1060,30,16,16,fill=DSP,stroke="none",rx=3); t(1084,44,"= one DSP multiplier block",13,"#4a5262")
r(1060,56,16,16,fill="#ffffff",stroke="#2f5c8a",rx=3); t(1084,70,"= one block RAM bank",13,"#4a5262")
# memory domain
r(20,100,950,870,fill="#eef4fb",stroke="#9db7d6",rx=14)
t(40,128,"MEMORY CLOCK DOMAIN · 90 MHz (DDR3-180)",15,"#2f5c8a",bold=True)
# left column
r(40,150,270,170,stroke="#2f5c8a",sw=2,rx=10)
t(175,180,"DDR3L · 512 MB",18,INK,"middle",True)
t(58,210,"Bonsai 1.7B, 1-bit: 246 MB",14,INK)
t(58,230,"sign bits + a scale per 128 weights",13)
t(58,256,"KV cache: int8 (V transposed)",14,INK)
t(58,290,"x16 at 180 MT/s",13); t(58,308,"~650 MB/s streamed",13,INK,bold=True)
ln(175,320,175,352,start=True)
r(40,356,270,120)
t(175,384,"DDR3 controller + PHY",16,INK,"middle",True)
t(175,407,"stream port: 16-byte bursts",13,anchor="middle")
t(175,425,"back to back, a row ahead",13,anchor="middle")
t(175,448,"CPU word port · read training",13,anchor="middle")
ln(175,510,175,480)
r(40,510,270,74)
t(175,539,"SD host",16,INK,"middle",True)
t(175,561,"native 4-bit bus, ~12 MB/s",13,anchor="middle")
ln(110,620,110,588)
r(40,620,140,80)
t(110,650,"microSD",16,INK,"middle",True); t(110,671,"BONSAI.MLM",13,anchor="middle"); t(110,688,"(FAT32)",13,anchor="middle")
# weight stream arrow
o.append(f'<line x1="310" y1="416" x2="360" y2="416" stroke="{DSP}" stroke-width="4"/>')
o.append(f'<polygon points="358,409 372,416 358,423" fill="{DSP}"/>')
t(341,402,"weights",11,DSP,"middle",True); t(341,437,"64 bit",11,DSP,"middle",True); t(341,451,"/ cycle",11,DSP,"middle",True)
# engine
EX,EY,EW=372,150,580
r(EX,EY,EW,800,fill="#fff6ef",stroke=DSP,sw=2.5,rx=12)
t(EX+EW/2,EY+30,"Matrix-vector engine",19,INK,"middle",True)
t(EX+EW/2,EY+50,"every weight read once per token (~242 MB); 16 of the chip's 39 DSP blocks in use",13,"#6a4a36","middle")
# X banks
y=EY+70
t(EX+20,y+14,"Input X: 16 block-RAM banks, 64 int8 values read per cycle",14,INK,bold=True)
bx=EX+20; bw=31; gap=3.2
for i in range(16):
    r(bx+i*(bw+gap),y+24,bw,30,fill="#ffffff",stroke="#2f5c8a",sw=1.2,rx=3)
    t(bx+i*(bw+gap)+bw/2,y+44,str(i),11,"#2f5c8a","middle")
t(EX+20,y+74,"3,072 values (MX format: blocks of 32 int8, a 4-bit exponent offset each)",13)
# lanes
y=EY+170
t(EX+20,y,"64 one-bit lanes per cycle, as two MX blocks of 32",14,INK,bold=True)
cols=[EX+20,EX+300]
for k,cx in enumerate(cols):
    t(cx,y+22,f"block {'ab'[k]}: 32 lanes",12,"#6a4a36")
    for i in range(32):
        r(cx+i*8.1,y+30,6.1,22,fill="#f4b48d",stroke="none",rx=1.5)
t(EX+20,y+72,"each lane: weight bit ? +x : −x",13)
# per-block pipeline
y=EY+262
for k,cx in enumerate(cols):
    w=260
    ln(cx+w/2,y-10,cx+w/2,y+2,w=1.6)
    r(cx,y+4,w,30,rx=6,stroke="#d99a72"); t(cx+w/2,y+24,"adder tree, 32 → 1",13,INK,"middle")
    ln(cx+w/2,y+34,cx+w/2,y+46,w=1.6)
    r(cx,y+48,w,32,fill=DSP,stroke="none",rx=6); t(cx+w/2,y+69,"× group scale (DSP)",13,"#ffffff","middle",True)
    ln(cx+w/2,y+80,cx+w/2,y+92,w=1.6)
    r(cx,y+94,w,30,rx=6,stroke="#d99a72"); t(cx+w/2,y+114,"<< (15 − block offset)",13,INK,"middle")
    ln(cx+w/2,y+124,EX+EW/2,y+150,w=1.6)
r(EX+150,y+152,EW-300,40,rx=7,stroke="#d99a72"); t(EX+EW/2,y+170,"both blocks added: exact int64 row sum",14,INK,"middle",True); t(EX+EW/2,y+186,"max |sum| tracked",12,anchor="middle")
ln(EX+EW/2,y+192,EX+EW/2,y+206,w=1.6)
# output banks
y=EY+500
t(EX+20,y,"2 output banks (one fills while the CPU reads the other)",14,INK,bold=True)
for i in range(2):
    r(EX+20+i*275,y+10,265,30,fill="#ffffff",stroke="#2f5c8a",sw=1.2,rx=4); t(EX+20+i*275+132,y+30,f"OUT bank {i}",13,"#2f5c8a","middle")
# 8-bit
y=EY+570
t(EX+20,y,"8-bit mode (attention over the KV cache): 8 int8 × int8 per cycle",14,INK,bold=True)
for i in range(8):
    r(EX+20+i*68,y+10,60,26,fill=DSP,stroke="none",rx=4); t(EX+50+i*68,y+28,"× DSP",11,"#ffffff","middle",True)
t(EX+20,y+54,"q · K rows for the scores, softmax · Vᵀ for the output",13)
# GLU
y=EY+655
t(EX+20,y,"GLU unit: SiLU(gate) × up, straight into the next X",14,INK,bold=True)
r(EX+20,y+10,130,26,rx=4,stroke="#2f5c8a",sw=1.2); t(EX+85,y+28,"sigmoid table",12,"#2f5c8a","middle")
for i in range(6):
    r(EX+162+i*66,y+10,58,26,fill=DSP,stroke="none",rx=4); t(EX+191+i*66,y+28,"× DSP",11,"#ffffff","middle",True)
# dual clock RAMs
y=EY+720
r(EX+20,y,EW-40,64,fill="#fbe3d2",stroke="#d99a72",dash="5 4",rx=7)
t(EX+EW/2,y+24,"the CPU's RAM, second port: DMA unit, packer, copy-out",14,INK,"middle",True)
t(EX+EW/2,y+46,"main memory ⇄ RAM · int32 → MX into X · results → RAM",13,anchor="middle")
# CPU domain
CX=990
r(CX,100,390,870,fill="#eef7f0",stroke="#93c2a0",rx=14)
t(CX+20,128,"CPU CLOCK DOMAIN · 49 MHz",15,"#2e6b40",bold=True)
r(CX+20,150,350,170,stroke="#2e6b40",sw=2,rx=10)
t(CX+195,180,"zeitlos32ps · RV32IM",18,INK,"middle",True)
t(CX+40,208,"firmware in 96 KB block RAM, integer only:",13)
for i,s in enumerate(["• tokenizer, chat template, sampling","• attention, KV cache, layer sequencing","• the same C code as the PC reference"]):
    t(CX+40,231+i*21,s,13,INK)
t(CX+40,302,"reads main memory through a 16-byte line",13,"#6a7282")
ln(CX+195,320,CX+195,346,w=3,end=False)
r(CX+20,346,350,180,stroke="#2e6b40",rx=10)
t(CX+195,376,"Vector unit",17,INK,"middle",True)
for i,s in enumerate(["RMSNorm · QK-norm · RoPE","residual adds · KV quantisation","softmax (fused, streamed)"]):
    t(CX+40,404+i*21,s,13,INK)
r(CX+40,470,16,16,fill=DSP,stroke="none",rx=3); t(CX+64,483,"streamed: 1–3 cycles an element",13)
t(CX+40,510,"on the CPU's RAM, while the CPU waits",13,"#6a7282")
r(CX+20,546,350,112,stroke="#2e6b40",rx=10)
t(CX+195,574,"Screen and keyboard",16,INK,"middle",True)
t(CX+40,598,"640x480 DVI: 80 × 25 characters of 8 × 16",13,INK)
r(CX+40,608,16,16,fill="#ffffff",stroke="#2f5c8a",rx=3); t(CX+64,621,"all 2,000 cells in one block RAM",13)
t(CX+40,644,"USB keyboard (HID, low speed)",13,INK)
r(CX+20,672,350,56)
t(CX+195,697,"UART console",16,INK,"middle",True)
t(CX+195,717,"1 Mbaud, alongside the screen",13,anchor="middle")
r(CX+20,744,350,104,dash="5 4")
t(CX+195,770,"clock-domain crossing",15,INK,"middle",True)
for i,s in enumerate(["one handshake for register and","main-memory accesses; the bulk","data crosses on the RAM's second port"]):
    t(CX+195,792+i*18,s,13,anchor="middle")
# engine <-> CPU
ln(CX+18,232,EX+EW+4,232); t((CX+EX+EW)/2,222,"runs",12,"#3a4250","middle")
ln(EX+EW+2,902,CX+18,902,start=True); t((CX+EX+EW)/2,892,"data",12,"#3a4250","middle")
o.append(f'<line x1="{EX+EW}" y1="902" x2="{EX+EW-18}" y2="902" stroke="#3a4250" stroke-width="2"/>')
# footer
t(40,1000,"Per token the engine streams all the weights once, 64 one-bit weights a cycle, while the CPU and the vector unit do the per-head work.",14)
t(40,1022,"Weights stay 1-bit end to end; activations are int8 with block exponents; every sum is exact.",14)
o.append('</svg>')
open('docs/img/architecture.svg','w').write('\n'.join(o))

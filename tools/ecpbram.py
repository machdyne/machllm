#!/usr/bin/env python3
# yowasp's ecpbram without its device database mounted (yowasp-nextpnr-ecp5
# leaves resources=["share"] out for it): the same, with it
import sys
import yowasp_runtime
sys.exit(yowasp_runtime.run_wasm("yowasp_nextpnr_ecp5", "ecpbram.wasm", resources=["share"],
                                 argv=["yowasp-ecpbram", *sys.argv[1:]]))

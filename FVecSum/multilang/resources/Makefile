PLATFORM = --platform /opt/xilinx/platforms/xilinx_u50_gen3x16_xdma_5_202210_1/xilinx_u50_gen3x16_xdma_5_202210_1.xpfm 
PROFILE_KERNEL = --profile.data --profile.exec

kernel:
	v++ $(PLATFORM) $(PROFILE_KERNEL) -o vecsum.xo -c vecsum.cl
	v++ $(PLATFORM) $(PROFILE_KERNEL) --save-temps --link vecsum.xo -o  vecsum.xclbin

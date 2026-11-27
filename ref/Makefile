CFLAGS ?= -O2 -g -Wall -Wextra -Wno-unused-parameter
SRC = mlm_run.c ../fw/common/mlm_model.c
mlm_run: $(SRC) ../fw/common/*.h
	$(CC) $(CFLAGS) -o $@ $(SRC)
clean:
	rm -f mlm_run

BUILD_DIR = _build

CC = gcc
CFLAGS = -O3 -w 

all: clean build
	$(CC) $(CFLAGS) -o $(BUILD_DIR)/test thread_level3.c -lpthread -static -L./ -lgemm

build:
	mkdir _build

clean:
	rm -rf _build

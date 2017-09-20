# src
# ------------------------------------------------------------------------------------------------

ASM_SOURCES := $(wildcard $(ASM_DIR)/*.S)
ASM_OBJECTS := $(patsubst %, $(BUILD_ASM_DIR)/%, $(notdir $(ASM_SOURCES:.S=.o)))

build_asm: $(ASM_OBJECTS)

$(BUILD_ASM_DIR)/%.o : $(ASM_DIR)/%.S
	@echo "$(RED)Compiling $< $(NC)"
	$(CC) $(CC_CFLAGS) -c $< -o $@


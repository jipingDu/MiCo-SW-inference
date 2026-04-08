# Make

CC = gcc # gcc, clang
OBJDUMP = objdump

MICO_DIR = MiCo-Lib

MICO_INCLUDES = $(MICO_DIR)/include $(MICO_DIR)/test
MICO_SOURCES = $(wildcard $(MICO_DIR)/src/*.c)
MICO_SOURCES += $(wildcard $(MICO_DIR)/src/mico/*.c)

OPT ?=
TARGET ?= host

CFLAGS ?=

CFLAGS += -O3
# CFLAGS += -Wall
LDFLAGS = -lm

# (Optional) Address Sanitizer
# CFLAGS += -fsanitize=address -static-libasan

# (Optional) Section GC
CFLAGS += -ffunction-sections -fdata-sections
LDFLAGS += -Wl,--gc-sections

INCLUDES = $(MICO_INCLUDES) ./

BUILD = build
MAIN = main

TEST_NUM ?= 1 # Not used by host

# RISC-V
RISCV_PREFIX = riscv64-unknown-elf
RISCV_SOURCE ?= 

# Targets
include $(MICO_DIR)/targets/common.mk
include $(MICO_DIR)/targets/$(TARGET).mk

OBJS := $(MICO_SOURCES) $(RISCV_SOURCE)
OBJS := $(OBJS:.c=.o)
OBJS := $(OBJS:.S=.o)
OBJS := $(OBJS:.s=.o)
OBJS := $(addprefix $(BUILD)/,$(OBJS))

LLAMA2_BIN ?=

ifneq ("$(LLAMA2_BIN)","")
CFLAGS += -DLLAMA2_BIN=\"$(LLAMA2_BIN)\"
endif

$(BUILD)/%.o: %.c | $(BUILD)
	@mkdir -p $(dir $@)
	@echo "Compiling Source File ($<)..."
	@$(CC) $(CFLAGS) -c -o $@ $< $(addprefix -I,$(INCLUDES))

$(BUILD)/%.o: %.S | $(BUILD)
	@mkdir -p $(dir $@)
	@echo "Compiling Source File ($<)..."
	@$(CC) $(CFLAGS) -c -o $@ $< $(addprefix -I,$(INCLUDES))

$(MAIN).elf: $(OBJS) $(MAIN).c
	@mkdir -p $(dir $@)
	@echo "Compiling Flags $(CFLAGS)"
	@echo "Linker Flags $(LDFLAGS)"
	@echo "Linking Executable (main)..."
	@$(CC) $(CFLAGS) -o $(MAIN).elf $(OBJS) $(MAIN).c $(addprefix -I,$(INCLUDES)) $(LDFLAGS)

$(MAIN).asm : $(MAIN).elf
	@echo "Generating Assembly Code..."
	@$(OBJDUMP) -d $(MAIN).elf > $(MAIN).asm

$(BUILD):
	@mkdir -p $(dir $@)
	
clean:
	@rm -rf $(BUILD)
	@rm -f *.elf *.asm
	@rm -f tests/*.elf tests/*.asm

debug: CFLAGS += -DDEBUG
debug: recompile

compile: $(MAIN).asm

recompile:
	@$(MAKE) clean
	@$(MAKE) compile


run-host: $(MAIN).elf
	./$<

all: compile

.PHONY: clean compile recompile run-host all debug
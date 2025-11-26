Needs rework for updates. NOT UP TO DATE.

# ExecuTorch CMSIS Project Template

A ready-to-fork template for deploying PyTorch models on ARM Cortex-M microcontrollers using ExecuTorch and CMSIS csolution.

## 🚀 Quick Start

### 1. Fork & Clone
```bash
git clone <your-fork-url>
cd executorch-csolution-template
```

### 2. Development Setup
- **CMSIS-Toolbox** with ARM GCC toolchain
- **VS Code** with CMSIS extension
- **Docker** (for containerized builds)


## 📁 Project Structure

```
├── ai_layer/                    # ExecuTorch runtime & model
│   ├── ai_layer.clayer.yml     # CMSIS layer definition  
│   └── engine/                 # Runtime libraries & headers
├── board/Corstone-300/         # Board support package
├── src/executor_runner/        # Application code
│   ├── arm_executor_runner.cpp # Main application
│   └── arm_memory_allocator.*  # Memory management
├── model/                      # Model conversion scripts
├── scripts/                    # Build automation
└── executorch_project.cproject.yml # Main project config
```

## 🔧 Customization

### Add Your Model
-  Replace ./model/aot_model.py with your own spec. It will be picked up from the github action from here.

### Target Different Hardware
- **Device**: Edit `board/Corstone-300/Board-U65.clayer.yml` 
- **Memory**: Adjust pools in `executorch_project.cproject.yml`
- **Optimization**: Modify compiler flags for your target

### Build Configuration
```yaml
# executorch_project.cproject.yml
define:
  - SEMIHOSTING                    # Enable for FVP simulator
  - ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE: 0x20000  # 128KB
  - ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE: 0x8000        # 32KB
```

## 🏃‍♂️ Running

### FVP Simulator
```bash
FVP_Corstone_SSE-300_Ethos-U55 \
    -C ethosu.num_macs=256 \
    -C mps3_board.uart0.out_file='-' \
    -C mps3_board.uart0.shutdown_on_eot=1 \
    -a "out/executorch_project/AVH-SSE-300-U65/Debug/executorch_project.elf"
```

## 📚 Key Files to Modify

| File | Purpose |
|------|---------|
| `src/executor_runner/arm_executor_runner.cpp` | Main application logic |
| `executorch_project.cproject.yml` | Memory settings, defines |
| `ai_layer/ai_layer.clayer.yml` | Runtime libraries |
| `board/Corstone-300/Board-U65.clayer.yml` | Target device config |

## 🐳 Docker Development Local
(to do)

Build the container:
```bash
./build-docker-local.sh --rebuild
```

Develop inside container:
```bash
docker run -it --rm -v $(pwd):/workspace executorch-arm-builder:latest
```

## 📖 References

- [ExecuTorch Documentation](https://pytorch.org/executorch/)
- [CMSIS-Toolbox](https://github.com/Open-CMSIS-Pack/cmsis-toolbox)
- [ARM Virtual Hardware](https://www.arm.com/products/development-tools/simulation/virtual-hardware)


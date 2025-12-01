

# ExecuTorch CMSIS Project Template

A ready-to-fork template for deploying PyTorch models on ARM Cortex-M microcontrollers using ExecuTorch and CMSIS csolution.

# Quick Start

This example is ready to run in Visual Studio Code. Follow these steps to get started:

1. Install **Keil Studio for VS Code** from the Visual Studio Code Marketplace.  
2. On the repository’s landing page, select **Use this Template**.  
3. Enter a name for your new repository to identify your project.  

Once the repository is created from the template, it will automatically start building the Docker-based CI environment. You can check the status in the Actions tab. This process can take up to an hour. It is not required to finish before you continue the Quick Start, but will be necessary for further customization. 

4. Clone the repository to your desktop. You can use the built-in [Git tools in VS Code](https://code.visualstudio.com/docs/sourcecontrol/intro-to-git) or the more approachable [GitHub Desktop](https://desktop.github.com/download/) if you’re new to Git.  
5. Open the cloned folder in Visual Studio Code.  
6. Visual Studio Code automatically downloads the required tools and software packs. You can monitor progress in **View > Output > CMSIS Solution**.  
7. Open the [CMSIS View](https://mdk-packs.github.io/vscode-cmsis-solution-docs/userinterface.html#2-main-area-of-the-cmsis-view).  
8. In the CMSIS View, use the [Action buttons](https://github.com/ARM-software/vscode-cmsis-csolution?tab=readme-ov-file#action-buttons) to build, load, and run the example on the preconfigured FastModel simulation.

## Prerequisites

Before you begin, ensure you have the following tools installed and a basic understanding of the concepts involved:

| Technology | Requirement | Purpose |
| --- | --- | --- |
| Github | Basic proficiency | Version control and repository management. |
| Docker | Installed and running | Building the AI layer in a containerized environment. |


## Making the Template Your Own

To get started, you will first need to create your own copy of the template repository.

1.  **Fork the Repository**: Click the "Use this template" button on the main repository page to create a new repository under your own account. This will give you a personal copy that you can modify freely.

2.  **Clone Your Repository**: Once you have created your own repository, clone it to your local machine:

    ```bash
    git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git
    cd YOUR-REPOSITORY-NAME
    ```

3.  **Explore the Structure**: Familiarize yourself with the repository's structure. The key directories and files are explained in the next section.

## Repository Structure

The repository is organized into several directories, each with a specific purpose. Understanding this structure is key to customizing the template for your own project.

| Directory / File | Purpose |
| --- | --- |
| `.docker/` | Contains the `Dockerfile` for building the containerized environment. This environment includes all the necessary tools and dependencies to build the ExecuTorch AI layer. |
| `.github/workflows/` | Defines the GitHub Actions workflows for continuous integration (CI). The `build_ai_layer.yml` workflow is triggered on changes to the `model/` directory, automatically building the AI layer. |
| `board/Corstone-300/` | Contains the board-specific files for the Arm Corstone-300. This is where you would add support for a different target board. |
| `model/` | This is where you will place your own PyTorch model. The template includes an example model to get you started. |
| `scripts/` | A collection of shell and Python scripts that automate the build process. These scripts are used by the CI workflow and can also be run locally. |
| `src/executor_runner/` | Contains the source code for the example application that runs the ExecuTorch model. |
| `executorch_project.cproject.yml` | The CMSIS project file. It defines the project structure and components. |
| `executorch_project.csolution.yml` | The CMSIS solution file. It ties together the project, board, and build configurations. |

## Customizing the Model

The heart of this template is the ability to deploy your own PyTorch model. The `model/` directory is where you will make your changes.

### 1. Replace the Example Model

The template comes with an example model in `model/aot_model.py`. To use your own model, you will need to:

1.  **Create your model**: Define your model as a `torch.nn.Module` in a Python file. You can either replace the contents of `aot_model.py` or create a new file and update the `local_workflow.sh` and `.github/workflows/build_ai_layer.yml` files to point to your new script.

2.  **Define your model architecture**: The `aot_model.py` script contains the model definition, quantization settings, and the compilation spec for the Ethos-U NPU. You will need to adapt this script to your own model's architecture and requirements.

### 2. Configure Operator Selection

To minimize the footprint of the ExecuTorch runtime, it is crucial to only include the operators that your model actually uses. The `build_stage2_selective.sh` script handles this, and you have two options for configuring it:

*   **Automatic Selection**: If you provide a `.pte` model file to the script, it will automatically analyze the model and determine the required operators. This is the recommended approach for most users.

*   **Manual Selection**: For more fine-grained control, you can create a file named `operators_minimal.txt` in the `model/` directory. This file should contain a list of the operators your model needs, one per line. The `local_workflow.sh` and `build_ai_layer.yml` files are already configured to use this file if it exists.

### 3. The `aot_model.py` Script

This script is responsible for converting your PyTorch model into the ExecuTorch format (`.pte`). It performs the following key steps:

*   **Quantization**: The script uses the Ethos-U quantizer to quantize the model for optimal performance on the NPU.
*   **Compilation**: It then uses the Ethos-U compiler to generate a compiled model that can be executed by the ExecuTorch runtime.
*   **Saving the Model**: Finally, it saves the compiled model as a `.pte` file.

You will need to modify this script to match your model's specific layers and quantization requirements.

## Customizing the Board Layer

The template is pre-configured for the Arm Corstone-300, but it is designed to be easily adapted to other target boards. The `board/` directory is where you will make these changes.

### 1. Add a New Board Directory

To add support for a new board, create a new directory under `board/` with the name of your target board. For example:

```
board/
├── Corstone-300/
└── YOUR_BOARD_NAME/
```

### 2. Add Board-Specific Files

Inside your new board directory, you will need to add the necessary board-specific files. 

### 3. Update the CMSIS Solution File

Finally, you will need to update the `executorch_project.csolution.yml` file to point to your new board. This file defines the build contexts for the project. You will need to add a new context for your board, specifying the target device and the path to your board-specific files.

For example:

```yaml
solution:
  # ...

  build-types:
    # ...

  target-types:
    - type: Corstone-300
      device: ARM::SSE-300-MPS3
    - type: YOUR_BOARD_NAME
      device: YOUR_DEVICE_NAME

  projects:
    - project: ./executorch_project.cproject.yml
      # ...
```

You will also need to update the `cproject` file to specify the correct board and device for your new build context.

## Continuous Integration with GitHub Actions

The template includes a powerful CI/CD pipeline based on GitHub Actions. This pipeline automates the process of building the AI layer whenever you make changes to your model.

### How it Works

The CI workflow is defined in the `.github/workflows/build_ai_layer.yml` file. Here's a breakdown of how it works:

1.  **Trigger**: The workflow is automatically triggered whenever you push changes to the `model/` directory. You can also trigger it manually from the Actions tab in your GitHub repository.

2.  **Docker Environment**: The workflow first checks if a Docker image for the build environment already exists in your repository's package registry. If not, it automatically builds the Docker image using the `Dockerfile` in the `.docker/` directory and pushes it to the registry. This ensures that the build environment is always up-to-date and consistent.

3.  **Build AI Layer**: Once the Docker environment is ready, the workflow runs the `local_workflow.sh` script inside the container. This script performs all the necessary steps to build the AI layer, as described in the "Local Development" section.

4.  **Commit Artifacts**: After the build is complete, the workflow automatically commits the generated artifacts (libraries, headers, report, and logs) back to your repository. This ensures that your repository always contains the latest version of the AI layer, ready to be integrated into your application.

### Viewing the Results

You can monitor the progress of the CI workflow from the Actions tab in your GitHub repository. When the workflow is complete, you will see a new commit in your repository containing the updated AI layer. The commit message will include the build timestamp and a summary of the changes.

You can also view the detailed build report (`REPORT.md`) and logs in the `ai_layer/` directory to get a comprehensive overview of the build process.

## References

[1] [Arm CMSIS Documentation](https://arm-software.github.io/CMSIS_6/latest/index.html)


## Understanding the Build Scripts

The `/scripts` directory contains a set of shell and Python scripts that automate the process of building the ExecuTorch AI layer. These scripts are orchestrated by the `local_workflow.sh` script and the GitHub Actions workflow.

### `build_stage2_selective.sh`

This script is responsible for building the model-selective portable operator registration library (`portable_ops_lib`). This is a crucial step in optimizing the ExecuTorch runtime for a specific model by including only the necessary operators.

**Purpose**

The primary goal of this script is to re-configure and build ExecuTorch with a selective set of operators. This can be done in two ways:

1.  **Model-based selection**: The script analyzes a given `.pte` model file to automatically determine the required operators.
2.  **Manual selection**: The script uses a predefined list of operators from a file or an environment variable.

**Usage**

```bash
./scripts/build_stage2_selective.sh <executorch_src> [model.pte] [build_dir] [toolchain.cmake] [ops_list.txt]
```

**Arguments**

| Argument | Description |
| --- | --- |
| `<executorch_src>` | The path to the ExecuTorch source directory. |
| `[model.pte]` | (Optional) The path to the `.pte` model file for model-based operator selection. |
| `[build_dir]` | (Optional) The directory where the build artifacts will be stored. Defaults to `build/stage2`. |
| `[toolchain.cmake]` | (Optional) The path to the CMake toolchain file for cross-compilation. |
| `[ops_list.txt]` | (Optional) The path to a file containing a list of operators for manual selection. |

**Environment Variables**

| Variable | Description |
| --- | --- |
| `EXECUTORCH_EXTRA_CMAKE_ARGS` | Space-separated extra CMake arguments to append to the build configuration. |
| `EXECUTORCH_SELECT_OPS` | A comma-separated list of operators to be included in the build. This overrides the model-based and file-based selection methods. |

**Logging**

The script creates a timestamped log directory in `/workspace2/ai_layer/logs` inside the Docker container. Each step of the workflow has its own log file, and a main log file aggregates the output of all steps.

### `package_sdk.sh`

This script is responsible for packaging the ExecuTorch SDK artifacts into a clean and portable structure. It collects the necessary headers, libraries, and optionally the model file, making it easy to integrate the AI layer into an external application.

**Purpose**

The script creates a self-contained SDK directory with the following structure:

```
<output_dir>/
├── include/            # ExecuTorch headers
├── lib/                # Core & kernel libs (+ optional selective portable ops lib)
├── model/              # Model .pte (if provided)
└── meta/selected_operators.yaml (if available)
```

**Usage**

```bash
./scripts/package_sdk.sh <stage1_assets_dir> [stage2_assets_dir] [model.pte] [output_dir]
```

**Arguments**

| Argument | Description |
| --- | --- |
| `<stage1_assets_dir>` | The directory containing the assets from the Stage 1 build (core libraries and headers). |
| `[stage2_assets_dir]` | (Optional) The directory containing the assets from the Stage 2 build (selective operator library). |
| `[model.pte]` | (Optional) The path to the `.pte` model file to be included in the SDK. |
| `[output_dir]` | (Optional) The directory where the SDK will be created. Defaults to `dist/sdk`. |

### `pte_to_header.py`

This Python script converts a `.pte` (PyTorch ExecuTorch) model file into a C header file. This allows the model to be embedded directly into the source code of a C/C++ application.

**Purpose**

The main purpose of this script is to transform the binary `.pte` model into a C-style character array. This array can then be compiled into the application, making the model data available at runtime without needing to load it from a separate file.

**Usage**

```bash
python3 scripts/pte_to_header.py -p <model.pte> -d <output_dir> -o <output_file> -s <section_name>
```

**Arguments**

| Argument | Description |
| --- | --- |
| `-p`, `--pte` | The path to the input `.pte` model file. (Required) |
| `-d`, `--outdir` | The output directory for the generated header file. (Default: current directory) |
| `-o`, `--outfile` | The name of the output header file. (Default: `model_pte.h`) |
| `-s`, `--section` | The section attribute for the data array in the generated header file. (Default: `network_model_sec`) |

### `generate_ai_layer_report.py`

This Python script generates a comprehensive report about the AI layer build. The report includes information about the build environment, library assets, model details, and more.

**Purpose**

The script collects and presents a wide range of information to provide a clear overview of the AI layer's composition and build process. This is useful for debugging, optimization, and documentation purposes.

**Report Contents**

The generated report (`REPORT.md`) includes the following sections:

*   **Build Summary**: A high-level overview of the build, including the number of libraries, models, and operators.
*   **Library Assets**: A detailed list of all static libraries in the AI layer, including their size, modification date, and hash.
*   **Model Assets**: Information about the model files, including the `.pte` model and the generated header file.
*   **Build Configuration**: Details about the CMake build configuration, such as the build type, toolchain file, and ExecuTorch build options.
*   **Model Conversion Log**: Information extracted from the model conversion script, including the Ethos-U compile specification and quantization configuration.
*   **Build Environment**: Details about the build environment, including the Python, CMake, and GCC versions.



### `local_workflow.sh` (Optionally for local docker deployment)

This script simulates the GitHub Actions workflow on a local machine. It provides a convenient way to build the entire AI layer without relying on the CI/CD pipeline.

**Purpose**

The script automates the following steps:

1.  **Model Conversion**: Converts the user's model to the ExecuTorch format.
2.  **Build ExecuTorch Core Libraries**: Builds the core ExecuTorch libraries (Stage 1).
3.  **Build ExecuTorch Selective Kernel Libraries**: Builds the selective kernel libraries based on the user's model (Stage 2).
4.  **Collect Artifacts**: Packages the build artifacts (libraries, headers, etc.) into a single directory.
5.  **Convert Model to Header**: Converts the `.pte` model to a C header file.
6.  **Generate AI Layer Report**: Creates a comprehensive report of the build process.

**Usage**

Running the container locally requires a docker environment to be installed on your computer. Installing [Docker Desktop](https://www.docker.com/products/docker-desktop/) is the easiest way to manage the setup.

Two VSC tasks make the use of the docker environment easy. Find them in **Terminal - Run Task**

 "Docker: Build ExecuTorch ARM Builder Image" : This builds the image and only needs to be run once, or whenever updates to .docker/Dockerfile are made.
 "Docker: Run ExecuTorch Build" : This runs the local_workflow.sh in an instance of the docker image. All output is immidiately generated into ./out and ./ai_layer.

You can also open an interactive instance with the following command, opening a bash shell inside the container:

```cmd
 docker run -it --rm -v $(pwd):/workspace2 executorch-arm-builder:latest 
 ```






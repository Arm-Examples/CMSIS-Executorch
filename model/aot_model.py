import torch


class InnerAdd(torch.nn.Module):
    """The add operation — this submodule gets quantized and delegated to Ethos-U."""
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class AddWithPostProcessing(torch.nn.Module):
    """
    Model that combines an Ethos-U-delegated add with CPU-side portable ops
    for testing pack-based operator component integration.

    Only the InnerAdd submodule is quantized (via set_module_name).
    The post-processing ops stay as float and run on the CPU as portable operators:
      - view_copy        (reshape the output)
      - mul              (scale the values)
      - add              (apply bias)
      - sigmoid          (squash to [0, 1])
      - unsqueeze_copy   (add batch dimension)
      - softmax          (normalize to probability distribution)
    """
    def __init__(self):
        super().__init__()
        self.inner_add = InnerAdd()
        # Float scale and bias for post-processing (not quantized)
        self.register_buffer('scale', torch.tensor([2.0]))
        self.register_buffer('bias', torch.tensor([0.5]))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # This add is quantized and delegated to Ethos-U NPU
        z = self.inner_add(x, y)
        # --- CPU portable ops (float, not delegated) ---
        z = z.view(1)                     # aten::view_copy.out
        z = z * self.scale                # aten::mul.out
        z = z + self.bias                 # aten::add.out
        z = torch.sigmoid(z)             # aten::sigmoid.out
        z = z.unsqueeze(0)               # aten::unsqueeze_copy.out
        z = torch.softmax(z, dim=-1)     # aten::_softmax.out
        return z

example_inputs = (torch.ones(1,1,1,1),torch.ones(1,1,1,1))

model = AddWithPostProcessing()
model = model.eval()
exported_program = torch.export.export(model, example_inputs)
graph_module = exported_program.module()

_ = graph_module.print_readable()

from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.quantizer import (
    EthosUQuantizer,
    get_symmetric_quantization_config,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

# Create a compilation spec describing the target for configuring the quantizer
# Some args are used by the Arm Vela graph compiler later in the example. Refer to Arm Vela documentation for an
# explanation of its flags: https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/blob/main/OPTIONS.md
compile_spec = EthosUCompileSpec(
            target="ethos-u55-128",
            system_config="Ethos_U55_High_End_Embedded",
            memory_mode="Shared_Sram",
            extra_flags=["--output-format=raw", "--debug-force-regor", "--verbose-all"],
        )

# Only quantize the inner_add submodule — post-processing stays as float CPU ops
quantizer = EthosUQuantizer(compile_spec)
operator_config = get_symmetric_quantization_config()
quantizer.set_module_name("inner_add", operator_config)

# Post training quantization
quantized_graph_module = prepare_pt2e(graph_module, quantizer)
quantized_graph_module(*example_inputs) # Calibrate the graph module with the example input
quantized_graph_module = convert_pt2e(quantized_graph_module)

_ = quantized_graph_module.print_readable()

# Create a new exported program using the quantized_graph_module
quantized_exported_program = torch.export.export(quantized_graph_module, example_inputs)

from executorch.backends.arm.ethosu import EthosUPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.extension.export_util.utils import save_pte_program

# Create partitioner from compile spec
partitioner = EthosUPartitioner(compile_spec)

# Lower the exported program to the Ethos-U backend
edge_program_manager = to_edge_transform_and_lower(
            quantized_exported_program,
            partitioner=[partitioner],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
            ),
        )

# Convert edge program to executorch
executorch_program_manager = edge_program_manager.to_executorch(
            config=ExecutorchBackendConfig(extract_delegate_segments=False)
        )

_ = executorch_program_manager.exported_program().module().print_readable()

# Save pte file
save_pte_program(executorch_program_manager, "ethos_u_minimal_example.pte")
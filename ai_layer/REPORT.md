# AI Layer Report

## Selected Operators

### Portable Operators (CPU)

| Operator | Pack Component |
|----------|---------------|
| `aten::add` | `Machine Learning:ExecuTorch:Operators Portable add` |
| `aten::exp` | `Machine Learning:ExecuTorch:Operators Portable exp` |
| `aten::mul` | `Machine Learning:ExecuTorch:Operators Portable mul` |
| `aten::reciprocal` | `Machine Learning:ExecuTorch:Operators Portable reciprocal` |
| `aten::sigmoid` | `Machine Learning:ExecuTorch:Operators Portable sigmoid` |
| `aten::sum` | `Machine Learning:ExecuTorch:Operators Portable sum` |
| `aten::unsqueeze_copy` | `Machine Learning:ExecuTorch:Operators Portable unsqueeze_copy` |

### Quantized Operators (NPU wrapper)

| Operator | Pack Component |
|----------|---------------|
| `quantized_decomposed::dequantize_per_tensor` | `Machine Learning:ExecuTorch:Operators Quantized dequantize` |
| `quantized_decomposed::quantize_per_tensor` | `Machine Learning:ExecuTorch:Operators Quantized quantize` |

**Total:** 7 portable + 2 quantized = 9 operator components

---

## Vela Conversion Log

### TOSA Graph — Before Optimisation

```
0     Const                tosa_rescale_default_2_output_zp
1     Const                tosa_rescale_default_2_input_zp
2     Const                tosa_rescale_default_2_shifts 
3     Const                tosa_rescale_default_2_multipliers
4     Const                tosa_rescale_default_1_output_zp
5     Const                tosa_rescale_default_1_input_zp
6     Const                tosa_rescale_default_1_shifts 
7     Const                tosa_rescale_default_1_multipliers
8     Transpose            tosa_transpose_default_1      
9     Rescale              tosa_rescale_default_1        
10    Const                tosa_rescale_default_output_zp
11    Const                tosa_rescale_default_input_zp 
12    Const                tosa_rescale_default_shifts   
13    Const                tosa_rescale_default_multipliers
14    Transpose            tosa_transpose_default        
15    Rescale              tosa_rescale_default          
16    Add                  aten_add_tensor               
17    Rescale              tosa_rescale_default_2        
18    Transpose            tosa_transpose_default_2
```

### TOSA Graph — After Optimisation

```
0     Transpose            tosa_transpose_default_1      
1     Add                  tosa_transpose_default_1_int32
2     Mul                  tosa_rescale_default_1        
3     Transpose            tosa_transpose_default        
4     Add                  tosa_transpose_default_int32  
5     Mul                  tosa_rescale_default          
6     Add                  tosa_rescale_default_2        
7     Transpose            tosa_transpose_default_2
```

### NPU Performance Summary

```
Original Operator    NNG Operator         Target Staging Usage  Peak% (Staging)  Op Cycles Network% (cycles)        NPU    SRAM AC    DRAM AC OnFlash AC OffFlash AC  MAC Count Network% (MAC)  Util% (MAC) Name                 
-------------------- -------------------- ------ ------------- ---------------- ---------- ----------------- ---------- ---------- ---------- ---------- ----------- ---------- -------------- ------------ -------------------- 
Transpose            Transpose            NPU               48            33.33         32              9.76         32          1          0          0           0          0         100.00         0.00 #70                  
Transpose            Transpose            NPU               48            33.33         32              9.76         32          1          0          0           0          0         100.00         0.00 tosa_transpose_default_1 
Rescale              Add                  NPU               96            66.67         34             10.37         34          2          0          0          16          0         100.00         0.00 tosa_transpose_default_1_int32 
Rescale              Mul                  NPU               80            55.56         33             10.06         33          4          0          0           0          0         100.00         0.00 tosa_rescale_default_1 
Transpose            Transpose            NPU               96            66.67         32              9.76         32          1          0          0           0          0         100.00         0.00 #85                  
Transpose            Transpose            NPU               96            66.67         32              9.76         32          1          0          0           0          0         100.00         0.00 tosa_transpose_default 
Rescale              Add                  NPU              144           100.00         34             10.37         34          2          0          0          16          0         100.00         0.00 tosa_transpose_default_int32 
Rescale              Mul                  NPU              128            88.89         33             10.06         33          4          0          0           0          0         100.00         0.00 tosa_rescale_default 
Add                  Add                  NPU              144           100.00          2              0.61          1          2          0          0           0          0         100.00         0.00 tosa_rescale_default_2 
Transpose            Transpose            NPU               32            22.22         32              9.76         32          1          0          0           0          0         100.00         0.00 #101                 
Transpose            Transpose            NPU               32            22.22         32              9.76         32          1          0          0           0          0         100.00         0.00 tosa_transpose_default_2
```

### Network Summary

```
Accelerator configuration               Ethos_U55_128
System configuration             Ethos_U55_High_End_Embedded
Memory mode                               Shared_Sram
Accelerator clock                                 500 MHz
Design peak SRAM bandwidth                       3.73 GB/s
Design peak Off-chip Flash bandwidth             0.47 GB/s

Total SRAM used                                  0.14 KiB
Total Off-chip Flash used                        0.03 KiB

CPU operators = 0 (0.0%)
NPU operators = 11 (100.0%)

Average SRAM bandwidth                           0.24 GB/s
Input   SRAM bandwidth                           0.00 MB/batch
Weight  SRAM bandwidth                           0.00 MB/batch
Output  SRAM bandwidth                           0.00 MB/batch
Total   SRAM bandwidth                           0.00 MB/batch
Total   SRAM bandwidth            per input      0.00 MB/inference (batch size 1)

Average Off-chip Flash bandwidth                 0.05 GB/s
Input   Off-chip Flash bandwidth                 0.00 MB/batch
Weight  Off-chip Flash bandwidth                 0.00 MB/batch
Output  Off-chip Flash bandwidth                 0.00 MB/batch
Total   Off-chip Flash bandwidth                 0.00 MB/batch
Total   Off-chip Flash bandwidth  per input      0.00 MB/inference (batch size 1)

Original Weights Size                            0.00 KiB
NPU Encoded Weights Size                         0.00 KiB

Neural network macs                                 0 MACs/batch

Info: The numbers below are internal compiler estimates.
For performance numbers the compiled network should be run on an FVP Model or FPGA.

Network Tops/s                                   0.00 Tops/s

NPU cycles                                        327 cycles/batch
SRAM Access cycles                                 20 cycles/batch
DRAM Access cycles                                  0 cycles/batch
On-chip Flash Access cycles                         0 cycles/batch
Off-chip Flash Access cycles                       32 cycles/batch
Total cycles                                      328 cycles/batch

Batch Inference time                 0.00 ms, 1524390.24 inferences/s (batch size 1)
```

### Final Exported Program Graph

```python
class GraphModule(torch.nn.Module):
    def forward(self, x, y):
        x: "f32[1, 1, 1, 1]"; y: "f32[1, 1, 1, 1]"; 
    
        x, y, = fx_pytree.tree_flatten_spec(([x, y], {}), self._in_spec)
        # No stacktrace found for following nodes
        _tensor_constant0: "f32[1]" = self._tensor_constant0
        _tensor_constant1: "f32[1]" = self._tensor_constant1
        alloc: "i8[1, 1, 1, 1]" = executorch_exir_memory_alloc(((1, 1, 1, 1), torch.int8))
        quantized_decomposed_quantize_per_tensor_default: "i8[1, 1, 1, 1]" = torch.ops.quantized_decomposed.quantize_per_tensor.out(x, 0.003921568859368563, -128, -128, 127, torch.int8, out = alloc);  x = alloc = None
        alloc_1: "i8[1, 1, 1, 1]" = executorch_exir_memory_alloc(((1, 1, 1, 1), torch.int8))
        quantized_decomposed_quantize_per_tensor_default_1: "i8[1, 1, 1, 1]" = torch.ops.quantized_decomposed.quantize_per_tensor.out(y, 0.003921568859368563, -128, -128, 127, torch.int8, out = alloc_1);  y = alloc_1 = None
        lowered_module_0 = self.lowered_module_0
        executorch_call_delegate = torch.ops.higher_order.executorch_call_delegate(lowered_module_0, quantized_decomposed_quantize_per_tensor_default, quantized_decomposed_quantize_per_tensor_default_1);  lowered_module_0 = quantized_decomposed_quantize_per_tensor_default = quantized_decomposed_quantize_per_tensor_default_1 = None
        getitem: "i8[1, 1, 1, 1]" = executorch_call_delegate[0];  executorch_call_delegate = None
        alloc_2: "f32[1, 1, 1, 1]" = executorch_exir_memory_alloc(((1, 1, 1, 1), torch.float32))
        quantized_decomposed_dequantize_per_tensor_default: "f32[1, 1, 1, 1]" = torch.ops.quantized_decomposed.dequantize_per_tensor.out(getitem, 0.007843137718737125, -128, -128, 127, torch.int8, out = alloc_2);  getitem = alloc_2 = None
        
        # File: /workspace2/model/aot_model.py:35 in forward, code: z = z.view(1)                     # aten::view_copy.out
        aten_view_copy_default: "f32[1]" = executorch_exir_memory_view(quantized_decomposed_dequantize_per_tensor_default, [1]);  quantized_decomposed_dequantize_per_tensor_default = None
        
        # No stacktrace found for following nodes
        alloc_3: "f32[1]" = executorch_exir_memory_alloc(((1,), torch.float32))
        
        # File: /workspace2/model/aot_model.py:36 in forward, code: z = z * self.scale                # aten::mul.out
        aten_mul_tensor: "f32[1]" = torch.ops.aten.mul.out(aten_view_copy_default, _tensor_constant0, out = alloc_3);  aten_view_copy_default = _tensor_constant0 = alloc_3 = None
        
        # No stacktrace found for following nodes
        alloc_4: "f32[1]" = executorch_exir_memory_alloc(((1,), torch.float32))
        
        # File: /workspace2/model/aot_model.py:37 in forward, code: z = z + self.bias                 # aten::add.out
        aten_add_tensor: "f32[1]" = torch.ops.aten.add.out(aten_mul_tensor, _tensor_constant1, out = alloc_4);  aten_mul_tensor = _tensor_constant1 = alloc_4 = None
        
        # No stacktrace found for following nodes
        alloc_5: "f32[1]" = executorch_exir_memory_alloc(((1,), torch.float32))
        
        # File: /workspace2/model/aot_model.py:38 in forward, code: z = torch.sigmoid(z)             # aten::sigmoid.out
        aten_sigmoid_default: "f32[1]" = torch.ops.aten.sigmoid.out(aten_add_tensor, out = alloc_5);  aten_add_tensor = alloc_5 = None
        
        # No stacktrace found for following nodes
        alloc_6: "f32[1, 1]" = executorch_exir_memory_alloc(((1, 1), torch.float32))
        
        # File: /workspace2/model/aot_model.py:39 in forward, code: z = z.unsqueeze(0)               # aten::unsqueeze_copy.out
        aten_unsqueeze_copy_default: "f32[1, 1]" = torch.ops.aten.unsqueeze_copy.out(aten_sigmoid_default, 0, out = alloc_6);  aten_sigmoid_default = alloc_6 = None
        
        # No stacktrace found for following nodes
        alloc_7: "f32[1, 1]" = executorch_exir_memory_alloc(((1, 1), torch.float32))
        
        # File: /workspace/executorch-venv/lib/python3.13/site-packages/executorch/backends/arm/_passes/decompose_softmax_unstable_pass.py:80 in call_operator, code: op1 = super().call_operator(exp_op, (_input,), {}, meta, True)
        aten_exp_default: "f32[1, 1]" = torch.ops.aten.exp.out(aten_unsqueeze_copy_default, out = alloc_7);  aten_unsqueeze_copy_default = alloc_7 = None
        
        # No stacktrace found for following nodes
        alloc_8: "f32[1, 1]" = executorch_exir_memory_alloc(((1, 1), torch.float32))
        
        # File: /workspace/executorch-venv/lib/python3.13/site-packages/executorch/backends/arm/_passes/decompose_softmax_unstable_pass.py:81 in call_operator, code: op2 = super().call_operator(sum_op, (op1, dim, True), {}, meta, True)
        aten_sum_dim_int_list: "f32[1, 1]" = torch.ops.aten.sum.IntList_out(aten_exp_default, [-1], True, out = alloc_8);  alloc_8 = None
        
        # No stacktrace found for following nodes
        alloc_9: "f32[1, 1]" = executorch_exir_memory_alloc(((1, 1), torch.float32))
        
        # File: /workspace/executorch-venv/lib/python3.13/site-packages/executorch/backends/arm/_passes/decompose_softmax_unstable_pass.py:82 in call_operator, code: op3 = super().call_operator(reciprocal_op, (op2,), {}, meta, True)
        aten_reciprocal_default: "f32[1, 1]" = torch.ops.aten.reciprocal.out(aten_sum_dim_int_list, out = alloc_9);  aten_sum_dim_int_list = alloc_9 = None
        
        # No stacktrace found for following nodes
        alloc_10: "f32[1, 1]" = executorch_exir_memory_alloc(((1, 1), torch.float32))
        
        # File: /workspace/executorch-venv/lib/python3.13/site-packages/executorch/backends/arm/_passes/decompose_softmax_unstable_pass.py:83 in call_operator, code: op4 = super().call_operator(mul_op, (op1, op3), {}, meta, True)
        aten_mul_tensor_1: "f32[1, 1]" = torch.ops.aten.mul.out(aten_exp_default, aten_reciprocal_default, out = alloc_10);  aten_exp_default = aten_reciprocal_default = alloc_10 = None
        return pytree.tree_unflatten((aten_mul_tensor_1,), self._out_spec)
```

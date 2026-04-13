
Arithmetic intensity

FLOPs in my kernel for arithmentic operations count

maybe need to use torchinfo

estimate bytes transferred assuming all operands are loaded from DRAM


|        Function         | tottime (s) | cumtime (s)|            What it is                 |
|----------------------- |-------------|------------|-----------------------------------------
| `TFE_Py_Execute`        | **14.797**  | 14.797     | Actual kernel execution (Dense matmul)|
| `TFE_Py_FastPathExecute`| **4.496**   | 4.554      | Fast-path Dense forward ops           |
| `c_parser_wrapper.read` | 1.487       | 1.530      | CSV load (not compute)                |
| `optional_ops.get_value`| 0.110       | 3.064      | Data pipeline fetching                |


**FLOPS**

Batch size = 256 ; Input features = 29 ; Output units = 20

Bias = Batch size * Hout = 5,120 FLOPs
ReLU = Batch size * Hout = 5,120 FLOPs

FLOPs = 2 * Batch size * Din * Dout + Bias + ReLu
     = 2 * 256 * 29 * 20 + 5120 + 5120 = 307,200 FLOPs/batch
     
Since there are three layers/batch, total FLOPs = 307,200 * 3 = 537,600 FLOPa/batch
     
     


**BYTES**

Input batch X = 4B * Batch size * Din = 256 * 29 = 29,696B
Weight matrix = 4B * Din * Dout = 2320B
Bias = Dout * 4B = 80 B
Hout = Batch size * Dout * 4B = 20480 B

Num of bytes = Input batch + Weight matrix + Bias + H_out
            =   29,696 + 2,320 + 80 + 20,480 = 52,576 B

**Arithmetic Intensity**

AI = FLOPs / Bytes = 537,600/52,576 B = 5.84 FLOPs/Byte






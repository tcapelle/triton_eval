import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.utils.data
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_mul_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4), (4, 1))
    assert_size_stride(primals_2, (4,), (1,))
    assert_size_stride(primals_3, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(primals_3, (64, 4), (4, 1), 0),
            reinterpret_tensor(primals_1, (4, 4), (1, 4), 0), out=buf0)
        del primals_1
        buf1 = reinterpret_tensor(buf0, (4, 4, 4, 4), (64, 16, 4, 1), 0)
        del buf0
        get_raw_stream(0)
        triton_poi_fused_mul_0[grid(256)](buf1, primals_2, 256, XBLOCK=256,
            num_warps=4, num_stages=1)
        del primals_2
        return buf1, reinterpret_tensor(primals_3, (64, 4), (4, 1), 0)


class LinearEmbeddingNew(nn.Module):

    def __init__(self, inp_size, d_model):
        super(LinearEmbeddingNew, self).__init__()
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, input_0):
        primals_1 = self.lut.weight
        primals_2 = self.lut.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
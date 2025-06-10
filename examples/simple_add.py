import tvm
from tvm import te
import numpy as np

def vector_add(n):
    """
    A simple vector addition example using modern TVM API.
    """
    # 1. Define the computation
    A = te.placeholder((n,), name='A')
    B = te.placeholder((n,), name='B')
    C = te.compute(A.shape, lambda i: A[i] + B[i], name='C')

    # 2. Create a PrimFunc using modern TVM API
    # This replaces the old schedule-based approach
    prim_func = te.create_prim_func([A, B, C])

    # 3. Compile the function
    # The target 'llvm' means we are compiling for the CPU.
    fadd = tvm.build(prim_func, target='llvm')

    # 4. Run the compiled function
    a = np.random.uniform(size=(n,)).astype(A.dtype)
    b = np.random.uniform(size=(n,)).astype(B.dtype)
    c = tvm.nd.array(np.zeros((n,), dtype=C.dtype))

    fadd(tvm.nd.array(a), tvm.nd.array(b), c)

    # 5. Verify the results
    tvm_result = c.numpy()
    numpy_result = a + b
    
    np.testing.assert_allclose(tvm_result, numpy_result, rtol=1e-5)
    
    print("TVM and NumPy results match!")
    print("TVM result:", tvm_result[:10], "..." if n > 10 else "")
    print("NumPy result:", numpy_result[:10], "..." if n > 10 else "")

if __name__ == "__main__":
    vector_add(128) 
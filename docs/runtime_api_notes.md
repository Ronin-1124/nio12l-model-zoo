# Neuron Runtime API notes

The main runtime API header is:

```cpp
#include <neuron/api/RuntimeV2.h>
```

Important functions:

```cpp
NeuronRuntimeV2_create
NeuronRuntimeV2_createFromBuffer
NeuronRuntimeV2_getInputNumber
NeuronRuntimeV2_getOutputNumber
NeuronRuntimeV2_getInputSize
NeuronRuntimeV2_getOutputSize
NeuronRuntimeV2_run
NeuronRuntimeV2_release
```

Basic flow:

```text
1. Create runtime from .dla
2. Query input/output number
3. Query input/output buffer sizes
4. Prepare IOBuffer arrays
5. Run synchronous inference
6. Read output buffers
7. Release runtime
```

Important note:
For normal user-space memory, IOBuffer.fd must be -1.

Example:

```cpp
IOBuffer input;
memset(&input, 0, sizeof(input));
input.buffer = input_ptr;
input.length = input_size;
input.fd = -1;
input.offset = 0;
```

If fd is uninitialized, Runtime may treat it as an imported buffer fd and fail with errors such as:

```bash
memImport: dup fd fail
APUSysEngine::MemImport() failed
```

For the C++ YOLOv5s demo in this repository, the RuntimeV2 flow is:

```text
1. NeuronRuntimeV2_create(model_path, 1, &runtime, 1)
2. NeuronRuntimeV2_getInputNumber / getOutputNumber
3. NeuronRuntimeV2_getInputSize / getOutputSize
4. Fill IOBuffer for input and output, with fd = -1
5. NeuronRuntimeV2_run(runtime, request)
6. Read raw output buffer and do postprocess in user code
7. NeuronRuntimeV2_release(runtime)
```
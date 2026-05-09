# Model conversion

TFLite models can be converted to DLA format using `ncc-tflite`.

For edge deployment, INT8 is recommended as the default path because it provides
better performance and smaller model size on-device.

Example for INT8 model:

```bash
ncc-tflite --arch=mdla2.0 -d model_int8.dla model_int8.tflite
```

Example for FP32 model:

```bash
ncc-tflite --arch=mdla2.0 -d model_fp32.dla model_fp32.tflite --relax-fp32
```

Run ncc-tflite -h to see detailed help imformation.
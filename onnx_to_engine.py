import os
import tensorrt as trt

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
TRT_LOGGER = trt.Logger()
onnx_file_path = 'sanjiaozhou.onnx'
engine_file_path = 'Unet337.engine'

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

# 1. 使用 BuilderConfig 创建配置
with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network,
                                                                                                           TRT_LOGGER) as parser:
    # 创建配置对象并设置最大工作空间
    builder_config = builder.create_builder_config()
    builder_config.max_workspace_size = 1 << 28  # 256 MiB
    builder.max_batch_size = 1  # 设置最大batch大小

    # 2. 检查 ONNX 文件是否存在
    if not os.path.exists(onnx_file_path):
        print(f'ONNX file {onnx_file_path} not found, please run yolov3_to_onnx.py first to generate it.')
        exit(0)

    print(f'Loading ONNX file from path {onnx_file_path}...')

    # 3. 加载和解析 ONNX 文件
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing...')
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))

    # 4. 设置输入层的形状
    network.get_input(0).shape = [1, 3, 320, 320]

    print('Completed parsing of ONNX file')

    # 5. 构建 engine
    print(f'Building an engine from file {onnx_file_path}; this may take a while...')
    engine = builder.build_engine(network, builder_config)  # 改为使用 build_engine

    print("Completed creating Engine")

    # 6. 序列化并保存 engine
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())

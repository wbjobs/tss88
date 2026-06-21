"""
生成一个用于测试的虚拟ONNX语音情感分类模型。

注意：这只是一个随机初始化的模型，输出是随机的概率分布，
仅用于验证整个服务流程是否正常工作。
请替换为你自己的真实预训练ONNX模型。

模型期望输入形状：(batch_size, 1, 128, 313)
  - batch_size: 动态
  - 1: 单通道
  - 128: 梅尔滤波器组数
  - 313: 时间帧数（对应5秒音频，16kHz采样率，hop_length=256）

模型输出形状：(batch_size, 4)
  - 4: 4类情感 [happy, sad, angry, neutral]
"""
import sys
from pathlib import Path

import numpy as np

try:
    import onnx
    from onnx import TensorProto, helper
except ImportError:
    print("请先安装 onnx 包: pip install onnx")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))
from server import get_mel_spec_shape


def build_cnn_model(n_mels: int, n_frames: int, n_classes: int = 4) -> onnx.ModelProto:
    X = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, ["batch_size", 1, n_mels, n_frames]
    )
    Y = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, ["batch_size", n_classes]
    )

    nodes = []
    initializers = []

    def make_conv(name_prefix, in_channels, out_channels, kernel_h, kernel_w, input_name, pad_h, pad_w):
        w_shape = [out_channels, in_channels, kernel_h, kernel_w]
        w_data = np.random.randn(*w_shape).astype(np.float32) * 0.01
        w_name = f"{name_prefix}_w"
        initializers.append(
            helper.make_tensor(w_name, TensorProto.FLOAT, w_shape, w_data.flatten().tolist())
        )
        b_shape = [out_channels]
        b_data = np.zeros(b_shape, dtype=np.float32).tolist()
        b_name = f"{name_prefix}_b"
        initializers.append(
            helper.make_tensor(b_name, TensorProto.FLOAT, b_shape, b_data)
        )
        conv_out = f"{name_prefix}_out"
        nodes.append(
            helper.make_node(
                "Conv",
                inputs=[input_name, w_name, b_name],
                outputs=[conv_out],
                kernel_shape=[kernel_h, kernel_w],
                pads=[pad_h, pad_w, pad_h, pad_w],
                name=f"{name_prefix}_conv",
            )
        )
        relu_out = f"{name_prefix}_relu"
        nodes.append(
            helper.make_node("Relu", inputs=[conv_out], outputs=[relu_out], name=f"{name_prefix}_relu")
        )
        pool_out = f"{name_prefix}_pool"
        nodes.append(
            helper.make_node(
                "MaxPool",
                inputs=[relu_out],
                outputs=[pool_out],
                kernel_shape=[2, 2],
                strides=[2, 2],
                name=f"{name_prefix}_pool",
            )
        )
        return pool_out

    conv1_out = make_conv("conv1", 1, 32, 3, 3, "input", 1, 1)
    conv2_out = make_conv("conv2", 32, 64, 3, 3, conv1_out, 1, 1)
    conv3_out = make_conv("conv3", 64, 128, 3, 3, conv2_out, 1, 1)

    flat_out = "flat"
    nodes.append(
        helper.make_node("Flatten", inputs=[conv3_out], outputs=[flat_out], axis=1, name="flatten")
    )

    h1, w1 = n_mels // 8, n_frames // 8
    fc1_in = 128 * h1 * w1
    fc1_w_shape = [fc1_in, 256]
    fc1_w_data = (np.random.randn(*fc1_w_shape).astype(np.float32) * 0.01).flatten().tolist()
    initializers.append(
        helper.make_tensor("fc1_w", TensorProto.FLOAT, fc1_w_shape, fc1_w_data)
    )
    fc1_b_data = np.zeros([256], dtype=np.float32).tolist()
    initializers.append(
        helper.make_tensor("fc1_b", TensorProto.FLOAT, [256], fc1_b_data)
    )
    fc1_out = "fc1_out"
    nodes.append(
        helper.make_node(
            "Gemm",
            inputs=[flat_out, "fc1_w", "fc1_b"],
            outputs=[fc1_out],
            alpha=1.0,
            beta=1.0,
            transB=1,
            name="fc1_gemm",
        )
    )
    fc1_relu = "fc1_relu"
    nodes.append(
        helper.make_node("Relu", inputs=[fc1_out], outputs=[fc1_relu], name="fc1_relu")
    )

    fc2_w_shape = [256, n_classes]
    fc2_w_data = (np.random.randn(*fc2_w_shape).astype(np.float32) * 0.01).flatten().tolist()
    initializers.append(
        helper.make_tensor("fc2_w", TensorProto.FLOAT, fc2_w_shape, fc2_w_data)
    )
    fc2_b_data = np.zeros([n_classes], dtype=np.float32).tolist()
    initializers.append(
        helper.make_tensor("fc2_b", TensorProto.FLOAT, [n_classes], fc2_b_data)
    )
    nodes.append(
        helper.make_node(
            "Gemm",
            inputs=[fc1_relu, "fc2_w", "fc2_b"],
            outputs=["output"],
            alpha=1.0,
            beta=1.0,
            transB=1,
            name="fc2_gemm",
        )
    )

    graph = helper.make_graph(
        nodes=nodes,
        name="emotion_cnn",
        inputs=[X],
        outputs=[Y],
        initializer=initializers,
    )

    opset_imports = [helper.make_opsetid("", 14)]
    model = helper.make_model(graph, opset_imports=opset_imports)
    model.ir_version = 7

    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model


def main():
    n_mels, n_frames = get_mel_spec_shape()
    print(f"生成模型 - 输入形状: [batch, 1, {n_mels}, {n_frames}]")

    model = build_cnn_model(n_mels, n_frames, n_classes=4)

    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "emotion_classifier.onnx"

    onnx.save(model, str(out_path))
    print(f"模型已保存到: {out_path.resolve()}")
    print("注意：这只是一个随机初始化的测试模型，请替换为真实预训练模型！")


if __name__ == "__main__":
    main()

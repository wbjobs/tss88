# 在此目录下放置预训练的ONNX模型：emotion_classifier.onnx
#
# 模型期望输入：形状为 (batch, 1, 128, 313) 的梅尔频谱图(float32)
#   - 128 = n_mels
#   - 313 = 时间帧数（5秒 × 16000Hz / 256 hop_length ≈ 313）
#
# 模型输出：形状为 (batch, 4) 的分类logits
#   - 类别顺序：happy(开心), sad(难过), angry(愤怒), neutral(中性)
#
# 如需生成测试模型，运行：
#   python scripts/generate_dummy_model.py

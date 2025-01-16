import numpy as np
import matplotlib.pyplot as plt
import time

def gerchberg_saxton_dynamic(output_amplitude, input_amplitude_known=None,
                             max_iterations=100, tolerance=1e-6):
    """
    实现具有动态迭代步数的 Gerchberg-Saxton 算法。

    Args:
        output_amplitude:  输出平面的幅度 (numpy array)。
        input_amplitude_known: 输入平面的已知幅度 (numpy array)，如果未知则为 None。
        max_iterations: 最大迭代次数。
        tolerance: 收敛容差，当连续两次迭代相位变化的 RMSE 低于此值时停止迭代。

    Returns:
        input_phase: 估计的输入平面相位 (numpy array)。
        output_phase: 估计的输出平面相位 (numpy array)。
        iterations_done: 实际迭代次数。
    """
    ny, nx = output_amplitude.shape
    input_phase = np.random.rand(ny, nx) * 2 * np.pi - np.pi  # 初始化输入平面相位为随机值
    input_field = output_amplitude * np.exp(1j * input_phase)  # 构建初始输入复数场

    previous_phase = np.zeros_like(input_phase)
    iterations_done = 0

    for i in range(max_iterations):
        iterations_done = i + 1

        # 前向传播 (傅里叶变换)
        output_field_fft = np.fft.fft2(input_field)
        output_phase = np.angle(output_field_fft)

        # 在输出平面施加幅度约束
        output_field_constrained = output_amplitude * np.exp(1j * output_phase)

        # 反向传播 (逆傅里叶变换)
        input_field_ifft = np.fft.ifft2(output_field_constrained)
        current_input_phase = np.angle(input_field_ifft)

        # 在输入平面施加幅度约束 (如果已知)
        if input_amplitude_known is not None:
            input_field = input_amplitude_known * np.exp(1j * current_input_phase)
        else:
            input_field = np.abs(input_field_ifft) * np.exp(1j * current_input_phase)

        # 检查收敛性
        phase_diff_rmse = np.sqrt(np.mean((current_input_phase - previous_phase)**2))
        if phase_diff_rmse < tolerance:
            print(f"算法在 {iterations_done} 次迭代后收敛，RMSE: {phase_diff_rmse:.8f}")
            break

        previous_phase = current_input_phase
    input_amplitude = np.abs(input_field)
    return current_input_phase, output_phase, input_amplitude, output_amplitude, iterations_done

# 示例用法
if __name__ == "__main__":
    # 定义输出平面的幅度（例如，一个圆形）
    size = 512
    x = np.arange(-size//2, size//2)
    y = np.arange(-size//2, size//2)
    xx, yy = np.meshgrid(x, y)
    radius = size // 4
    output_amplitude = np.zeros((size, size))
    output_amplitude[xx**2 + yy**2 <= radius**2] = 1
    # 定义输入平面幅度（高斯光束）
    input_amplitude = 5*np.exp(-((xx**2 + yy**2) / (2 * (size // 8)**2)))
    # 运行动态迭代的 Gerchberg-Saxton 算法
    start_time = time.time()
    estimated_input_phase, estimated_output_phase, estimated_input_amplitude, estimated_output_amplitude, iterations = gerchberg_saxton_dynamic(
        output_amplitude, input_amplitude, max_iterations=1000, tolerance=1e-4
    )
    end_time = time.time()
    print(f"算法运行时间: {end_time - start_time:.4f} 秒")
    print(f"实际迭代次数: {iterations}")

    # 可视化结果
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(output_amplitude, cmap='gray')
    plt.title('Estimated Output Amplitude')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(estimated_input_amplitude, cmap='gray')
    plt.title('Estimated Input amplitude')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(estimated_input_phase, cmap='hsv')
    plt.title('Estimated Input Phase')
    plt.colorbar()

    plt.tight_layout()
    plt.show()
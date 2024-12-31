# 表面振动到光信号的转换：物理与数学原理

## 概述

激光多普勒振动测量仪（LDV）是一种高精度的非接触式振动测量设备。它利用激光的多普勒效应来测量物体表面的振动速度和位移。本文件将详细解释 LDV 系统中如何将表面振动（通过模态分解描述）转换为光强度变化信号，并推导出其背后的物理和数学公式。

---

## 1. 背景与动机

### 1.1 振动测量的需求

在工程领域，精确测量结构的振动特性对于评估其安全性、可靠性和性能至关重要。传统的接触式测量方法（如加速度计）存在一些局限性，例如：

- **附加质量效应**：传感器本身的质量会影响被测结构的振动特性。
- **安装限制**：难以在高温、高速或狭小空间等环境下安装。
- **测量点有限**：通常只能进行单点或少数点的测量。

### 1.2 LDV 的优势

LDV 技术克服了传统方法的局限性，具有以下优势：

- **非接触式测量**：无需与被测物直接接触，避免了附加质量效应。
- **高空间分辨率**：可以实现微米级的空间分辨率。
- **宽频带测量**：可以测量从 DC 到 MHz 的振动频率。
- **远距离测量**：可以在几米甚至更远的距离进行测量。

**动机**：利用 LDV 技术可以实现对各种复杂结构的高精度、非接触式振动测量，满足工程应用的需求。

---

## 2. LDV 的工作原理

### 2.1 多普勒效应

LDV 的核心原理是光学多普勒效应。当激光照射到振动物体表面时，反射光的频率会发生微小的变化，该变化与物体表面的振动速度成正比。

**公式**：

$$
f_D = \frac{2v}{\lambda}
$$

其中：

- \( f_D \)：多普勒频移
- \( v \)：物体表面沿激光方向的速度
- \( \lambda \)：激光波长

### 2.2 干涉测量

LDV 系统通常采用干涉测量的方法来检测多普勒频移。典型的迈克尔逊干涉仪结构如下：

1. **激光光源**：发出单频、高相干性的激光束。
2. **分束器**：将激光束分成两束：参考光束和测量光束。
3. **参考镜**：反射参考光束。
4. **被测物体**：反射测量光束。
5. **光电探测器**：检测参考光束和测量光束的干涉信号。

**过程**：

- 参考光束经过固定路径后到达光电探测器。
- 测量光束照射到被测物体表面，发生反射，并携带了物体的振动信息。
- 两束光在光电探测器处发生干涉，产生干涉条纹。
- 物体表面的振动导致测量光束的光程发生变化，从而引起干涉条纹的移动。
- 光电探测器将干涉条纹的移动转换为光强度变化信号。

---

## 3. 表面振动到光信号的转换

### 3.1 模态分解

实际结构的振动通常可以分解为多个模态的叠加。每个模态具有特定的固有频率和模态形状。

**公式**：

$$
u(x, y, t) = \sum_{n=1}^{\infty} \Phi_n(x, y) q_n(t)
$$

其中：

- \( u(x, y, t) \)：表面在位置 \( (x, y) \) 处、时间 \( t \) 的位移
- \( \Phi_n(x, y) \)：第 \( n \) 阶模态的模态形状函数
- \( q_n(t) \)：第 \( n \) 阶模态的模态坐标（时间函数）

### 3.2 模态坐标

模态坐标 \( q_n(t) \) 描述了每个模态随时间的变化情况。对于简谐振动，模态坐标可以表示为：

$$
q_n(t) = A_n \cos(\omega_n t + \phi_n)
$$

其中：

- \( A_n \)：第 \( n \) 阶模态的振幅
- \( \omega_n \)：第 \( n \) 阶模态的固有频率
- \( \phi_n \)：第 \( n \) 阶模态的初相位

### 3.3 表面速度

物体表面的振动速度可以通过对位移求时间导数得到：

$$
v(x, y, t) = \frac{\partial u(x, y, t)}{\partial t} = \sum_{n=1}^{\infty} \Phi_n(x, y) \dot{q}_n(t)
$$

其中 \( \dot{q}_n(t) \) 为：

$$
\dot{q}_n(t) = -A_n \omega_n \sin(\omega_n t + \phi_n)
$$

### 3.4 光程差

测量光束的光程会受到表面位移的影响。光程差 \( \Delta L \) 可以表示为：

$$
\Delta L(x, y, t) = 2u(x, y, t)
$$

**解释**：由于光束往返两次经过表面位移，因此光程差是位移的两倍。

### 3.5 相位差

光程差的变化会导致测量光束和参考光束之间产生相位差 \( \Delta \phi \)：

$$
\Delta \phi(x, y, t) = \frac{2\pi}{\lambda} \Delta L(x, y, t) = \frac{4\pi}{\lambda} u(x, y, t)
$$

### 3.6 光场表示

参考光束和测量光束的电场可以表示为：

$$
E_{ref}(t) = E_r \cos(\omega t)
$$

$$
E_{meas}(x, y, t) = E_m \cos(\omega t + \Delta \phi(x, y, t))
$$

其中：

- \( E_r \)：参考光束的振幅
- \( E_m \)：测量光束的振幅
- \( \omega \)：激光角频率

### 3.7 干涉强度

光电探测器检测到的干涉强度 \( I(x, y, t) \) 为：

$$
I(x, y, t) = |E_{ref}(t) + E_{meas}(x, y, t)|^2
$$

展开并化简后得到：

$$
I(x, y, t) = E_r^2 + E_m^2 + 2E_r E_m \cos(\Delta \phi(x, y, t))
$$

将相位差 \( \Delta \phi(x, y, t) \) 代入：

$$
I(x, y, t) = E_r^2 + E_m^2 + 2E_r E_m \cos\left(\frac{4\pi}{\lambda} \Delta L(x, y, t)\right)
$$

**分析**：

- 干涉强度 \( I(x, y, t) \) 是关于时间 \( t \) 的函数，包含了所有模态的振动信息。
- 由于余弦函数的非线性，干涉强度与表面位移之间不是简单的线性关系。
- 通过对干涉强度信号进行解调，可以提取出表面振动的信息。

---

## 4. 代码实现与公式对应

在 `ldv.py` 的 `analyze_vibration` 函数中，实现了将表面振动转换为光信号的计算过程。以下将详细解释第 3 部分中的物理数学公式是如何对应到代码中的具体实现的。

### 4.1 表面位移和速度计算

- **公式回顾**:

  - 表面位移公式 (3.1):
    $$
    u(x, y, t) = \sum_{n=1}^{\infty} \Phi_n(x, y) q_n(t)
    $$
  - 模态坐标公式 (3.2):
    $$
    q_n(t) = A_n \cos(\omega_n t + \phi_n)
    $$
  - 表面速度公式 (3.3):
    $$
    v(x, y, t) = \frac{\partial u(x, y, t)}{\partial t} = \sum_{n=1}^{\infty} \Phi_n(x, y) \dot{q}_n(t)
    $$
    其中
    $$
    \dot{q}_n(t) = -A_n \omega_n \sin(\omega_n t + \phi_n)
    $$

- **代码实现**:

  ```python:src/ldv.py
  # ...
  displacement, velocity = self.vibration_model.calculate_surface_response(x, y, time_points)
  # ...
  ```

  - `calculate_surface_response` 函数对应于公式 (3.1) 和 (3.3)。它根据模态分解模型计算表面在给定位置 `(x, y)` 和时间点 `time_points` 的位移 `displacement` 和速度 `velocity`。
  - 具体计算方法在 `ClassicalModalAnalysis` 或 `BesselModalAnalysis` 类中实现，分别对应于 `classical` 和 `bessel` 两种分析类型。
  - `ClassicalModalAnalysis` 类中的 `calculate_modal_response` 函数实现了公式 (3.1) 的求和部分，而 `calculate_single_mode_response` 函数则计算了每个模态的贡献，对应于公式 (3.2) 和 (3.3) 中的 \(q_n(t)\) 和 \(\dot{q}_n(t)\)。
  - `BesselModalAnalysis` 类中的 `calculate_modal_response` 函数和 `calculate_single_mode_response` 函数也实现了类似的计算，但基于贝塞尔函数。

### 4.2 光程差和相位差计算

- **公式回顾**:

  - 光程差公式 (3.4):
    $$
    \Delta L(x, y, t) = 2u(x, y, t)
    $$
  - 相位差公式 (3.5):
    $$
    \Delta \phi(x, y, t) = \frac{2\pi}{\lambda} \Delta L(x, y, t) = \frac{4\pi}{\lambda} u(x, y, t)
    $$

- **代码实现**:

  ```python:src/ldv.py
  # ...
  E_meas = self.optical_system.calculate_measurement_beam(x, y, displacement[t_idx], t, self.material)
  # ...
  ```

  ```python:src/models/optical.py
  # ...
  class OpticalSystem:
      # ...
      def calculate_measurement_beam(self, x: float, y: float, 
                                  displacement: float, t: float,
                                  material: MaterialProperties) -> complex:
          """計算測量光場
          
          Args:
              x, y: 測量位置 [m]
              displacement: 表面位移 [m]
              t: 時間 [s]
              material: 材料特性
              
          Returns:
              complex: 測量光場複數振幅
          """
          # ...
          # 計算總相位
          # ...
          path_phase = 2 * np.pi / self.wavelength * (self.measurement_path + 2*displacement)
          # ...
          total_phase = path_phase + time_phase + gaussian_phase + tilt_phase
          
          return E0 * np.exp(1j * total_phase)
      # ...
  ```

  - `calculate_measurement_beam` 函数计算测量光束的电场，其中考虑了表面位移 `displacement` 引起的相位变化。
  - `path_phase = 2 * np.pi / self.wavelength * (self.measurement_path + 2*displacement)` 这一行代码直接对应了公式 (3.4) 和 (3.5)。其中 `2*displacement` 对应于光程差 \(\Delta L\)，`2 * np.pi / self.wavelength` 对应于 \(\frac{2\pi}{\lambda}\)，两者相乘即得到相位差 \(\Delta \phi\)。

### 4.3 光场计算

- **公式回顾**:

  - 参考光束电场公式 (3.6):
    $$
    E_{ref}(t) = E_r \cos(\omega t)
    $$
  - 测量光束电场公式 (3.7):
    $$
    E_{meas}(x, y, t) = E_m \cos(\omega t + \Delta \phi(x, y, t))
    $$

- **代码实现**:

  ```python:src/ldv.py
  # ...
  E_ref = self.optical_system.calculate_reference_beam(t)
  E_meas = self.optical_system.calculate_measurement_beam(x, y, displacement[t_idx], t, self.material)
  # ...
  ```

  ```python:src/models/optical.py
  # ...
  class OpticalSystem:
      # ...
      def calculate_reference_beam(self, t: float) -> complex:
          """計算參考光場
          
          Args:
              t: 時間 [s]
              
          Returns:
              complex: 參考光場複數振幅
          """
          # ...
          phase = (2 * np.pi / self.wavelength * self.reference_path - 
                  omega_laser * t)
          
          return E0 * np.exp(1j * phase)

      def calculate_measurement_beam(self, x: float, y: float, 
                                  displacement: float, t: float,
                                  material: MaterialProperties) -> complex:
          """計算測量光場
          
          Args:
              x, y: 測量位置 [m]
              displacement: 表面位移 [m]
              t: 時間 [s]
              material: 材料特性
              
          Returns:
              complex: 測量光場複數振幅
          """
          # ...
          # 計算總相位
          omega_laser = 2 * np.pi * 3e8 / self.wavelength
          path_phase = 2 * np.pi / self.wavelength * (self.measurement_path + 2*displacement)
          time_phase = -omega_laser * t
          gaussian_phase = (2 * np.pi / self.wavelength * r2/(2*self.R_z) - 
                          self.gouy_phase)
          tilt_phase = (2 * np.pi * self.w_0 * np.sin(tilt_angle) / self.wavelength)
          total_phase = path_phase + time_phase + gaussian_phase + tilt_phase
          
          return E0 * np.exp(1j * total_phase)
      # ...
  ```

  - `calculate_reference_beam` 函数计算参考光束的电场，对应于公式 (3.6)。
  - `calculate_measurement_beam` 函数计算测量光束的电场，对应于公式 (3.7)。其中 `total_phase` 包含了相位差 \(\Delta \phi(x, y, t)\) 以及其他相位项。

### 4.4 干涉强度计算

- **公式回顾**:

  - 干涉强度公式 (3.8):
    $$
    I(x, y, t) = |E_{ref}(t) + E_{meas}(x, y, t)|^2
    $$
    展开并化简后得到：
    $$
    I(x, y, t) = E_r^2 + E_m^2 + 2E_r E_m \cos(\Delta \phi(x, y, t))
    $$

- **代码实现**:

  ```python:src/ldv.py
  # ...
  intensity = self.optical_system.calculate_interference_intensity(E_ref, E_meas)
  interference_intensity.append(intensity)
  # ...
  ```

  ```python:src/models/optical.py
  # ...
  class OpticalSystem:
      # ...
      def calculate_interference_intensity(self, E_ref: complex, 
                                        E_meas: complex) -> float:
          """計算干涉強度
          
          Args:
              E_ref: 參考光場
              E_meas: 測量光場
              
          Returns:
              float: 干涉強度
          """
          # 考慮相干性的干涉計算
          intensity = np.abs(E_ref + E_meas * self.coherence_factor)**2
          
          return intensity
      # ...
  ```

  - `calculate_interference_intensity` 函数计算干涉强度，对应于公式 (3.8)。
  - `np.abs(E_ref + E_meas * self.coherence_factor)**2` 直接对应了 \(|E_{ref}(t) + E_{meas}(x, y, t)|^2\)，其中 `self.coherence_factor` 考虑了相干性的影响。

---

## 5. 总结

LDV 系统将表面振动转换为光强度变化信号的过程可以总结为：

1. **模态分解**：将复杂的表面振动分解为多个模态的叠加。
2. **计算表面位移和速度**：根据模态分解模型计算每个时刻的表面位移和速度。
3. **计算光程差和相位差**：表面位移导致测量光束的光程发生变化，进而引起相位差。
4. **计算光场**：根据相位差计算参考光束和测量光束的电场。
5. **计算干涉强度**：根据参考光和测量光的电场计算干涉强度，该强度包含了表面振动的信息。

通过以上步骤，LDV 系统成功地将表面振动信息编码到光强度变化信号中，实现了非接触式、高精度的振动测量。

---

## 6. 进一步分析

- **非线性解调**：由于干涉强度与表面位移之间存在非线性关系，需要采用适当的解调算法来提取振动信息。常用的解调算法包括相位解卷、反正切解调等。
- **噪声抑制**：实际测量中会受到各种噪声的影响，如激光相位噪声、光电探测器噪声、环境振动等。需要采用滤波、信号平均等方法来抑制噪声，提高信噪比。
- **多点测量**：通过扫描激光束或使用多个光电探测器，可以实现多点甚至全场的振动测量。

---

## 参考文献

1. Drain, L. E. (1980). *The Laser Doppler Technique*. John Wiley & Sons.
2. Rothberg, S. J., Baker, J. R., & Halliwell, N. A. (1989). Laser vibrometry: Pseudo-vibrations. *Journal of Sound and Vibration*, *135*(3), 516-522.
3. Castellini, P., Martarelli, M., & Tomasini, E. P. (2006). Laser Doppler Vibrometry: Development of advanced solutions answering to technology’s needs. *Mechanical Systems and Signal Processing*, *20*(6), 1265-1285. 
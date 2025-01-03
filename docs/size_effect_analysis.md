# 尺寸效應比較：問題與逐步分析

以下內容是基於您提出的需求與問題，進行的分解重述與詳細物理／數學推理過程。此文件不直接下結論，而是透過一步步的分析與推理，協助您理解兩種不同板子尺寸在相同外力激振下，應該會產生哪些物理上的差異，以及在視覺化動畫中可能會呈現出哪些差異。

---

## 1. 問題重述與需求

1) 您想比較兩種幾何尺寸不同的矩形板，在相同外力激振下的振動行為。  
   - 板子 A 尺寸：長 1.0 m × 寬 1.0 m  
   - 板子 B 尺寸：長 2.0 m × 寬 0.5 m  

2) 外力條件保持一致：  
   - 相同的力幅 (amplitude)、相同的頻率 (frequency)、相同的作用位置 (position，相對於板子的中心)。  
   - 在可視化呈現上，彼此的外力箭頭長度與範圍也保持相同，目的在於更客觀地比較因「板子尺寸不同」而產生的振動行為差異，而非因外力尺度改變。
   - 若要更貼近「麥克風圓形振膜」，並比較不同半徑時，可以在程式中使用 `use_total_tension=True`，
     令 material['T'] 轉為「總張力 (N)」並動態換算成線張力。

3) 程式利用簡化的薄板力學模型 (Kirchhoff-Love 理論) 與假設小撓度的自由度進行模態分析，以及進行加上簡諧力時的強迫振動分析。均假設為簡支邊界條件 (simply supported)。

4) 您要求：
   - 不要跳到最終結論或直斷哪個板子「一定怎樣」。  
   - 要有逐步的物理與數學公式推導，釐清為什麼尺寸改變會使得自然頻率與響應產生差異。  
   - 同時說明視覺化上可能觀察到的差別。

---

## 2. 物理與數學分析步驟

下面，我們以逐步推理的方式，先針對自然頻率 (free vibration) 與強迫振動 (forced vibration) 兩個層面進行討論。這裡採用的模型方程式為「薄板振動方程」，同時考慮簡支板的模態函數。

### 2.1 板子的自然頻率公式

在相同材料（鋼材）、相同厚度 (h) 與相同材料參數 (E, ρ, ν) 下，矩形板 (長 L、寬 W) 的 (m,n) 模態自然頻率 (ωₘₙ) 通常可以寫成下列形式（省略邊界條件導出的常數，僅保留關鍵部分）：

1) 板的「彎曲剛度」：  
   D = (E · h³) / [12 · (1 - ν²)]  

2) 板的「質量單位」：  
   ρ 代表密度，h 代表厚度，所以質量 per unit area = ρ·h  
   而板整體的面積 = L×W  

3) 在簡支條件下，(m,n) 模態的自然頻率約略可表示為：  
   ωₘₙ ≈ kₘₙ · √(D / (ρ·h))  
   其中 kₘₙ 與 (m/L)² + (n/W)² 成正比（因為模態形狀由正弦函數組成）。  
   也就是說，  
   ωₘₙ ∝ √( (E·h³)/(ρ·h) ) × √( (m/L)² + (n/W)² )  
   => ωₘₙ ∝ √(E/ρ ) × h × √( (m/L)² + (n/W)² )

#### 步驟推理：
- 從公式可看出，固定材料 (E, ρ, ν) 與固定厚度 (h) 的前提下，若 L 與 W 不同，則 (m/L)² + (n/W)² 不同，導致自然頻率同樣會判斷出差異。  
- 例如：板子 A (1.0×1.0) 與 板子 B (2.0×0.5) 的 (1,1) 模態，因為 L 與 W 的比值不同，導致 (1/L)² + (1/W)² 的數值會不一樣。

### 2.2 強迫振動方程與位移響應

加入簡諧力 F(t) = F₀ · cos(ω·t) 時，系統會有強迫振動。對於每個模態 (m,n)，其強迫響應可以透過模式疊加法表示成下列形式：

u(x,y,t) = Σₘₙ [ φₘₙ(x,y) · qₘₙ(t) ]

其中，qₘₙ(t) 的方程式類似單自由度系統：  
mₘₙ · q̈ₘₙ + cₘₙ · q̇ₘₙ + kₘₙ · qₘₙ = Fₘₙ · cos(ω·t)

- mₘₙ, cₘₙ, kₘₙ 分別對應到該模態的「模態質量、模態阻尼、模態剛度」。  
- Fₘₙ 則為該模態對應的等效力，與施力位置相對應的模態形狀有關。  

當激振力的頻率 ω 與某個模態的自然頻率 ωₘₙ 接近時，該模態響應可預期會顯著放大（即共振現象）。

#### 步驟推理：
1. 模態質量與模態剛度同樣依賴板子幾何尺寸 (L, W)。  
   - 質量通常與 L×W 成正比，因此較大尺寸的板子，質量會比較大。  
   - 剛度與 (1/L)⁴ 或 (1/L²)(1/W²) 之類的因子有關（視板振動模式而定）。  
2. 在相同力幅 F₀、相同激振頻率 ω 的情況下，不同板子因質量與剛度不同，產生的位移振幅與相位也會不同。  

### 2.3 兩塊板子幾何尺寸不同的效果

假設兩塊板子材料、厚度皆相同，一塊是 (1.0×1.0)、另一塊是 (2.0×0.5)，施加相同振幅、相同頻率的外力在中心點。逐步推理如下：

1. 質量比較：  
   - 板子 B (2.0×0.5) 之面積 = 1.0 m²，但它的形狀是長邊 2.0 m、寬邊 0.5 m → 跟板子 A (1.0×1.0) 同樣是 1.0 m² 面積的情況作比較，就會發現它的長寬比不同，但總面積一樣；若有其他尺寸 (例如 2×2) 則面積就會更大。  
   - 若面積不同，也可以預期更大面積 → 更大質量；若同樣面積則質量近似，但展開形狀不同，對模態頻率的計算仍不同。  

2. 自然頻率比較：  
   - 對於 (1,1) 模態：ω₁₁ ∝ √( (1/L)² + (1/W)² )。  
   - 板子 A (1×1)：(1/L)² + (1/W)² = 1² + 1² = 2  
   - 板子 B (2×0.5)：(1/2)² + (1/0.5)² = 0.25 + 4 = 4.25  
   - 單就這個 (m=1, n=1) 而言，(2×0.5) 在這個比例下有不同計算值，所以自然頻率也會不同（當然實際的其他因子、邊界條件都要完整代入，但大致可看出差異）。

3. 在相同激振頻率 ω 下：  
   - 如果 ω 剛好更接近某一塊板子的自然頻率，則該板子會顯示較高的振幅響應。  
   - 另一塊板子若沒有匹配在這個頻率，就可能響應振幅較小。  

4. 視覺化差異：  
   1) 在兩塊板子圖上，外力箭頭同樣大小（基準長度一樣），表示它們受到同樣大小的外力。  
   2) 由於板子尺寸幾何不同，您會觀察到它們的整體變形形狀 (變形波節、節線分佈) 有所差別，尤其當 m=1, n=2 等不對稱模態時，長邊方向的週期數量可能比短邊多。  
   3) 可能有一塊板子在此激振頻率下產生更大振幅，而另一塊相對小。這是因為共振條件和質量剛度特性都不同。  

   - 若帶入「圓形振膜」的範例，也可選擇程式的 use_total_tension=True，
     使得 material['T'] 被解釋為「總張力」，從而逼近實際麥克風設計：
       - 大 R 不會過度鬆弛
       - 各半徑的張力都按照圓周長分配
     如此呈現出較符合工程經驗的情境。

---

## 3. 小結論與提醒

透過上述步驟，我們**沒有**直接斷言哪塊板子「一定振幅較大」或「一定自然頻率較低」，而是說明計算機制在哪裡可能產生差異、以及外力頻率相對於各板子自然頻率差異時，會如何改變其振幅大小。

因為實際振動響應除了尺寸，更會受到材料阻尼係數、邊界條件真實度、厚度分布等影響；在這裡以理想化的簡支邊界與薄板假設下，可以看見主要的幾何因素帶來的頻率與形狀變化。

在視覺上：
- 外力箭頭基準長度相同 → 可以明確看出外力固定。  
- 兩塊板子因尺寸不同導致的模態頻率或質量不同 → 振動振幅分布會不一樣。  
- 觀察動畫可以看到它們的固定邊界、形變週期差異，以及疊加模態下是否產生較多波節或較少波節等等。  

這些正是兩塊尺寸不同的板子，在相同外力下所產生的「物理上」和「視覺呈現上」差別的原因所在。  

## 詳細分析與推理

在麥克風振膜的研究情境下，程式碼執行發現「較大振膜 (右邊) 出現更大位移」的結果，其物理與數學原理分析如下：

1. 振膜模型背景  
   - 麥克風振膜可理想化為「周邊固定、具有線張力 T 的圓形薄膜」。  
   - 若 T 為每單位長度的張力 (N/m)，且半徑為 R，面密度為 ρₛ，則第 (m,n) 模態的自然角頻率約為:
     ωₘₙ = (λₘₙ / R) × √(T / ρₛ)  
     其中 λₘₙ 為對應 Bessel 函數零點。R 越大，系統自然頻率越低。

2. 在相同外力頻率下的振幅比較  
   - 若同樣使用線張力 T=2000 N/m，而半徑 R₂ > R₁，則較大 R₂ 的自然頻率更靠近外力頻率 (1000 Hz)。  
   - 當外力頻率靠近振膜的自然頻率，即可使該模態的振幅放大。因此看起來較大振膜的位移幅度更大。

3. 工程直覺與程式設定  
   - 若想比較「同樣總周邊拉力」(單位 N) 而非「同樣線張力」(N/m)，需依 R 動態重算 T_line = (總張力) / (2πR)。否則程式中將大 R 與小 R 都使用相同的線張力量值，讓大 R 的『有效剛度』變小，導致對外力的響應位移變大。  
   - 此設定本身在公式與模擬上一致，並未違背物理定律，只是必須和實務設計意圖配合理解。

4. 結論  
   - 在此設定下，大 R 較容易出現與外力頻率貼近之模態，使得振幅加大。並非程式出錯，乃是對應到較低固有頻率的特性。  
   - 若要符合某些工程直覺 (如同樣總張力下改變尺寸的影響)，則需更新程式內 T 的計算方式。

(檔案保存：docs/size_effect_analysis.md) 
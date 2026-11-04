## Image and Audio Processing
*in Python*

&nbsp;

### 🖼️ P1 – JPEG Image Compression

Implementation of the sequential JPEG baseline encoder and decoder pipeline to analyze compression efficiency, quantization, and distortion metrics. Including:
* **Color Space Conversion:** RGB to YCbCr conversion and its inverse.
* **Preprocessing:** Boundary padding and unpadding for dimensions not divisible by 32 times 32.
* **Sub-sampling:** Chrominance downsampling and upsampling.
* **Transform Coding:** Discrete Cosine Transform (DCT) and Inverse DCT (IDCT).
* **Quantization:** Lossy coefficient quantization across various quality factors.
* **Differential Coding:** DPCM encoding and decoding of DC coefficients.
* **Evaluation Metrics:** End-to-end distortion measurement including Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Signal-to-Noise Ratio (SNR), Peak Signal-to-Noise Ratio (PSNR), max difference, and average difference.

&nbsp;

---

### 🎵 P2 – Music Information Retrieval

Content-based music recommendation and feature analysis engine built using *librosa* and custom signal processing algorithms. Including:
* **Feature Extraction:** Extracts 190 statistical features (mean, standard deviation, skewness, kurtosis, median, max, min) from spectral (MFCC, centroid, bandwidth, contrast, flatness, rolloff), temporal (f_0, RMS, zero-crossing rate), and rhythmic (tempo) domains.
* **Custom Spectral Centroid:** Hand-crafted implementation of the spectral centroid feature using *NumPy* and *SciPy* validated against *librosa* via Pearson correlation and RMSE.
* **Similarity Metrics:** Computes similarity rankings using Euclidean distance, Manhattan distance, and Cosine similarity on normalized feature vectors.
* **Evaluation Framework:** Objective evaluation via metadata cross-referencing (artist, genre, mood).

&nbsp;

---

**NOTE:** This project was developed within the scope of a Computer Science course by Cláudia Torres, Diogo Silva, Henrique Marques and Pedro Bento.

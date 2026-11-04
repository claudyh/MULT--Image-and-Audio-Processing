#-----MULT project--------------------


#Imports ------------------------------
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import cv2
import scipy.fftpack


#Global vars ---------------------------
matrix_Y = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]])

matrix_Cb_Cr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                        [18, 21, 26, 66, 99, 99, 99, 99],
                        [24, 26, 56, 99, 99, 99, 99, 99],
                        [47, 66, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99]])


#Show functions -----------------------
def showImg(img, cmap=None, caption=""):
    plt.figure()
    plt.imshow(img, cmap)
    plt.axis('off')
    plt.title(caption)
    plt.show()
    
def show_channels(Y, Cb, Cr, title_Y, title_Cb, title_Cr):
    # Show each channel
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.imshow(Y, 'gray')
    plt.title(title_Y)
    #plt.axis('off')

    plt.subplot(132)
    plt.imshow(Cb, 'gray')
    plt.title(title_Cb)
    #plt.axis('off')

    plt.subplot(133)
    plt.imshow(Cr, 'gray')
    plt.title(title_Cr)
    #plt.axis('off')

    plt.show()


#Encode functions ---------------------
def pad_image(img):
    # Store img dimentions
    original_shape = img.shape

    # Calculates the number of rows and columns to be added
    pad_rows = (32 - (original_shape[0] % 32))
    pad_cols = (32 - (original_shape[1] % 32))

    # Replicates the last column and puts that on the right (hstack = horizontal stack)
    repeated_cols = np.repeat(img[:, -1:, :], pad_cols, 1)
    padded_img = np.hstack([img, repeated_cols])

    # Replicates the last row and puts that at the bottom (vstack = vertical stack)
    repeated_rows = np.repeat(padded_img[-1:, :, :], pad_rows, 0)
    padded_img = np.vstack([padded_img, repeated_rows])

    return padded_img, original_shape

def splitChannels(img):
    # Split matrices:
    C1 = img[:, :, 0]
    C2 = img[:, :, 1]
    C3 = img[:, :, 2]

    return C1, C2, C3

def rgb_to_ycbcr(img_rgb):
    # Conversion Matrix
    conversion_matrix = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
    offset = np.array([0, 128, 128])

    # Apply Transformation
    img_ycbcr = np.dot(img_rgb, conversion_matrix.T) + offset

    return img_ycbcr

def downsampling(img_ycbcr, downsamp_factor_y, downsamp_factor_cb, downsamp_factor_cr, interpolation_func):
    Y, Cb, Cr = splitChannels(img_ycbcr)
    
    #Get channels size
    h, w = Cb.shape[:2]
    
    #Determine the downsampling factor
    if downsamp_factor_cr == 0:
        dsf_w_cb = dsf_w_cr = dsf_h_cb = dsf_h_cr = downsamp_factor_cb / downsamp_factor_y
    else:
        dsf_w_cb = downsamp_factor_cb / downsamp_factor_y
        dsf_w_cr = downsamp_factor_cr / downsamp_factor_y
        dsf_h_cb = dsf_h_cr = 1
    
    #Resize Cb and Cr channels
    Cb_d = cv2.resize(Cb, (int(w * dsf_w_cb), int(h * dsf_h_cb)), interpolation_func)
    Cr_d = cv2.resize(Cr, (int(w * dsf_w_cr), int(h * dsf_h_cr)), interpolation_func)
    
    #Print sizes
    #print("Y", Y.shape)
    #print("Cb_d", Cb_d.shape)
    #print("Cr_d", Cr_d.shape)

    return Y, Cb_d, Cr_d

def calculate_dct(channel):
    dct_result = scipy.fftpack.dct(scipy.fftpack.dct(channel, axis=0, norm='ortho'), axis=1, norm='ortho')
    return dct_result

def block_dct(channel, block_size):
    rows, cols = channel.shape
    dct_result = np.zeros((rows, cols))

    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            block = channel[i : i+block_size, j : j+block_size]
            dct_block= scipy.fftpack.dct(scipy.fftpack.dct(block, norm='ortho').T, norm='ortho').T
            dct_result[i : i+block_size, j : j+block_size] = dct_block

    return dct_result

def get_quantization_matrix(channel_number, quality_factor):
    if channel_number == 0:
        base_matrix = matrix_Y
    else:
        base_matrix = matrix_Cb_Cr

    if quality_factor < 50:
        scaling_factor = 50 / quality_factor
    else:
        scaling_factor = (100 - quality_factor) / 50
        
    if scaling_factor==0:
        quantization_matrix = np.ones((8, 8))
    else:
        quantization_matrix = np.round((scaling_factor * base_matrix))
    
    quantization_matrix= np.clip(quantization_matrix, 1, 255).astype(np.uint8)
        
    return quantization_matrix

def calc_block_dct_qnt(channel, channel_number, block_size, quality_factor):
    rows, cols = channel.shape
    dct_result = np.zeros((rows, cols))
    dct_quantized_result = np.zeros((rows, cols))
    quantization_matrix = get_quantization_matrix(channel_number, quality_factor)

    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            block = channel[i:i+block_size, j:j+block_size]
            dct_block = scipy.fftpack.dct(scipy.fftpack.dct(block, norm='ortho').T, norm='ortho').T
            quantized_block = np.round(dct_block / quantization_matrix)
            dct_result[i:i+block_size, j:j+block_size] = dct_block
            dct_quantized_result[i:i+block_size, j:j+block_size] = quantized_block

    return dct_result, dct_quantized_result

def encode_dpcm(channel, block_size):
    rows, cols = channel.shape
    differences = np.copy(channel)
    
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            i_ant, j_ant = i, j
            if j < block_size:
                i_ant = i - block_size
                j_ant = cols
            if i == 0 and j == 0:
                continue
            else:
                differences[i, j] = channel[i, j] - channel[i_ant, j_ant - block_size]
            
    return differences


#Encoder ------------------------------
def encoder(img, qf):
    
    print("- - - ENCODER - - -")
    
    #Ex 4.1 - padding
    padded_img, original_shape = pad_image(img)
    
    #Ex 3.4 - split RGB components
    R, G, B = splitChannels(padded_img)

    #Ex 3.6 - visualize RGB channels
    cm_red = clr.LinearSegmentedColormap.from_list("red", [(0, 0, 0), (1, 0, 0)], 256)
    cm_green = clr.LinearSegmentedColormap.from_list("green", [(0, 0, 0), (0, 1, 0)], 256)
    cm_blue = clr.LinearSegmentedColormap.from_list("blue", [(0, 0, 0), (0, 0, 1)], 256)
    
    #showImg(R, cm_red, "Red")
    #showImg(G, cm_green, "Green")
    #showImg(B, cm_blue, "Blue")
    
    #Ex 5.3.1 - convert RGB to YCbCr
    img_ycbcr = rgb_to_ycbcr(padded_img)
    
    #Ex 5.3.2 - YCbCr channels with colormaps
    Y, Cb, Cr = splitChannels(img_ycbcr)
    #show_channels(Y, Cb, Cr, "Y", "Cb", "Cr")
    
    #Ex 6.3 - Downsampling 4:2:2
    Y_d, Cb_d, Cr_d = downsampling(img_ycbcr, 4, 2, 2, cv2.INTER_LINEAR)
    #show_channels(Y_d, Cb_d, Cr_d, "Y downsampling 4:2:2", "Cb downsampling 4:2:2", "Cr downsampling 4:2:2")
    
    #Ex 7.1.3 - DCT
    Y_dct = calculate_dct(Y_d)
    Cb_dct = calculate_dct(Cb_d)
    Cr_dct = calculate_dct(Cr_d)

    #Visualize DCT coefficients using a logarithmic transformation
    #show_channels(np.log(np.abs(Y_dct) + 0.0001), np.log(np.abs(Cb_dct) + 0.0001), np.log(np.abs(Cr_dct) + 0.0001), "Y_dct (log scale)", "Cb_dct (log scale)", "Cr_dct (log scale)")
    
    #Ex 7.3 - DCT on 64x64 blocks
    Y_dct64= block_dct(Y_d, 64)
    Cb_dct64= block_dct(Cb_d, 64)
    Cr_dct64= block_dct(Cr_d, 64)
    #show_channels(np.log(np.abs(Y_dct64) + 0.0001), np.log(np.abs(Cb_dct64) + 0.0001), np.log(np.abs(Cr_dct64) + 0.0001), "Y_dct 64x64 blocks", "Cb_dct 64x64 blocks", "Cr_dct 64x64 blocks")
    
    #Ex 7.2.3 and 8.3 - DCT on 8x8 and block quantization
    Y_dct8, Y_q = calc_block_dct_qnt(Y_d, 0, 8, qf)
    Cb_dct8, Cb_q = calc_block_dct_qnt(Cb_d, 1, 8, qf)
    Cr_dct8, Cr_q = calc_block_dct_qnt(Cr_d, 1, 8, qf)
    #show_channels(np.log(np.abs(Y_dct8) + 0.0001), np.log(np.abs(Cb_dct8) + 0.0001), np.log(np.abs(Cr_dct8) + 0.0001), "Y_dct 8x8 blocks", "Cb_dct 8x8 blocks", "Cr_dct 8x8 blocks")
    #show_channels(np.log(np.abs(Y_q) + 0.0001), np.log(np.abs(Cb_q) + 0.0001), np.log(np.abs(Cr_q) + 0.0001), "Y DCT quantized", "Cb DCT quantized", "Cr DCT quantized")
    
    print("Y_DCT:")
    print(Y_dct8[8:16, 8:16])
    
    print("\nY_QNT:")
    print(Y_q[8:16, 8:16])
    
    #Ex 9.3 - DPCM codification
    Y_dpcm = encode_dpcm(Y_q, 8)
    Cb_dpcm = encode_dpcm(Cb_q, 8)
    Cr_dpcm= encode_dpcm(Cr_q, 8)
    show_channels(np.log(np.abs(Y_dpcm) + 0.0001), np.log(np.abs(Cb_dpcm) + 0.0001), np.log(np.abs(Cr_dpcm) + 0.0001), "Y DPCM", "Cb DPCM", "Cr DPCM")
    
    print("\nY_DPCM:")
    print(Y_dpcm[8:16, 8:16])
    
    return Y, Y_dpcm, Cb_dpcm, Cr_dpcm, original_shape


#Decode functions ---------------------
def dpcm_decoding(differences, block_size):
    rows, cols = differences.shape
    channel = np.copy(differences)
    
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            i_ant, j_ant = i, j
            if j < block_size:
                i_ant = i - block_size
                j_ant = cols
            if i == 0 and j == 0:
                continue
            else:
                channel[i, j] = differences[i, j] + channel[i_ant, j_ant - block_size]
            
    return channel

def inverse_quantized_dct_blocks(Channel_q, block_size, quantization_matrix):
    rows, cols = Channel_q.shape
    inv_qnt_result = np.zeros((rows, cols))

    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            block = Channel_q[i:i+block_size, j:j+block_size]
            inv_qnt_block= np.round(block * quantization_matrix)
            inv_qnt_result[i: i + block_size, j: j + block_size] = inv_qnt_block

    return inv_qnt_result

def calc_block_idct(dct_result, block_size):
    rows, cols = dct_result.shape
    inv_dct_result = np.zeros((rows, cols))

    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            dct_block = dct_result[i: i + block_size, j: j + block_size]
            inv_dct_block= scipy.fftpack.idct(scipy.fftpack.idct(dct_block, norm='ortho').T, norm='ortho').T
            inv_dct_result[i: i + block_size, j: j + block_size] = inv_dct_block

    return inv_dct_result

def calculate_inverse_dct(dct_result):
    inverse_dct_result = scipy.fftpack.idct(scipy.fftpack.idct(dct_result, axis=0, norm='ortho'), axis=1, norm='ortho')
    return inverse_dct_result

def upsampling(Y, Cb_d, Cr_d, upsamp_factor_y, upsamp_factor_cb, upsamp_factor_cr, interpolation_func):
    #Get channels size
    h, w = Cb_d.shape[:2]
    
    #Determine the downsampling factor
    if upsamp_factor_cr == 0:
        dsf_w_cb = dsf_w_cr = dsf_h_cb = dsf_h_cr = upsamp_factor_cb / upsamp_factor_y
    else:
        dsf_w_cb = upsamp_factor_cb / upsamp_factor_y
        dsf_w_cr = upsamp_factor_cr / upsamp_factor_y
        dsf_h_cb = dsf_h_cr = 1
    
    #Resize Cb and Cr channels
    Cb_u = cv2.resize(Cb_d, (int(w / dsf_w_cb), int(h / dsf_h_cb)), interpolation_func)
    Cr_u = cv2.resize(Cr_d, (int(w / dsf_w_cr), int(h / dsf_h_cr)), interpolation_func)

    return Y, Cb_u, Cr_u

def joinChannels(C1, C2, C3):
    #Stack the channels along the last axis to form the original image
    img = np.stack((C1, C2, C3), axis=-1)
    return img

def ycbcr_to_rgb(img_ycbcr):
    # Inverse Conversion Matrix
    inverse_conversion_matrix = np.linalg.inv(np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]]))
    offset = np.array([0, 128, 128])

    # Subtract offset from the YCbCr image
    img_ycbcr_offset = img_ycbcr - offset
    # Apply Inverse Transformation
    img_rgb = np.dot(img_ycbcr_offset, inverse_conversion_matrix.T)

    # Round the values
    img_rgb = np.round(img_rgb)
    # Make sure the values are between [0, 255]
    img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

    return img_rgb

def unpad_image(padded_img, original_shape):
    return padded_img[:original_shape[0], :original_shape[1], :]


#Decoder ------------------------------
def decoder(Y_dpcm, Cb_dpcm, Cr_dpcm, original_shape, qf):
    
    print("\n- - - DECODER - - -")
    
    #Ex 9.4 - decoding DPCM
    Y_q = dpcm_decoding(Y_dpcm, 8)
    Cb_q = dpcm_decoding(Cb_dpcm, 8)
    Cr_q = dpcm_decoding(Cr_dpcm, 8)
    #show_channels(np.log(np.abs(Y_q) + 0.0001), np.log(np.abs(Cb_q) + 0.0001), np.log(np.abs(Cr_q) + 0.0001), "Y_q decoded", "Cb_q decoded", "Cr_q decoded")
    
    print("Y_IDPCM (Y_q):")
    print(Y_q[8:16, 8:16])
    
    #Ex 8.4 - dequantize DCT coefficients
    Y_dct8 = inverse_quantized_dct_blocks(Y_q, 8, get_quantization_matrix(0, qf))
    Cb_dct8 = inverse_quantized_dct_blocks(Cb_q, 8, get_quantization_matrix(1, qf))
    Cr_dct8 = inverse_quantized_dct_blocks(Cr_q, 8, get_quantization_matrix(1, qf))
    #show_channels(np.log(np.abs(Y_dct8) + 0.0001), np.log(np.abs(Cb_dct8) + 0.0001), np.log(np.abs(Cr_dct8) + 0.0001), "Y_dct8 decoded", "Cb_dct8 decoded", "Cr_dct8 decoded")
    
    print("\nY_IQ (Y_dct8):")
    print(Y_dct8[8:16, 8:16])
    
    #Ex 7.2.4 - DCT on 8x8 blocks DECODER
    Y_d = calc_block_idct(Y_dct8, 8)
    Cb_d = calc_block_idct(Cb_dct8, 8)
    Cr_d = calc_block_idct(Cr_dct8, 8)
    #show_channels(Y_d, Cb_d, Cr_d, "Y_d decoded", "Cb_d decoded", "Cr_d decoded")
    
    print("\nY_IDCT (Y_d):")
    print(Y_d[8:16, 8:16])
    
    #Ex 7.1.4 - un-dct
    #Y_d= calculate_inverse_dct(Y_dct)
    #Cb_d= calculate_inverse_dct(Cb_dct)
    #Cr_d= calculate_inverse_dct(Cr_dct)
    
    #Ex 6.4 - upsampling
    Y, Cb, Cr = upsampling(Y_d, Cb_d, Cr_d, 4, 2, 2, cv2.INTER_LINEAR)
    #show_channels(Y, Cb, Cr, "Y Channel Upsampled", "Cb Channel Upsampled", "Cr Channel Upsampled")
    
    #Ex 5.4 - convert YCbCr to RGB
    img_rgb = ycbcr_to_rgb(joinChannels(Y, Cb, Cr))
    #showImg(img_rgb, None, "Img RGB decoded")
    
    #Ex 4.2 - unpadding
    unpadded_img = unpad_image(img_rgb, original_shape)
    showImg(unpadded_img, None, "Img unpadded decoded")

    return Y, unpadded_img


#Differences --------------------------
def calculate_differences(Y_init, Y_rec):
    difference_image = np.abs(Y_init - Y_rec)
    showImg(difference_image, 'gray', "Differences")
    
    print("\n- - - METRICS - - -")
    
    #Maximum and average differences
    diff = np.abs(Y_init - Y_rec)
    max_diff = np.max(diff)
    avg_diff = np.mean(diff)
    print("Max diff: " + str(max_diff))
    print("Avg diff: " + str(avg_diff))

def metrics(img_init, img_rec):
    diff = np.abs(img_init - img_rec)
    nl, nc, _= img_init.shape
    
    #Mean Square Error (MSE)
    mse = np.sum(np.square(diff))/(nl*nc)
    print("MSE = " + str(mse))
    
    #Root Mean Square Error (RMSE)
    rmse = np.sqrt(mse)
    print("RMSE = " + str(rmse))
    
    #Signal-to-Noise Ratio (SNR)
    p= np.sum(np.square(img_init))/(nl*nc)
    snr = 10 * np.log10(p/mse)
    print("SNR = " + str(snr))
    
    #Peak Signal-to-Noise Ratio (PSNR)
    psnr = 10 * np.log10(np.max(img_init) ** 2 / mse)
    print("PSNR = " + str(psnr))
    
def differences(img):
    
    #Ex 10.1 a 10.4 - encode/decode images and show differences
    for qf in [75]: #10, 25, 50, 75, 100
        print("QF= " + str(qf) + "\n")
        Y_init, Y_dpcm, Cb_dpcm, Cr_dpcm, original_shape= encoder(img, qf)
        Y_rec, img_rec = decoder(Y_dpcm, Cb_dpcm, Cr_dpcm, original_shape, qf)
        calculate_differences(Y_init, Y_rec)
        metrics(img.astype(float), img_rec.astype(float))


#Main ---------------------------------
def main():
    
    img1_airport = plt.imread("assets/airport.bmp")
    showImg(img1_airport, None, "Imagem original: airport.bmp")
    print("Imagem 1 - airport.bmp - - - - - - - - - - - - -\n")
    differences(img1_airport)
    """
    img2_geometric = plt.imread("assets/geometric.bmp")
    showImg(img2_geometric, None, "Imagem original: geometric.bmp")
    print("\nImagem 2 - geometric.bmp - - - - - - - - - - - - -\n")
    differences(img2_geometric)
    
    img3_nature = plt.imread("assets/nature.bmp")
    showImg(img3_nature, None, "Imagem original: nature.bmp")
    print("\nImagem 3 - nature.bmp - - - - - - - - - - - - - -\n")
    differences(img3_nature)
    """
    
    #print(img.shape) #img size
    #print(img.dtype) #img type
    print("\nThe End - - - - - - - - - - - - - - - - - - - -\n")


if __name__ == "__main__":
    main()
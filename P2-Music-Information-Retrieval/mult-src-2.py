#MULT
import librosa
import numpy as np
import pandas as pd
import scipy.signal.windows
from scipy.stats import skew, kurtosis
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import os
import math


#----------------------------------------------
#2.1.1 - Função para extrair as características
def extract_features(file_path):
    y, fs = librosa.load(file_path, sr=22050, mono=True)
    statistics_matrix = np.zeros(190)  # num_features, n_statistics

    feature = librosa.feature.mfcc(y=y, n_mfcc=13)  # mfcc
    previous_feature_column = calc_statistics(statistics_matrix, feature, 0)

    feature = librosa.feature.spectral_centroid(y=y)  # centroid
    previous_feature_column = calc_statistics(statistics_matrix, feature, previous_feature_column)

    feature = librosa.feature.spectral_bandwidth(y=y)  # bandwidth
    previous_feature_column = calc_statistics(statistics_matrix, feature, previous_feature_column)

    feature = librosa.feature.spectral_contrast(y=y)  # contrast
    previous_feature_column = calc_statistics(statistics_matrix, feature, previous_feature_column)

    feature = librosa.feature.spectral_flatness(y=y)  # flatness
    previous_feature_column = calc_statistics(statistics_matrix, feature, previous_feature_column)

    feature = librosa.feature.spectral_rolloff(y=y)  # rolloff
    previous_feature_column = calc_statistics(statistics_matrix, feature, previous_feature_column)

    feature = librosa.yin(y, fmin=20, fmax=fs / 2)  # f0
    feature[feature == fs / 2] = 0
    previous_feature_column = calc_statistics(statistics_matrix, feature, previous_feature_column)

    feature = librosa.feature.rms(y=y)  # rms
    previous_feature_column = calc_statistics(statistics_matrix, feature, previous_feature_column)

    feature = librosa.feature.zero_crossing_rate(y)  # zero_crossing_rate
    previous_feature_column = calc_statistics(statistics_matrix, feature, previous_feature_column)

    feature = librosa.feature.tempo(y=y, sr=fs)  # tempo
    previous_feature_column = calc_statistics(statistics_matrix, feature, previous_feature_column)

    return statistics_matrix

#2.1.2 - Função para calcular estatísticas
def calc_statistics(statistics_matrix, feature, previous_feature_column):
    if feature.ndim == 1:
        if feature.shape[0] > 1:
            feature = feature[np.newaxis, :]
        else:
            return feature

    num_features = feature.shape[0]
    lastcolumn = 0  # lastcolumn used by 1 iteration of the same feature

    for i in range(num_features):
        statistics_matrix[previous_feature_column + lastcolumn + 1] = np.mean(feature[i])
        statistics_matrix[previous_feature_column + lastcolumn + 2] = np.std(feature[i])
        statistics_matrix[previous_feature_column + lastcolumn + 3] = skew(feature[i])
        statistics_matrix[previous_feature_column + lastcolumn + 4] = kurtosis(feature[i])
        statistics_matrix[previous_feature_column + lastcolumn + 5] = np.median(feature[i])
        statistics_matrix[previous_feature_column + lastcolumn + 6] = np.max(feature[i])
        statistics_matrix[previous_feature_column + lastcolumn + 7] = np.min(feature[i])

        lastcolumn += 7

    return previous_feature_column + (num_features * 7)  # equivalent to previous_feature_column

#2.1.3 - Função para normalizar as features no intervalo [0, 1]
def normalize_features(features_matrix, min_values, max_values):
    normalized_features = np.zeros_like(features_matrix)
    for i in range(features_matrix.shape[1]):
        min_val = min_values[i]
        max_val = max_values[i]
        if min_val == max_val:
            normalized_features[:, i] = 0
        else:
            normalized_features[:, i] = (features_matrix[:, i] - min_val) / (max_val - min_val)
    return normalized_features

def query_normalize_features(query_features, min_val, max_val):
    normalized_query_features = np.zeros_like(query_features)
    
    for i in range(len(query_features)):
        min_val = min_values[i]
        max_val = max_values[i]
        if min_val == max_val:
            normalized_query_features[i] = 0
        else:
            normalized_query_features[i] = (query_features[i] - min_val) / (max_val - min_val)
    
    return normalized_query_features

#2.2.1 - Função para calcular o spectral centroid de raíz
def calculate_spectral_centroid(file_path, window_length=2048, hop_length=512):
    # Load audio file and extract signal (y) and sampling rate (fs)
    y, fs = librosa.load(file_path, sr=22050, mono=True)

    # Calculate the spectral centroid using librosa function
    centroid_librosa = librosa.feature.spectral_centroid(y=y, sr=fs, window=scipy.signal.windows.hann)
    centroid_librosa = centroid_librosa[0]

    centroid_length = math.ceil((len(y) - window_length) / hop_length) + 1

    y_frames = [[] for _ in range(centroid_length)]
    for i in range(0, centroid_length):
        y_frames[i] = y[(i * hop_length): (i * hop_length) + window_length]

    y_frames[-1] = np.pad(y_frames[-1], (0, window_length - len(y_frames[-1])))

    # Calculate the spectral centroid manually
    centroids_manual = np.zeros(centroid_length)

    for i in range(0, centroid_length):

        hann_window = np.hanning(window_length)
        y_hann = y_frames[i] * hann_window

        magnitude_frame = np.abs(np.fft.rfft(y_hann))
        freq_frame = np.arange(0, fs / 2 + 1, fs / window_length)  # len(magnitude_frame)

        # proctect from division by 0
        magnitude_frame_sum = np.sum(magnitude_frame)
        if magnitude_frame_sum != 0:
            centroid_dev = sum(magnitude_frame * freq_frame) / np.sum(magnitude_frame)  # Weighted mean
        else:
            centroid_dev= 0
        
        centroids_manual[i] = (centroid_dev)

    centroid_librosa = centroid_librosa[2:]

    if len(centroid_librosa) > len(centroids_manual):
        centroid_librosa = centroid_librosa[:-(len(centroid_librosa) - len(centroids_manual))]
    elif len(centroid_librosa) < len(centroids_manual):
        centroids_manual = centroids_manual[:-(len(centroids_manual) - len(centroid_librosa))]

    return centroid_librosa, centroids_manual

#2.2.2 - Função para calcular o coeficiente de correlação de Pearson e o erro quadrático médio (RMSE)
def calculate_metrics(centroid_librosa, centroid_dev):
    pearson_corr, _ = pearsonr(centroid_librosa, centroid_dev)
    rmse = np.sqrt(mean_squared_error(centroid_librosa, centroid_dev))

    print("Pearson correlation:", pearson_corr)
    print("RMSE:", rmse)
    return pearson_corr, rmse


#----------------------------------------------
#3.1 - Função para calcular a similaridade utilizando diferentes métricas de distância
def similarity_metrics(x, y):
    euclidean_distance = np.linalg.norm(x - y)
    manhattan_distance = np.sum(np.abs(x - y))
    cosine_similarity = 1 - (np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    
    return euclidean_distance, manhattan_distance, cosine_similarity
 

#----------------------------------------------
#4.1.1 - Função para o ranking top10
def calculate_context_similarity(database_metadata, query_metadata):
    num_lines = len(database_metadata)
    num_columns = len(database_metadata.columns)
    similarity_matrix = np.zeros((num_lines, 1))

    for i in range(num_lines):
        counter = 0

        if database_metadata['Artist'].values[i] == query_metadata['Artist'].values[0]:
            counter += 1

        genres_database = database_metadata['GenresStr'].values[i].split('; ')
        genres_query = query_metadata['GenresStr'].values[0].split('; ')
        #np.intersect1d - pode ser usada para comparar os ficheiros
        
        for genre in genres_database:
            if genre in genres_query:
                counter += 1

        moods_database = database_metadata['MoodsStrSplit'].values[i].split('; ')
        moods_query = query_metadata['MoodsStrSplit'].values[0].split('; ')
        for mood in moods_database:
            if mood in moods_query:
                counter += 1
        
        similarity_matrix[i, 0] = counter

    return similarity_matrix

#4.1.2 - Função para calcular a precision
def calculate_precision(distance_ranking, similarity_ranking):
    num_relevant_songs = 0
    num_recommended_songs = 10

    for song in distance_ranking:
        if song in similarity_ranking:
            num_relevant_songs += 1

    precision = (num_relevant_songs / num_recommended_songs) * 100

    return precision
  

#----------------------------------------------
if __name__ == "__main__":

    music_folder = "./Queries/AllMusic"
    files = os.listdir(music_folder)
    files.sort()
    num_files = len(files)
    features_matrix = np.zeros((num_files + 2, 190))
    centroid_matrix = np.zeros((num_files, 2))
    index = 2
    centroid_index= 0
    
    #2.1 ----------
    # Extract features of the 900 files
    for file_name in range(num_files):
        file_path = os.path.join(music_folder, files[file_name])
        features = extract_features(file_path)
        features_matrix[index] = features
        index += 1
        print(index)

    # Calcular e adicionar os valores min e max de cada feature
    min_values = np.min(features_matrix[2:, :], axis=0)
    max_values = np.max(features_matrix[2:, :], axis=0)
    features_matrix[0, :] = min_values
    features_matrix[1, :] = max_values

    # Save to file
    np.savetxt("features.csv", features_matrix, delimiter=",")

    # Normalizar as features
    normalized_features = normalize_features(features_matrix[2:, :], min_values, max_values)
    features_matrix[2:] = normalized_features

    # Save to file
    np.savetxt("normalized_features.csv", features_matrix, delimiter=",")
    
    
    #2.2 ----------
    # Calcular spectral centroid
    for file_name in range(num_files):
        file_path = os.path.join(music_folder, files[file_name])
        # Calcular o spectral centroid
        centroid_librosa, centroid_dev = calculate_spectral_centroid(file_path)

        # Calcular as métricas de erro
        print("Music nº", file_name)
        pearson_corr, rmse = calculate_metrics(centroid_librosa, centroid_dev)
        centroid_matrix[centroid_index, 0] = pearson_corr
        centroid_matrix[centroid_index, 1] = rmse
        centroid_index += 1

    # Save to file
    np.savetxt("metrics_centroid.csv", centroid_matrix, delimiter=",")
    
    #3.1 e 3.2 ----------
    features_matrix = np.loadtxt("./features.csv", dtype=float, delimiter=',')
    min_values = np.min(features_matrix[2:, :], axis=0)
    max_values = np.max(features_matrix[2:, :], axis=0)
    features_matrix[0, :] = min_values
    features_matrix[1, :] = max_values
    normalized_features = normalize_features(features_matrix[2:, :], min_values, max_values)
    features_matrix[2:] = normalized_features

    query_path = "./Queries/OneMusic/MT0000414517.mp3"
    query_features = extract_features(query_path)
    normalized_query_features = query_normalize_features(query_features, min_values, max_values)

    # Distâncias
    euclidean_distance = np.zeros(num_files)
    manhattan_distance = np.zeros(num_files)
    cosine_distance = np.zeros(num_files)
    # Remover maxs e mins
    features_matrix = features_matrix[2:]
    
    for music_n in range(num_files):
        euclidean_distance[music_n], manhattan_distance[music_n], cosine_distance[music_n] = np.array(similarity_metrics(features_matrix[music_n], normalized_query_features))

    # Save to file
    np.savetxt("similarity_metrics.csv", np.column_stack((euclidean_distance, manhattan_distance, cosine_distance)), delimiter=",")
    
    
    #3.3 ----------
    num_rankings = 11
    file_text = ""

    # Get the top 10 rankings for each similarity metric
    euclidean_ranking = np.argsort(euclidean_distance)[1:num_rankings]
    manhattan_ranking = np.argsort(manhattan_distance)[1:num_rankings]
    cosine_ranking = np.argsort(cosine_distance)[1:num_rankings]

    # Print the top 10 rankings for each similarity metric
    file_text += "- Rankings:\n"
    file_text += "\nRanking: Euclidean-------------\n"
    file_text += str(np.asarray([files[index] for index in euclidean_ranking]))
    file_text += "\n\ndist =" + str(np.asarray([euclidean_distance[index] for index in euclidean_ranking]))
    file_text += "\n\nRanking: Manhattan-------------\n"
    file_text += str(np.asarray([files[index] for index in manhattan_ranking]))
    file_text += "\n\ndist =" + str(np.asarray([manhattan_distance[index] for index in manhattan_ranking]))
    file_text += "\n\nRanking: Cosine-------------\n"
    file_text += str(np.asarray([files[index] for index in cosine_ranking]))
    file_text += "\n\ndist =" + str(np.asarray([cosine_distance[index] for index in cosine_ranking]))
    
    
    #4.1.1 ----------
    
    # Load metadata from CSV files
    database_metadata = pd.read_csv("panda_dataset_taffc_metadata.csv")
    query_metadata = pd.read_csv("query_metadata.csv")

    # Calculate context-based similarity matrix
    similarity_matrix = calculate_context_similarity(database_metadata, query_metadata)
    similarity_ranking = np.argsort(similarity_matrix.flatten())[::-1]
    similarity_ranking= (similarity_ranking.flatten())[1:num_rankings]

    # Print context-based similarity ranking
    file_text += "\n\nRanking: Metadata-------------\n"
    file_text += str(np.asarray([files[index] for index in similarity_ranking]))
    file_text += "\nscore =" + str([similarity_matrix[index][0] for index in similarity_ranking])
    
    # Save similarity matrix to file
    np.savetxt("context_similarity_matrix.csv", similarity_matrix, delimiter=",")

    #4.1.2 ----------
    # Calculate precision for each similarity metric
    file_text += "\n\nPrecision de: " + str(calculate_precision(euclidean_ranking, similarity_ranking))
    file_text += "\nPrecision dm: " + str(calculate_precision(manhattan_ranking, similarity_ranking))
    file_text += "\nPrecision dc: " + str(calculate_precision(cosine_ranking, similarity_ranking))
    
    # Save rankings to file
    with open("ranks_precision.txt", "w") as file:
        file.write(file_text)
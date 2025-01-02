import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from itertools import combinations
import chardet

# This code assumes that IR spectra are saved as .txt files in a single directory.
# It also assumes that the .txt files contain wavenumbers in the first column and absorbance in the second column.

# Define the directory path
directory_path = r'your_file_path_goes_here'

# Define thresholds
peak_threshold = 0.3  # Adjust as necessary. 
similarity_threshold = 0.3  # Adjust as necessary. 

def plot_ir_spectra_and_annotate_peaks(directory, peak_threshold):
    """
    Plot IR spectra, detect peaks, and annotate them.

    Parameters:
    directory (str): The directory path containing the spectra files.
    peak_threshold (float): The threshold for detecting peaks.

    Returns:
    dict: A dictionary containing the peaks data for similarity calculation.
    """
    spectra_data = {}

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check if it is a file and has a .txt extension
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            try:
                # Detect encoding
                with open(file_path, 'rb') as f:
                    result = chardet.detect(f.read())
                encoding = result['encoding']

                # Fallback to 'latin1' if encoding detection fails
                if encoding is None:
                    encoding = 'latin1'

                # Read the content of the file using numpy with the detected encoding
                data = np.loadtxt(file_path, encoding=encoding)

                # Extract wavenumber and absorbance columns
                wavenumbers = data[:, 0]
                absorbances = data[:, 1]

                # Detect peaks
                peaks, _ = find_peaks(absorbances, height=peak_threshold)

                # Plot the data
                plt.figure()
                plt.plot(wavenumbers, absorbances, label=filename)
                plt.plot(wavenumbers[peaks], absorbances[peaks], "x", label='Peaks')

                # Annotate each peak
                for peak in peaks:
                    plt.annotate(f'{wavenumbers[peak]:.2f}',
                                 (wavenumbers[peak], absorbances[peak]),
                                 textcoords="offset points",
                                 xytext=(0, 5),
                                 ha='center')

                plt.xlabel(r'Wavenumber (cm$^{-1}$)')
                plt.ylabel('Absorbance (a.u)')
                plt.title(f'IR Spectrum with Peaks - {filename}')
                plt.legend(frameon=False)

                # Show the plot
                plt.show()

                # Store the peaks data for similarity calculation
                spectra_data[filename] = set(wavenumbers[peaks])

                print(f'Plotted {filename} with peaks')

            except Exception as e:
                # Print the error message and continue with the next file
                print(f'Error processing {filename}: {e}')

    return spectra_data

def jaccard_similarity(set1, set2):
    """
    Calculate the Jaccard similarity index between two sets.

    Parameters:
    set1 (set): The first set of peaks.
    set2 (set): The second set of peaks.

    Returns:
    float: The Jaccard similarity index.
    """
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

def calculate_and_display_similarity(spectra_data, similarity_threshold):
    """
    Calculate spectra similarity indices and display results.

    Parameters:
    spectra_data (dict): A dictionary containing the peaks data for each spectrum.
    similarity_threshold (float): The threshold for determining similar spectra.
    """
    similarity_data = {}

    # Get all combinations of spectra pairs
    spectra_pairs = list(combinations(spectra_data.keys(), 2))

    # Calculate the Jaccard similarity index for each pair
    for spec1, spec2 in spectra_pairs:
        set1 = spectra_data[spec1]
        set2 = spectra_data[spec2]
        similarity = jaccard_similarity(set1, set2)
        similarity_data[(spec1, spec2)] = similarity
        print(f'Jaccard similarity index between {spec1} and {spec2}: {similarity:.2f}')

    # Display similarity results based on the similarity_threshold
    similar_spectra = {}
    for spec1, spec2 in similarity_data:
        if similarity_data[(spec1, spec2)] >= similarity_threshold:
            if spec1 not in similar_spectra:
                similar_spectra[spec1] = []
            similar_spectra[spec1].append(spec2)

    # Print similar spectra statement
    print(f"Similar spectra according to Jaccard index with a threshold of {similarity_threshold}:")
    for spec, similar_list in similar_spectra.items():
        print(f'{spec} is similar to {", ".join(similar_list)}')

    # Plot similar spectra
    for spec, similar_list in similar_spectra.items():
        plt.figure()
        colors = plt.cm.tab10(np.linspace(0, 1, len(similar_list) + 1))
        for i, similar_spec in enumerate([spec] + similar_list):
            file_path = os.path.join(directory_path, similar_spec)
            try:
                # Detect encoding
                with open(file_path, 'rb') as f:
                    result = chardet.detect(f.read())
                encoding = result['encoding']

                # Fallback to 'latin1' if encoding detection fails
                if encoding is None:
                    encoding = 'latin1'

                # Read the content of the file using numpy with the detected encoding
                data = np.loadtxt(file_path, encoding=encoding)

                # Extract wavenumber and absorbance columns
                wavenumbers = data[:, 0]
                absorbances = data[:, 1]

                plt.plot(wavenumbers, absorbances, label=similar_spec, color=colors[i])
            except UnicodeDecodeError as e:
                # Print the error message and continue with the next file
                print(f'Error processing {similar_spec}: {e}')
            except Exception as e:
                # Print the error message and continue with the next file
                print(f'Error processing {similar_spec}: {e}')

        plt.xlabel(r'Wavenumber (cm$^{-1}$)')
        plt.ylabel('Absorbance (a.u)')
        plt.title(f'Similar Spectra - {spec}')
        plt.legend(frameon=False)
        plt.show()

# Call the functions
spectra_data = plot_ir_spectra_and_annotate_peaks(directory_path, peak_threshold)
calculate_and_display_similarity(spectra_data, similarity_threshold)

# IR-spectra-handling 

This project analyzes IR spectra saved as `.txt` files in a single directory. It assumes that the .txt file contains wavenumbers in the first column, and absorbance values in the second one. The analysis includes plotting the spectra, detecting peaks, annotating them, and calculating the Jaccard similarity index to determine the similarity between spectra.

Example of a spectrum ploted automatically can be found in Figure 1: 

![image](https://github.com/user-attachments/assets/434a80f8-397f-43f9-9c46-1800495ebb8a)

Figure 1. IR spectra plotted automatically based on the .txt file

## Limitations

While the method used in this project may not be 100% accurate, it effectively detects spectra with similarities based on the wavenumbers of their peaks. An example of two spectra found to be similar is illustrated in Figure 2.

![image](https://github.com/user-attachments/assets/d9366255-1dd3-461c-ba87-9d475caf3387)

Figure 2. Plots of spectra found to be similar based on the wavenumbers of their peaks.


## Features

- Plot IR spectra with annotated peaks.
- Detect peaks in the absorbance data.
- Calculate the Jaccard similarity index between spectra based on peak wavenumbers.
- Display and plot similar spectra.



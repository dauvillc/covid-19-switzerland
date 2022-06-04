# A model for the SARS-CoV-2 epidemic waves in Switzerland based on socio-economic variables
EPFL TRANSP-OR Lab - Professor Michel Bierlaire. Semester project hosted by Cloe Cortes Balcells and Silvia Varotto.  
[Report](https://drive.google.com/file/d/1sBhczMEXTnYg4U6B-3ubErjL8FIK8Car/view?usp=sharing).

## Repo organisation
* ```model```: Python3 package that implements the SIR model, the calibration process, and visualisation.
* ```scripts```: notebooks in which all analysis and results are implemented and documented. ```extract_people_df.ipynb``` was used to process the MATSIM XML output file; ```contact_matrices.ipynb``` is in two parts: computation of the individual contact matrices from the mobility data, and socioeconomic data analysis / machine learning problem. Finally, ```calibration.ipynb``` implements the calibration of the model using the FOPH data.
* ```figures```, ```data```: directories that contain the figures produced by the notebooks, and the datasets (MATSIM, processed matsim, contact matrices, FOPH data, ...). The data is not included on the online Github repo because of its size and especially confidentiality.
* ```furnished```: contains any code that was provided by the lab.
* ```tests```: contains test scripts for the model package.

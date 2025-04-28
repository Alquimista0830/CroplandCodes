# CroplandCodes
The processing codes for "Xizang_cropland_parcels" dataset

%1---preprocessing.py
Including TIFF read, morphological opening-and-closing operation, and TIFF write process.
Input: path of TIFFs the user need.
Output: path to output the TIFFs after preprocessing. 

%2--- mosaic.py
Mosaic all TIFFs covering the region user needed, then build pyramid.
Before running, a TXT file should be prepared, which include the filename list of all TIFFs. e.g. "filelist2020.txt"

%3---open the image 
Then the datasets could be visualized in ArcGIS or QGIS software.

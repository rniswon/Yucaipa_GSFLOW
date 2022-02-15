Future scenario models for Yucaipa Vally Watershed Integrated Hydrologic Model
-----------------------------------------------------------
This folder contains nine models used to extend the published YIHM.
Eight models extend the model from 2014 through 2099 using two GHG scenarios 
(RCP45 and RCP85) each from four different GCM models. The last model is a sin-up 
model that is used to save model heads and PRMS input variables at the beginning 
of 2013. The climate scenario models start at the beginning of 2013 and use the 
spin-up model data as starting values. 

All MODFLOW and PRMS input files are shared by the scenario models except the 
PRMS data file that contains the climate data for each scenario. All output files 
for each scenario are stored locally. The spin-up model does not use shared files
because it has a different model period.

FOLDERS:
-----------------------------------------------------------
bin: contains model executable file gsflow_new.exe
external_files: shared MODFLOW input files and startup files
external_input: shared PRMS input files

CanESM2_rcp45: model CanESM2 RCP45 GHG scenario model
CanESM2_rcp85: model CanESM2 RCP85 GHG scenario model
CNRMCM5_rcp45: model CNRMCM5 RCP45 GHG scenario model
CNRMCM5_rcp85: model CNRMCM5 RCP85 GHG scenario model
HadGEM2ES_rcp45: model HadGEM2ES rcp45 GHG scenario model
HadGEM2ES_rcp85: model HadGEM2ES rcp85 GHG scenario model
MIROC5_rcp45: model MIROC5 rcp45 GHG scenario model
MIROC5_rcp85: model MIROC5 rcp85 GHG scenario model

Each model folder has the following subfolders:
	input: contains the PRMS data file for the model
	output: MODFLOW output files
	output2: PRMSand GSFLOW output files
-----------------------------------------------------------

FILES
-----------------------------------------------------------
Each model can be run using a batch file in the root model folder: gsflow_yuc.bat
Each model has a GSFLOW control file and MODFLOW nam file with the 
GCM model name and GHG scenario:
Yuc_gsflow_<GCM name>_<RCP scenario>.control
yuc_<GCM name>_<RCP scenario>.nam

Each model also has a PRMS data file in its input folder named in a similar manner
Yuc_<GCM name>_<RCP scenario>.data


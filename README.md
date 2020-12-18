# FUCCI Decoder

FUCCI Decoder, for automatically obtaining high-throughput cell cycle quantitative information at single cell-level with live-cell imaging on FUCCI cell line as input. 


## Usage

1. Download the project, navigate to project repository.
2. Activate python environment with requirements installed.  (see requirements.txt)
3. Check python version 3.6 (software developed under 3.6.10)
4. To run single dataset:
   - main.py -g <GFP image> -m <mCherry image> -d <DIC image> -o <output directory>
5. To run in batch:
   -i <input directory> -o <output directory>  
  
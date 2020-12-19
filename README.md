# FUCCI Decoder

FUCCI Decoder, for automatically obtaining high-throughput cell cycle quantitative information at single cell-level with live-cell imaging on FUCCI cell line as input. 


## Usage

1. Download the project, navigate to project repository.
2. Activate python environment with requirements installed.  (see requirements.txt)
3. Check python version 3.6 (software developed under 3.6.10)
4. To run single dataset (images should be in tif stack format with same prefix and dimensionality (txy/tyx)):
   ```
   python main.py -g <GFP image> -m <mCherry image> -d <DIC image> -o <output directory>

5. To run in batch (images should be in tif stack format with same prefix and dimensionality (txy/tyx)):
   ```
   python main.py -i <input directory> -o <output directory> 

6. Customize behavior:
   ```
   -v: verbose.
   ```
   
   Time frame threshold for associating cell-cell relationships, default is 5 frames. Tracks with gaps smaller than the threshold will be joined.
   ```
   -t <int>
   ```

   Distance threshold for associating cell-cell relationships, default is 90 pixels. Objects in two consecutive frames with distance smaller than threshold will be associated. When detecting cell division, the threshold is amplified by 1.5.
   ```
   -f <int>
   ```
## Run demo

Demo data is a set of short FUCCI image in tif format.
```
python main.py -g ./softwareData/demo_GFP.tif  -m ./softwareData/demo_mCherry.tif -d ./softwareData/demo_DIC.tif -o ./softwareData/demoOutput/ -v -t 5 -f 90
```
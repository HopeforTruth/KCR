# KCR(Multivariate Time Series Anomaly Detection with K-distance Calibrated Reconstruction)
## Code Description
There are 11 files/folders in the source.(file named ".gitkeep" indicates empty Folder)
- ablation models: The models in ablation experiments are stored in this foldr
- dataset: The original dataset folder.
- hyper parameters: We save the parameters in this folder.
- image: Figures and tables of KCR.
- model: The net work of KCR is defined here.
- processed dataset: Where we store and load processed dataset.
- savedModel: The trained model will be automatically saved here, according to its parameter.
- scripts: Experiment scripts. You can reproduce the result by these scripts.
- tools: Some tools are used in KCR.
- main.py: The main python file. You can run the experiment directly with main.py too.
- preprocess.py: Preprocess the dataset before running scripts.
- requirements.txt: Python packages needed to run this repo.

## Dataset Access
Download datasets, rename folder, and put them into folder 'dataset' in this project.
- SMD: Access <https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset>, rename the folder into SMD
- SMAP&MSL: Run scripts below, and rename the folder form 'data' to 'SMAPMSL'
  ```bash
    wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip
    cd data && wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
   ```
- SWat: Check official website <https://itrust.sutd.edu.sg/itrust-labs_datasets/>. Click "Requests for datasets", fill the request table, and wait official email. Name the folder with  'SWat'.
- PSM: Access <https://github.com/eBay/RANSynCoders/tree/main/data>, download datasets and name the folder with PSM.
## Reproduce
This project use pytorch-2.0.1+cu118 to construct the model.To reproduce the results, you are supposed to follow the steps below.
1. Install packages in requirements.txt
```bash
pip install -r requirements.txt
```
2. Put original datasets in dataset folder, and run main function in preprocess.py
```bash
python preprocess.py
```
3. Train and evaluate on specific datasets. Run scripts or manually run main.py.
```bash
bash ./scripts/SMD.sh
bash ./scripts/PSM.sh
bash ./scripts/MSL.sh
bash ./scripts/SMAP.sh
bash ./scripts/SWat.sh
```
4. Ablation experiments through scripts.
```bash
bash ./scripts/Abalation1.sh
bash ./scripts/Abalation2.sh
```


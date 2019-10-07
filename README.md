# Semina Breast Classifer

This is a Project from Semina which seeks to create AIS (Artificical Intelligent Systems) which enables pathologist and medical experts to be able to classify a woman to be etheir be Benign or Malignant when it comes to being Diagnosed for Breast Cancer Either by FNA Biopsy (Fine Needle Aspiration) or Mammography. The DataSets for this project comes makes use of Data from Wincosin which is FNA  based Data of cells, DDSM and MIAS which are Dataset obtained from Mammography which in retrospect are mammograms on is in form of serialized tfrecords and the later in bpm (bitmap) Images.

The project is highly experiment this is a guide to Semina clinically implemement the AIS after overcoming the obstacles meet in classification and also coming up with a standard way of obtaining either standardised Images from MRI and also having standard input parameters for FNA Biopsy.

The project is being developed by DeveloperPrince being guided by Advisory Pathologist who's names shall not be exposed in this documentation.

## Requirements

1.) Virtual Machine or HPC with the following Specifications:  
    1.) 8 Gig Ram
    2.) CPU: Intel Xeon, Intel i3 2.5 GHz QuadProcessor, Intel i5 2.3 GHz QuadProcessor, Intel i7        2.3 GHz QuadProcessor (either of the processors as a minimum requirement)
    3.) GPU: - NVidia P800 (4GB dedicated video memory)
    4.) Storage: 40GB HDD preferable SDD.

2.) Programing Languages:
    1.) Python
    2.) R(optional)

2.) Core Libraries and Frameworks:
    1.) Tensorflow
    2.) Theano
    3.) CNTK
    2.) Pandas
    3.) Numpy
    4.) Keras

3.) Optional Libraries:
    1.) OpenCV

4.) Tools:
    1.) Anaconda (Optional)
    2.) Jupyter Notebooks , Azure Jupyter Notebooks, Kaggle Jupyter Notebooks (either of the listed)
    3.) AWS Virtual Instance, GCP VM(virtual machine), Azure VM(Virual Machine) (either of the           listed)

## Configuration

    The project assumes that you have experience with Python and Basics in ML & NN(Machine Learning and Nueral Networks). It assumes you have basic or an appreciation into statical mathematical computation and Data Science and Mathematics. You should have knowledge in Cancer and its behaviour. 

    NB: Make sure you switch to virtual environment to set the project

    The configurations and setup shall be listed below:

### Install and Setup

    Before you get started in Developing your AI models for Semina please make sure you meet the following requirements
    1. Anaconda
    2. Python3.5 or better

    If you don't have Python & Anaconda set up please visit the [Python Website](https://www.python.org/downloads/) & [Anaconda Website](https://www.anaconda.com/distribution/)

    1. If you meet the above requirements ensure you create the conda environments

    ```bash 
    conda create --n semina-ai python=3.7
    ```
    or if you desire to create a conda environment with a specific python version like 3.6 (which I recommend)

    ```bash 
    conda create -n semina-ai python=3.6
    ```

    2. Activate your conda environment

    ```bash 
    conda activate semina-ai
    ```

    3. Install Python Packages

    ```bash
    pip install -r requirements.txt
    ```

    After these three steps you are good to go.

## Architecture

```bash
semina_medical_ml_workshop
|___FNA
|   |   create_dataset.py
|   |   generate_model.py
|___input
|___Mammography
|   |   create_dataset.py
|   |   generate_model.py
|___Research
|   .gitignore
|   classifier_ddsm.py
|   classifier_mias.py
|   classifier_wdbc.py
|   classifier_wdbc1.py
|   README.md
|   requirements.tx
```
### Data Data

    Data links shall provided below for the used data sets.

#### Wisconsin Data

https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

#### MIAS 

https://www.kaggle.com/kmader/mias-mammography/downloads/mias-mammography.zip/3



### Contact Details

    The project is for Semina to justify their case and eventually create an entprise solution for clinical implementation. Contact DevelopPrince for more information: +263 716 661 298 / 
    princekudzaimaposa94@gmail.coms

 
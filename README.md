# Semina Breast Classifer

This is a Project from Semina which seeks to create AIS (Artificial Intelligent Systems) which enables pathologist and medical experts to be able to classify a woman to be either be Benign or Malignant when it comes to being Diagnosed for Breast Cancer Either by FNA Biopsy (Fine Needle Aspiration) or Mammography. The DataSets for this project comes makes use of Data from Wincosin which is FNA  based Data of cells, DDSM and MIAS which are Data set obtained from Mammography which in retrospect are mammograms on is in form of serialized tfrecords and the later in bpm (bitmap) Images.

The project is highly experiment this is a guide to Semina clinically implement the AIS after overcoming the obstacles meet in classification and also coming up with a standard way of obtaining either standardised Images from MRI and also having standard input parameters for FNA Biopsy.

The project is being developed by DeveloperPrince being guided by Advisory Pathologist who's names shall not be exposed in this documentation.

## Requirements

1.) Virtual Machine or HPC with the following Specifications:  
    1.) 4 Gig or 8 Gig Ram (16 Gig preferable)
    2.) CPU: Intel Xeon, Intel i3 2.5 GHz QuadProcessor, Intel i5 2.3 GHz QuadProcessor, Intel i7        2.3 GHz QuadProcessor (either of the processors as a minimum requirement)
    3.) GPU: - NVidia P800 (4GB dedicated video memory) (optional)
    4.) Storage: 40GB HDD preferable SDD.

2.) Programing Languages:
    1.) Python
    2.) R(optional)

2.) Core Libraries and Frameworks:
    1.) Tensorflow (optional)
    2.) Theano
    3.) CNTK (optional)
    2.) Pandas
    3.) Numpy
    4.) Keras

3.) Secondary Libraries:
    1.) OpenCV

4.) Tools:
    1.) Anaconda 
    2.) Jupyter Notebooks , Azure Jupyter Notebooks, Kaggle Jupyter Notebooks (either of the listed)
    3.) AWS Virtual Instance, GCP VM(virtual machine), Azure VM(Virtual Machine) (either of the           listed)

## Configuration

    The project assumes that you have experience with Python and Basics in ML & NN(Machine Learning and Neural Networks). It assumes you have basic or an appreciation into statical mathematical computation and Data Science and Mathematics. You should have knowledge in Cancer and its behaviour. 

    NB: Make sure you switch to virtual environment to set the project

    The configurations and setup shall be listed below:

### Install and Setup

    Before you get started in Developing your AI models for Semina please make sure you meet the following requirements
    1. Anaconda
    2. Python3.5 or better

    If you don't have Python & Anaconda set up please visit the [Python Website](https://www.python.org/downloads/) & [Anaconda Website](https://www.anaconda.com/distribution/)

    Also check Requirements section for additional requirements needs

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

## Build and Test

    After installation run the following command to build your simple FNA model using Wisconsin Data Set 

    ```bash
    export KERAS_BACKEND=theano #if Using Unix based system
    python classifier_wdbc.py

    ```
    or 

    ```bash
    set KERAS_BACKEND=theano #if using windows based system (powershell)
    python classifier_wdbc.py
    ```
    OUTPUT:

    ```bash
    Using Theano backend.
       842302  M  17.99  10.38   122.8    1001   0.1184   0.2776   0.3001  ...  17.33   184.6    2019   0.1622   0.6656  0.7119  0.2654  0.4601   0.
    1189
    0      842517  M  20.57  17.77  132.90  1326.0  0.08474  0.07864  0.08690  ...  23.41  158.80  1956.0  0.12380  0.18660  0.2416  0.1860  0.2750  0.08902
    1    84300903  M  19.69  21.25  130.00  1203.0  0.10960  0.15990  0.19740  ...  25.53  152.50  1709.0  0.14440  0.42450  0.4504  0.2430  0.3613  0.08758
    2    84348301  M  11.42  20.38   77.58   386.1  0.14250  0.28390  0.24140  ...  26.50   98.87   567.7  0.20980  0.86630  0.6869  0.2575  0.6638  0.17300
    3    84358402  M  20.29  14.34  135.10  1297.0  0.10030  0.13280  0.19800  ...  16.67  152.20  1575.0  0.13740  0.20500  0.4000  0.1625  0.2364  0.07678
    4      843786  M  12.45  15.70   82.57   477.1  0.12780  0.17000  0.15780  ...  23.75  103.40   741.6  0.17910  0.52490  0.5355  0.1741  0.3985  0.12440
    ..        ... ..    ...    ...     ...     ...      ...      ...      ...  ...    ...     ...     ...      ...      ...     ...     ...     ...      ...
    563    926424  M  21.56  22.39  142.00  1479.0  0.11100  0.11590  0.24390  ...  26.40  166.10  2027.0  0.14100  0.21130  0.4107  0.2216  0.2060  0.07115
    564    926682  M  20.13  28.25  131.20  1261.0  0.09780  0.10340  0.14400  ...  38.25  155.00  1731.0  0.11660  0.19220  0.3215  0.1628  0.2572  0.06637
    565    926954  M  16.60  28.08  108.30   858.1  0.08455  0.10230  0.09251  ...  34.12  126.70  1124.0  0.11390  0.30940  0.3403  0.1418  0.2218  0.07820
    566    927241  M  20.60  29.33  140.10  1265.0  0.11780  0.27700  0.35140  ...  39.42  184.60  1821.0  0.16500  0.86810  0.9387  0.2650  0.4087  0.12400
    567     92751  B   7.76  24.54   47.92   181.0  0.05263  0.04362  0.00000  ...  30.37   59.16   268.6  0.08996  0.06444  0.0000  0.0000  0.2871  0.07039

    [568 rows x 32 columns]
    /home/developerprince/anaconda3/envs/semina/lib/python3.7/site-packages/sklearn/preprocessing/label.py:235: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
    y = column_or_1d(y, warn=True)
        17.99  10.38   122.8    1001   0.1184   0.2776   0.3001   0.1471  ...   184.6    2019   0.1622   0.6656  0.7119  0.2654  0.4601   0.1189
    0    20.57  17.77  132.90  1326.0  0.08474  0.07864  0.08690  0.07017  ...  158.80  1956.0  0.12380  0.18660  0.2416  0.1860  0.2750  0.08902
    1    19.69  21.25  130.00  1203.0  0.10960  0.15990  0.19740  0.12790  ...  152.50  1709.0  0.14440  0.42450  0.4504  0.2430  0.3613  0.08758
    2    11.42  20.38   77.58   386.1  0.14250  0.28390  0.24140  0.10520  ...   98.87   567.7  0.20980  0.86630  0.6869  0.2575  0.6638  0.17300
    3    20.29  14.34  135.10  1297.0  0.10030  0.13280  0.19800  0.10430  ...  152.20  1575.0  0.13740  0.20500  0.4000  0.1625  0.2364  0.07678
    4    12.45  15.70   82.57   477.1  0.12780  0.17000  0.15780  0.08089  ...  103.40   741.6  0.17910  0.52490  0.5355  0.1741  0.3985  0.12440
    ..     ...    ...     ...     ...      ...      ...      ...      ...  ...     ...     ...      ...      ...     ...     ...     ...      ...
    563  21.56  22.39  142.00  1479.0  0.11100  0.11590  0.24390  0.13890  ...  166.10  2027.0  0.14100  0.21130  0.4107  0.2216  0.2060  0.07115
    564  20.13  28.25  131.20  1261.0  0.09780  0.10340  0.14400  0.09791  ...  155.00  1731.0  0.11660  0.19220  0.3215  0.1628  0.2572  0.06637
    565  16.60  28.08  108.30   858.1  0.08455  0.10230  0.09251  0.05302  ...  126.70  1124.0  0.11390  0.30940  0.3403  0.1418  0.2218  0.07820
    566  20.60  29.33  140.10  1265.0  0.11780  0.27700  0.35140  0.15200  ...  184.60  1821.0  0.16500  0.86810  0.9387  0.2650  0.4087  0.12400
    567   7.76  24.54   47.92   181.0  0.05263  0.04362  0.00000  0.00000  ...   59.16   268.6  0.08996  0.06444  0.0000  0.0000  0.2871  0.07039

    [568 rows x 30 columns]
        17.99  10.38   122.8    1001   0.1184   0.2776    0.3001    0.1471  ...   184.6    2019   0.1622   0.6656    0.7119   0.2654  0.4601   0.1189
    338  23.510  24.27  155.10  1747.0  0.10690  0.12830  0.230800  0.141000  ...  202.40  2906.0  0.15150  0.26780  0.481900  0.20890  0.2593  0.07738
    427  11.130  16.62   70.47   381.1  0.08151  0.03834  0.013690  0.013700  ...   74.35   421.1  0.10300  0.06219  0.045800  0.04044  0.2383  0.07083
    406  12.850  21.37   82.63   514.5  0.07551  0.08316  0.061260  0.018670  ...   91.63   645.8  0.09402  0.19360  0.183800  0.05601  0.2488  0.08151
    96    9.787  19.94   62.11   294.5  0.10240  0.05301  0.006829  0.007937  ...   68.81   366.1  0.13160  0.09473  0.020490  0.02381  0.1934  0.08988
    490  17.850  13.23  114.60   992.1  0.07838  0.06217  0.044450  0.041780  ...  127.10  1210.0  0.09862  0.09976  0.104800  0.08341  0.1783  0.05871
    ..      ...    ...     ...     ...      ...      ...       ...       ...  ...     ...     ...      ...      ...       ...      ...     ...      ...
    277  13.590  17.84   86.24   572.3  0.07948  0.04052  0.019970  0.012380  ...   98.91   739.1  0.10500  0.07622  0.106000  0.05185  0.2335  0.06263
    9    16.020  23.24  102.70   797.8  0.08206  0.06669  0.032990  0.033230  ...  123.80  1150.0  0.11810  0.15510  0.145900  0.09975  0.2948  0.08452
    359  12.540  18.07   79.42   491.9  0.07436  0.02650  0.001194  0.005449  ...   86.82   585.7  0.09293  0.04327  0.003581  0.01635  0.2233  0.05521
    192  12.340  26.86   81.15   477.4  0.10340  0.13530  0.108500  0.045620  ...  101.70   768.9  0.17850  0.47060  0.442500  0.14590  0.3215  0.12050
    559  14.050  27.15   91.38   600.4  0.09929  0.11260  0.044620  0.043040  ...  100.20   706.7  0.12410  0.22640  0.132600  0.10480  0.2250  0.08321

    [454 rows x 30 columns]
    338    1
    427    0
    406    0
    96     0
    490    0
        ..
    277    0
    9      1
    359    0
    192    1
    559    0
    Length: 454, dtype: int64
    -----------------------------Run Test---------------------------
        17.99  10.38   122.8    1001   0.1184   0.2776   0.3001   0.1471  ...   184.6    2019   0.1622   0.6656   0.7119   0.2654  0.4601   0.1189
    512  14.580  13.66   94.29   658.8  0.09832  0.08918  0.08222  0.04349  ...  108.50   862.0  0.12230  0.19280  0.24920  0.09186  0.2626  0.07048
    457  13.000  25.13   82.61   520.2  0.08369  0.05073  0.01206  0.01762  ...   91.06   628.5  0.12180  0.10930  0.04462  0.05921  0.2306  0.06291
    439  10.970  17.20   71.73   371.5  0.08915  0.11130  0.09457  0.03613  ...   90.14   476.4  0.13910  0.40820  0.47790  0.15550  0.2540  0.09532
    298  10.510  23.09   66.85   334.2  0.10150  0.06797  0.02495  0.01875  ...   70.10   362.7  0.11430  0.08614  0.04158  0.03125  0.2227  0.06777
    37   14.990  25.20   95.54   698.8  0.09387  0.05131  0.02398  0.02899  ...   95.54   698.8  0.09387  0.05131  0.02398  0.02899  0.1565  0.05504
    ..      ...    ...     ...     ...      ...      ...      ...      ...  ...     ...     ...      ...      ...      ...      ...     ...      ...
    213  14.190  23.81   92.87   610.7  0.09463  0.13060  0.11150  0.06462  ...  115.00   811.3  0.15590  0.40590  0.37440  0.17720  0.4724  0.10260
    519   9.295  13.90   59.96   257.8  0.13710  0.12250  0.03332  0.02421  ...   67.84   326.6  0.18500  0.20970  0.09996  0.07262  0.3681  0.08982
    432  18.820  21.97  123.70  1110.0  0.10180  0.13890  0.15940  0.08744  ...  145.30  1603.0  0.13900  0.34630  0.39120  0.17080  0.3007  0.08314
    516  19.890  20.26  130.50  1214.0  0.10370  0.13100  0.14110  0.09431  ...  160.50  1646.0  0.14170  0.33090  0.41850  0.16130  0.2549  0.09136
    500  13.820  24.49   92.33   595.9  0.11620  0.16810  0.13570  0.06759  ...  106.00   788.0  0.17940  0.39660  0.33810  0.15210  0.3651  0.11830

    [114 rows x 30 columns]
    512    0
    457    0
    439    0
    298    0
    37     1
        ..
    213    1
    519    0
    432    1
    516    1
    500    1
    Length: 114, dtype: int64
    /home/developerprince/Documents/Developer-Work/DeveloperPrince/semina/Semina_Medical_ML_WorkShop/FNA/generate_model.py:31: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(10, input_dim=30, activation="relu", kernel_initializer="uniform")`
    model.add(Dense(10, input_dim=x_dim, init='uniform', activation='relu'))
    /home/developerprince/Documents/Developer-Work/DeveloperPrince/semina/Semina_Medical_ML_WorkShop/FNA/generate_model.py:32: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(60, activation="relu", kernel_initializer="uniform")`
    model.add(Dense(60, init='uniform', activation='relu'))
    /home/developerprince/Documents/Developer-Work/DeveloperPrince/semina/Semina_Medical_ML_WorkShop/FNA/generate_model.py:33: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(30, activation="sigmoid", kernel_initializer="uniform")`
    model.add(Dense(30, init='uniform', activation='sigmoid'))
    /home/developerprince/Documents/Developer-Work/DeveloperPrince/semina/Semina_Medical_ML_WorkShop/FNA/generate_model.py:34: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(1, activation="sigmoid", kernel_initializer="uniform")`
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 10)                310       
    _________________________________________________________________
    dense_2 (Dense)              (None, 60)                660       
    _________________________________________________________________
    dense_3 (Dense)              (None, 30)                1830      
    _________________________________________________________________
    dense_4 (Dense)              (None, 1)                 31        
    =================================================================
    Total params: 2,831
    Trainable params: 2,831
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/150
    - 0s - loss: 0.7026 - accuracy: 0.3700
    Epoch 2/150
    - 0s - loss: 0.6889 - accuracy: 0.4449
    Epoch 3/150
    - 0s - loss: 0.6754 - accuracy: 0.8877
    Epoch 4/150
    - 0s - loss: 0.6630 - accuracy: 0.9119
    Epoch 5/150
    - 0s - loss: 0.6515 - accuracy: 0.7930
    Epoch 6/150
    - 0s - loss: 0.6402 - accuracy: 0.7070
    Epoch 7/150
    - 0s - loss: 0.6294 - accuracy: 0.6872
    Epoch 8/150
    - 0s - loss: 0.6180 - accuracy: 0.6916
    Epoch 9/150
    - 0s - loss: 0.6060 - accuracy: 0.7026
    Epoch 10/150
    - 0s - loss: 0.5927 - accuracy: 0.7225
    Epoch 11/150
    - 0s - loss: 0.5794 - accuracy: 0.7489
    Epoch 12/150
    - 0s - loss: 0.5656 - accuracy: 0.8040
    Epoch 13/150
    - 0s - loss: 0.5509 - accuracy: 0.8414
    Epoch 14/150
    - 0s - loss: 0.5372 - accuracy: 0.8612
    Epoch 15/150
    - 0s - loss: 0.5229 - accuracy: 0.8943
    Epoch 16/150
    - 0s - loss: 0.5090 - accuracy: 0.9053
    Epoch 17/150
    - 0s - loss: 0.4981 - accuracy: 0.9053
    Epoch 18/150
    - 0s - loss: 0.4794 - accuracy: 0.9141
    Epoch 19/150
    - 0s - loss: 0.4588 - accuracy: 0.9097
    Epoch 20/150
    - 0s - loss: 0.4239 - accuracy: 0.9141
    Epoch 21/150
    - 0s - loss: 0.3989 - accuracy: 0.9009
    Epoch 22/150
    - 0s - loss: 0.3798 - accuracy: 0.9141
    Epoch 23/150
    - 0s - loss: 0.3568 - accuracy: 0.9141
    Epoch 24/150
    - 0s - loss: 0.3405 - accuracy: 0.9229
    Epoch 25/150
    - 0s - loss: 0.3284 - accuracy: 0.9229
    Epoch 26/150
    - 0s - loss: 0.3141 - accuracy: 0.9273
    Epoch 27/150
    - 0s - loss: 0.3035 - accuracy: 0.9251
    Epoch 28/150
    - 0s - loss: 0.2984 - accuracy: 0.9207
    Epoch 29/150
    - 0s - loss: 0.3000 - accuracy: 0.9141
    Epoch 30/150
    - 0s - loss: 0.2940 - accuracy: 0.9207
    Epoch 31/150
    - 0s - loss: 0.2749 - accuracy: 0.9229
    Epoch 32/150
    - 0s - loss: 0.2708 - accuracy: 0.9229
    Epoch 33/150
    - 0s - loss: 0.2692 - accuracy: 0.9207
    Epoch 34/150
    - 0s - loss: 0.2631 - accuracy: 0.9251
    Epoch 35/150
    - 0s - loss: 0.2658 - accuracy: 0.9185
    Epoch 36/150
    - 0s - loss: 0.2589 - accuracy: 0.9207
    Epoch 37/150
    - 0s - loss: 0.2545 - accuracy: 0.9273
    Epoch 38/150
    - 0s - loss: 0.2467 - accuracy: 0.9251
    Epoch 39/150
    - 0s - loss: 0.2487 - accuracy: 0.9251
    Epoch 40/150
    - 0s - loss: 0.2719 - accuracy: 0.9163
    Epoch 41/150
    - 0s - loss: 0.2518 - accuracy: 0.9229
    Epoch 42/150
    - 0s - loss: 0.2785 - accuracy: 0.9075
    Epoch 43/150
    - 0s - loss: 0.2543 - accuracy: 0.9207
    Epoch 44/150
    - 0s - loss: 0.2410 - accuracy: 0.9185
    Epoch 45/150
    - 0s - loss: 0.2416 - accuracy: 0.9229
    Epoch 46/150
    - 0s - loss: 0.2452 - accuracy: 0.9229
    Epoch 47/150
    - 0s - loss: 0.2346 - accuracy: 0.9273
    Epoch 48/150
    - 0s - loss: 0.2356 - accuracy: 0.9251
    Epoch 49/150
    - 0s - loss: 0.2289 - accuracy: 0.9273
    Epoch 50/150
    - 0s - loss: 0.2309 - accuracy: 0.9317
    Epoch 51/150
    - 0s - loss: 0.2243 - accuracy: 0.9207
    Epoch 52/150
    - 0s - loss: 0.2290 - accuracy: 0.9185
    Epoch 53/150
    - 0s - loss: 0.2246 - accuracy: 0.9251
    Epoch 54/150
    - 0s - loss: 0.2214 - accuracy: 0.9185
    Epoch 55/150
    - 0s - loss: 0.2233 - accuracy: 0.9229
    Epoch 56/150
    - 0s - loss: 0.2223 - accuracy: 0.9295
    Epoch 57/150
    - 0s - loss: 0.2178 - accuracy: 0.9229
    Epoch 58/150
    - 0s - loss: 0.2243 - accuracy: 0.9207
    Epoch 59/150
    - 0s - loss: 0.2173 - accuracy: 0.9295
    Epoch 60/150
    - 0s - loss: 0.2160 - accuracy: 0.9295
    Epoch 61/150
    - 0s - loss: 0.2106 - accuracy: 0.9295
    Epoch 62/150
    - 0s - loss: 0.2175 - accuracy: 0.9163
    Epoch 63/150
    - 0s - loss: 0.2103 - accuracy: 0.9295
    Epoch 64/150
    - 0s - loss: 0.2108 - accuracy: 0.9251
    Epoch 65/150
    - 0s - loss: 0.2104 - accuracy: 0.9273
    Epoch 66/150
    - 0s - loss: 0.2069 - accuracy: 0.9251
    Epoch 67/150
    - 0s - loss: 0.2117 - accuracy: 0.9273
    Epoch 68/150
    - 0s - loss: 0.2043 - accuracy: 0.9339
    Epoch 69/150
    - 0s - loss: 0.2080 - accuracy: 0.9273
    Epoch 70/150
    - 0s - loss: 0.2094 - accuracy: 0.9251
    Epoch 71/150
    - 0s - loss: 0.2250 - accuracy: 0.9251
    Epoch 72/150
    - 0s - loss: 0.1999 - accuracy: 0.9295
    Epoch 73/150
    - 0s - loss: 0.2011 - accuracy: 0.9273
    Epoch 74/150
    - 0s - loss: 0.2027 - accuracy: 0.9273
    Epoch 75/150
    - 0s - loss: 0.1977 - accuracy: 0.9317
    Epoch 76/150
    - 0s - loss: 0.1981 - accuracy: 0.9339
    Epoch 77/150
    - 0s - loss: 0.2063 - accuracy: 0.9251
    Epoch 78/150
    - 0s - loss: 0.2020 - accuracy: 0.9273
    Epoch 79/150
    - 0s - loss: 0.2008 - accuracy: 0.9251
    Epoch 80/150
    - 0s - loss: 0.1950 - accuracy: 0.9361
    Epoch 81/150
    - 0s - loss: 0.1954 - accuracy: 0.9273
    Epoch 82/150
    - 0s - loss: 0.1953 - accuracy: 0.9383
    Epoch 83/150
    - 0s - loss: 0.1918 - accuracy: 0.9339
    Epoch 84/150
    - 0s - loss: 0.1903 - accuracy: 0.9339
    Epoch 85/150
    - 0s - loss: 0.1901 - accuracy: 0.9339
    Epoch 86/150
    - 0s - loss: 0.1920 - accuracy: 0.9251
    Epoch 87/150
    - 0s - loss: 0.1938 - accuracy: 0.9295
    Epoch 88/150
    - 0s - loss: 0.1933 - accuracy: 0.9317
    Epoch 89/150
    - 0s - loss: 0.1929 - accuracy: 0.9251
    Epoch 90/150
    - 0s - loss: 0.1910 - accuracy: 0.9273
    Epoch 91/150
    - 0s - loss: 0.1866 - accuracy: 0.9251
    Epoch 92/150
    - 0s - loss: 0.1993 - accuracy: 0.9295
    Epoch 93/150
    - 0s - loss: 0.1907 - accuracy: 0.9295
    Epoch 94/150
    - 0s - loss: 0.1854 - accuracy: 0.9339
    Epoch 95/150
    - 0s - loss: 0.1849 - accuracy: 0.9317
    Epoch 96/150
    - 0s - loss: 0.1833 - accuracy: 0.9339
    Epoch 97/150
    - 0s - loss: 0.1905 - accuracy: 0.9207
    Epoch 98/150
    - 0s - loss: 0.1826 - accuracy: 0.9295
    Epoch 99/150
    - 0s - loss: 0.1845 - accuracy: 0.9273
    Epoch 100/150
    - 0s - loss: 0.1800 - accuracy: 0.9339
    Epoch 101/150
    - 0s - loss: 0.1811 - accuracy: 0.9339
    Epoch 102/150
    - 0s - loss: 0.1831 - accuracy: 0.9361
    Epoch 103/150
    - 0s - loss: 0.1795 - accuracy: 0.9295
    Epoch 104/150
    - 0s - loss: 0.1801 - accuracy: 0.9273
    Epoch 105/150
    - 0s - loss: 0.1794 - accuracy: 0.9317
    Epoch 106/150
    - 0s - loss: 0.1778 - accuracy: 0.9339
    Epoch 107/150
    - 0s - loss: 0.1807 - accuracy: 0.9295
    Epoch 108/150
    - 0s - loss: 0.1870 - accuracy: 0.9251
    Epoch 109/150
    - 0s - loss: 0.1981 - accuracy: 0.9251
    Epoch 110/150
    - 0s - loss: 0.2034 - accuracy: 0.9185
    Epoch 111/150
    - 0s - loss: 0.2031 - accuracy: 0.9317
    Epoch 112/150
    - 0s - loss: 0.1900 - accuracy: 0.9251
    Epoch 113/150
    - 0s - loss: 0.1767 - accuracy: 0.9317
    Epoch 114/150
    - 0s - loss: 0.1753 - accuracy: 0.9383
    Epoch 115/150
    - 0s - loss: 0.1771 - accuracy: 0.9251
    Epoch 116/150
    - 0s - loss: 0.1766 - accuracy: 0.9295
    Epoch 117/150
    - 0s - loss: 0.1735 - accuracy: 0.9317
    Epoch 118/150
    - 0s - loss: 0.1723 - accuracy: 0.9339
    Epoch 119/150
    - 0s - loss: 0.1761 - accuracy: 0.9251
    Epoch 120/150
    - 0s - loss: 0.1725 - accuracy: 0.9317
    Epoch 121/150
    - 0s - loss: 0.1800 - accuracy: 0.9405
    Epoch 122/150
    - 0s - loss: 0.1655 - accuracy: 0.9361
    Epoch 123/150
    - 0s - loss: 0.1730 - accuracy: 0.9295
    Epoch 124/150
    - 0s - loss: 0.1706 - accuracy: 0.9295
    Epoch 125/150
    - 0s - loss: 0.1659 - accuracy: 0.9383
    Epoch 126/150
    - 0s - loss: 0.1741 - accuracy: 0.9273
    Epoch 127/150
    - 0s - loss: 0.1751 - accuracy: 0.9317
    Epoch 128/150
    - 0s - loss: 0.1760 - accuracy: 0.9295
    Epoch 129/150
    - 0s - loss: 0.1850 - accuracy: 0.9339
    Epoch 130/150
    - 0s - loss: 0.1953 - accuracy: 0.9339
    Epoch 131/150
    - 0s - loss: 0.1787 - accuracy: 0.9207
    Epoch 132/150
    - 0s - loss: 0.1638 - accuracy: 0.9405
    Epoch 133/150
    - 0s - loss: 0.1664 - accuracy: 0.9317
    Epoch 134/150
    - 0s - loss: 0.1741 - accuracy: 0.9273
    Epoch 135/150
    - 0s - loss: 0.1734 - accuracy: 0.9339
    Epoch 136/150
    - 0s - loss: 0.1706 - accuracy: 0.9295
    Epoch 137/150
    - 0s - loss: 0.1901 - accuracy: 0.9251
    Epoch 138/150
    - 0s - loss: 0.1726 - accuracy: 0.9273
    Epoch 139/150
    - 0s - loss: 0.1610 - accuracy: 0.9383
    Epoch 140/150
    - 0s - loss: 0.1677 - accuracy: 0.9339
    Epoch 141/150
    - 0s - loss: 0.1686 - accuracy: 0.9361
    Epoch 142/150
    - 0s - loss: 0.1632 - accuracy: 0.9427
    Epoch 143/150
    - 0s - loss: 0.1596 - accuracy: 0.9405
    Epoch 144/150
    - 0s - loss: 0.1602 - accuracy: 0.9405
    Epoch 145/150
    - 0s - loss: 0.1701 - accuracy: 0.9361
    Epoch 146/150
    - 0s - loss: 0.1591 - accuracy: 0.9383
    Epoch 147/150
    - 0s - loss: 0.1590 - accuracy: 0.9427
    Epoch 148/150
    - 0s - loss: 0.1565 - accuracy: 0.9383
    Epoch 149/150
    - 0s - loss: 0.1584 - accuracy: 0.9449
    Epoch 150/150
    - 0s - loss: 0.1563 - accuracy: 0.9383
    114/114 [==============================] - 0s 16us/step

    Accuracy of your FNA Breast Cancer DNN AI Model is :  92.1052634716034%
    ```

    Your Accuracy may vary from 91% to 93% but it should be approximately 92%
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

 
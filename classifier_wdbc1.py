import FNA.create_dataset as dt 
import FNA.generate_model as gm 
import os

os.getcwd()
os.listdir(os.getcwd())

def main():
    #Initiate Data Set Object 
    create_dataset = dt.CREATE_DATASET()

    #Obtain and Extract Text file or CSV file
    input_data = create_dataset.__read_csv__('input/breast-cancer-wisconsin.data')

    #Print Dataset

    print(input_data)

    #Obtain Training and Testing DataSet
    input_x, features_train, features_test, label_train, label_test = create_dataset.__obtain_data__csv__el__lbencode__('input/breast-cancer-wisconsin.data', number_features=9, number_labels=1)

    print(input_x)

    print(features_train)

    print(label_train)

    print("-----------------------------Run Test---------------------------")

    print(features_test)

    print(label_test)

    #Initiate Generate Model Object
    classifer = gm.GENERATE_MODEL() 

        #Model Generation and Metric Evaluation
    metric1 = classifer.kr_train_DNN_Seq_02(9, features_train, features_test, label_train,  label_test, batch_size=100)

    print(f'\nAccuracy of your FNA Breast Cancer DNN AI Model is : \033[1m \033[92m{metric1}%')

if __name__ == "__main__":
    main()
# VHD-survival

This repository contains scripts for data processing and building deep survival models to predict the survival of patients with VHD (Valvular Heart Disease). The project includes preprocessing of ECG data, training a deep survival model, and making predictions.

## Model - Introduction of Our Code

The provided model works with two types of data files:

- **DAT files**: Raw ECG data, typically of shape `12 x 5000`.
- **H5 files**: Processed data generated from DAT files after preprocessing.

### Preprocessing and Model Workflow

The data preprocessing steps are as follows:

1. **Reading DAT Files**:
    The `read_dat` function is used to read raw ECG data from `.dat` files and preprocess the data for model training.

    ```python
    def read_dat(file_path):
        f = open(file_path, "rb")
        ecg_signal = []
        for i in range(8):
            lead_data = []
            for pos in range(0, 500 * 10):
                b = f.read(2)
                value = struct.unpack("h", b)[0]
                lead_data.append(value)
            ecg_signal.append(np.array(lead_data))
        ecg_signal = np.vstack(ecg_signal) * 0.00244
        ecg_signal = np.transpose(ecg_signal)

        ecg_signal = bandpass_filtering(ecg_signal, 500, 0.5, 100)
        ecg_signal = notch_filtering(ecg_signal, 500, 50)
        ecg_signal = resampling(ecg_signal, 500, 400, method='polyphase')
        ecg_signal = zero_padding(ecg_signal)

        ecg_signal = np.transpose(ecg_signal)

        return ecg_signal
    ```

    - **Bandpass filtering**: Filters the signal to remove noise.
    - **Notch filtering**: Removes powerline interference at 50Hz.
    - **Resampling**: Changes the sampling rate from 500Hz to 400Hz using the polyphase method.
    - **Zero padding**: Pads the signal to the required dimensions.

2. **Preprocessing the Labels**:
    Before training, the labels (survival data) must be included in a CSV file, which should contain a column `fu_days` indicating the follow-up days for each patient. This file will be used for training the model.

    ```python
    main(model=model,
        label='fu_label',
        train_csv_path='dataset/VHD/train_VHD.csv',
        val_csv_path='dataset/VHD/val_VHD.csv')
    ```

### Training the Model

To train the deep survival model, use the following command after setting up your environment and preprocessing the data:

```bash
  python train_survival.py
```
Make sure that your dataset is properly set up in the dataset/VHD/ folder, and the CSV files containing the training and validation data (train_VHD.csv and val_VHD.csv) are correctly formatted.

### Testing the Model

Once the model has been trained, you can use it to make predictions using the following command:

```bash
  python predict.py
```
This will allow you to test the model's performance and predict survival for new input data.

### Dependencies
Install the required dependencies using:
```bash
  pip install -r requirements.txt
```
### Acknowledgements
We would like to extend our sincere gratitude to the creators of the [nnet-survival](https://github.com/MGensheimer/nnet-survival) repository, whose work served as an inspiration for this project and provided valuable insights into survival analysis and deep learning techniques.
# SkinSift - Machine Learning

<p align="justify">
From this project, we developed a model to detect skin types. These detection results are then combined with assessment results using rule-based methods to recommend skincare products tailored to the user's needs and skin condition.</p>

## Model
<p align="justify">
To enhance dataset diversity, we performed image data augmentation using ImageDataGenerator. The dataset is divided into three subsets: training, validation, and test. All images are normalized or rescaled by dividing each pixel value by 255, ensuring the values fall within the range [0,1]. We addressed class imbalance by calculating class weights using compute_class_weight.

We applied transfer learning using the MobileNetV2 model pretrained on ImageNet. Several additional layers were added, including `Conv2D`, `BatchNormalization`, `GlobalAveragePooling2D`, and a Dense output layer with 4 classes using softmax activation.

The optimizer used is Nadam, with a loss function of `sparse_categorical_crossentropy` and accuracy as the metric. The training process employed the early stopping callback, which stops training when validation loss does not improve for 5 consecutive epochs. Additionally, a learning rate scheduler was implemented to reduce the learning rate if the validation loss stagnates for 3 epochs. After training the model, here is what we obtained:</p>

![image](https://github.com/user-attachments/assets/2eecf023-1eb6-46ef-a387-6975fe6c4d14)

![image](https://github.com/user-attachments/assets/a1e09f3d-fe67-4a4f-84f1-564555380f8e)

![image](https://github.com/user-attachments/assets/d2b9fe64-0554-4ce0-8ab8-a00bef378f45)

![image](https://github.com/user-attachments/assets/079e1c57-c19f-4859-a5e2-d17d23045fa8)

![image](https://github.com/user-attachments/assets/996859bd-516e-45e4-9f50-950aaa882f57)

<p align="justify">
After the last epoch, the training accuracy was 92.26%, and the validation accuracy was 84.57%. Meanwhile, the test accuracy reached 93.03%. Although the accuracy is quite high, the graph indicates that the model's performance may be suboptimal, which could be attributed to the relatively small dataset size. Based on the F1-score, we achieved an accuracy of 87%. While it may not be exceedingly high (above 90%), we are satisfied with the model as it performs well in predicting skin types on actual images that users might input. The model is saved in two formats: .h5 and .tflite, and we decided to use .tflite. We have tested both formats and found that using .tflite can be significantly faster. Additionally, .tflite is well-suited for platforms like mobile devices.</p>
</p>

## Scraping
| Library          | Version |
| ---------------- | ------- |
| Beautifulsoup4   | 4.12.3  |
| Selenium         | 4.27.1  |
| Pandas           | 2.2.3   |
| Python           | 3.11.1  |

## Model
| Library      | Version |
| ------------ | ------- |
| Tensorflow   | 2.17.1  |
| Keras        | 3.5.0   |
| Matplotlib   | 3.8.2   |
| NumPy        | 1.26.4  |
| Pandas       | 2.2.2   |
| Scikit-learn | 1.5.2   |
| Seaborn      | 0.13.2  |
| Python       | 3.10.12 |

## Assessment
| Library      | Version |
| ------------ | ------- |
| Tensorflow   | 2.17.1  |
| NumPy        | 1.26.4  |
| Pandas       | 2.2.3   |
| Python       | 3.11.1  |

## OCR
| Library      | Version |
| ------------ | ------- |
| pytesseract  | 0.3.13  |
| Pandas       | 2.2.3   |
| Python       | 3.11.1  |

## Content in the Repository
### Assessment - Rule Based with Model
<p align="justify">
In this folder, you can find a Notebook (.ipynb) that combines rule-based methods (based on dataset labels) with a skin type detection model to provide skincare product recommendations. The Test Image folder contains 4 images that can be used to test the first step of the assessment: skin type detection. To try the assessment, first, input a face image to detect the skin type. Then:

`Index 0` of the response can have the value 'a' for sensitive or 'b' for non-sensitive.

`Index 1` of the response can have the value 'a' for a cleanser product, 'b' for a treatment product, 'c' for a mask product, or 'd' for a moisturizer product.

`Index 2` of the response can have the value 'a' for anti-aging, 'b' for moisturizing, 'c' for brightening, 'd' for refreshing, 'f' for calming, and 'g' for no additional function (only the main function). This option can be combined, such as ['a', 'b'], or it can be a single option, such as ['a'].

`Index 3` of the response can have the value 'a' for yes and 'b' for no.

`Index 4` of the response uses a text input indicating the cosmetic ingredients the user wants to avoid. If there are typos in the input, a Levenshtein check will be performed with a threshold of 1.</p>

### Dataset - Prepo, Cleaning, Visualization
<p align="justify">
This folder contains several other subfolders, such as Clean Data, Photos, Raw Data, and Skin Type:

`Clean Data`: Contains cleaned CSV datasets.

`Photos`: Contains images of skincare products.

`Raw Data`: Contains raw or unprocessed data that has been collected.

`Skin Type`: Contains skin type datasets to train the model (public dataset from kaggle).

We collected approximately 26,000 data entries (before preprocessing) on skincare ingredient compositions by scraping the website: www.paulaschoice.com. The code used for scraping can be found in scraping_ingredient.ipynb. Skincare product data was collected manually without scraping, with a total of 233 products (after cleaning), from various sources, including names and brands from https://femaledaily.com/, descriptions, key ingredients, and product ingredients from https://incidecoder.com/, while product images were sourced from various other websites. Labels for the assessment were assigned manually, and the results can be found in ./Clean Data/product/product_asesmen.csv.

The data preprocessing steps can be found in 4 additional .ipynb files: ingredients.ipynb, product.ipynb, skin_type.ipynb, and translate_ingredients.ipynb. We also visualized the Skin Type dataset to examine the distribution of each class, which can also be accessed in skin_type.ipynb.</p>

### Non Model - Pytesseract
<p align="justify">
It is important to note that the OCR used here is not a custom-built model. Pytesseract is an effective OCR, especially when combined with Levenshtein, which is why it was chosen over other open-source options. Pytesseract, in combination with Levenshtein distance, is used to read and match the detected ingredients with those in the dataset. Several specific rules are applied, such as a threshold of 1 and the use of keywords like "komposisi," "ingredient," and "ingredients." The text following these keywords is matched; if no keyword is found, matching begins from the start of the detected text.</p>

### Skin Type Detection Model
<p align="justify">
In this folder, there is a Notebook (.ipynb) containing the process of model transfer learning, training, and other related tasks. The model is converted into both .h5 and .tflite formats, both of which are accessible within this folder. Lastly, there is a subfolder named Test Data that contains 2 images used for experimentation (not part of the test dataset). These 2 images closely resemble real-life images or those that users might input.</p>

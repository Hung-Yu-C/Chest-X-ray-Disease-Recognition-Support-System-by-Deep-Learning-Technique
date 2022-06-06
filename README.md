# Chest X-ray Disease Recognition Support System by Deep Learning Technique
此專題為實作兩個辨識胸腔 X 光圖的神度學習模型，一個為辨識十四類不同肺部疾病的模型，另一個則為專門辨識新冠肺炎的模型，並將其整合為一個病徵辨識系統，最後將其移植至嵌入式系統上，以方便醫療人員於行動裝置上使用。<br>

此系統是基於 ChestXNet 論文中所使用的胸腔 X 光圖辨識病徵的模型改良而成，使用 ChestX-ray14 資料集訓練出一個辨識 14 種肺部疾病的模型，因為當時 Covid-19 的開放資料還不夠多，因此我們使用了 Data Augmentation 來增加可以使用的資料，再基於針對 14 種肺部疾病的模型進行遷移學習訓練出了能夠辨識 Covid-19 的模型。<br>

接著使用多回合辨識的演算法，對測試資料使用不同的  Data augmentation 來進行疊加組合來強化判斷的準確度。<br>


## 功能
可以輸入胸腔 X 光圖來進行肺炎和其他肺部疾病的判斷。<br>
此系統將會輸出是否有得新冠肺炎，如果判斷為不是新冠肺炎則是會輸出是否有其他的肺部疾病。

## 使用方法
* 於 "image_test" 資料夾中放入想測試的照片，並將照片的資料放入 "output" 資料夾中的 "test.csv" 。<br>

* 執行 "test.py" 。<br>

* 判斷的結果會輸出於 "result" 資料夾中的 "prediction_of_all_patients.csv" 。

## 檔案說明

* "data" 資料夾中存放著此專題所做的報告。 

* "Jetson nano" 資料夾中存放著使用 tensorrt 於 Jetson nano 上加速時使用的程式。

# AI Tennis Analysis

This is a rebuild project from scratch based on the [original work by Abdullah Tarek](https://github.com/abdullahtarek/tennis_analysis), which analyzes tennis players in video to measure key metrics such as player speed, ball shot speed, and detects both players and the tennis ball on the court using YOLO, while extracting court keypoints with CNNs.

## Output Example

This is a snippet from the final output video:

![image](example.gif)

## Dependencies

* numpy
* pandas
* python-dotenv
* requests
* roboflow
* torch
* torchvision
* tqdm
* ultralytics

## Installation

* Clone the project

```bash
git clone https://github.com/philsv/ai_tennis_analysis.git
```

* Install dependencies

```bash
cd ai_tennis_analysis
pip install -r requirements.txt
```

* Download the models from [Google Drive](https://drive.google.com/drive/folders/1SIoKD8Yi2c8qN0vltrzw2QuwSRUT9_HT?usp=sharing)

## Usage

* Run the project

```bash
python main.py
```

This will create the output video and save it in the `outputs` folder.

## Models Used

* YOLO v8 for player detection
* Fine-tuned YOLO-Model for ball detection
* Court Key point extraction using Pytorch

## Datasets

* Tennis ball detection dataset [link](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection)
* Tennis court key points dataset [link](https://drive.usercontent.google.com/download?id=1lhAaeQCmk2y440PmagA0KmIVBIysVMwu&export=download&authuser=0&confirm=t&uuid=3077628e-fc9b-4ef2-8cde-b291040afb30&at=APZUnTU9lSikCSe3NqbxV5MVad5T%3A1708243355040)

## Author

* Abdullah Tarek

## Reference

* [Build an AI/ML Tennis Analysis system with YOLO, PyTorch, and Key Point Extraction](https://www.youtube.com/watch?v=L23oIHZE14w&t=8069s)

## To-Do

* Improve Mini-Court Ball Detection

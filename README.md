This project is a gas classification program based on scikit-learn

# GasClassification
# 📝 Gas Classification Program with Scikit-learn 🌡️

This project is a gas classification program based on scikit-learn, which adheres to the GPL3.0 license. It is designed to classify gases based on input data, which can be edited in the `data.xls` file located in the `data` directory. 


# 🖥️ CPU & GPU System Information
This project is built on a system with the following specifications:
- **CPU**: Intel i7-12700F
- **GPU**: NVIDIA RTX 3060
- **Operating System**: Ubuntu 20.04 64bit
- **Python Version**: Python 3.8.10


## 🛠️ Requirements
- Python 3.5 or higher
- Scikit-learn

# 📖 Project Description

This Github project is a Python-based machine learning project that uses deep learning models to classify images. The project uses the PyTorch framework to train and test these deep learning models. 

Users can conduct supervised classification through independent numbering, and the program will automatically colorize the PCA scatter plot without requiring the user to manually specify colors


supervised classification example:

![image](https://github.com/ToreyGu/GasClassification/assets/77352146/08ce6098-836d-410e-9e70-ed60b5358a46)![image](https://github.com/ToreyGu/GasClassification/assets/77352146/fe82f413-2369-42d1-9848-deda2fdcc02b)![image](https://github.com/ToreyGu/GasClassification/assets/77352146/97d8004c-a2ad-4598-a4f6-3368ab6371f8)


 PCA scatter：
![20%humi](https://github.com/ToreyGu/GasClassification/assets/77352146/561ceb05-f857-46b3-bc86-c914187d7555)


This program can automatically draw the boundary of a model, and the boundary accuracy can be controlled by setting the testing density. Typically, higher accuracy results in a more complex model and longer running time.

setting the testing density(The significance of 500 is that 500 points are scanned for each unit length on the coordinate axis.)


![image](https://github.com/ToreyGu/GasClassification/assets/77352146/301b2ca0-cb80-4ac9-bd84-42095ff6c231)


boundary:
![LDA](https://github.com/ToreyGu/GasClassification/assets/77352146/1e62fe06-18e7-4242-b21f-4fe66392e972)

## 🔧 Installation
1. Clone the repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Edit the `data/data.xls` file with your desired input data.
4. Run the program with `python main.py`.

## 📊 Usage
The gas classification program uses a machine learning algorithm to classify gases based on input data. The input data is edited in the `data.xls` file located in the `data` directory. Once the input data has been edited, the program can be run with the `python main.py` command.

## 📜 License
This project adheres to the GPL3.0 license.

## 👨‍💻 Contributing
Contributions to this project are welcome. Please submit a pull request with any changes or improvements.

## 📧 Contact
If you have any questions or concerns about this project, please contact the project owner at theiiku@foxmail.

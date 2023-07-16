# Writing On Air

Writing On Air is a project that recognizes handwritten text in the air using hand gestures captured through a camera. The project uses MediaPipe for hand landmark detection and a custom OCR system trained from scratch.

## Demo

Check out our demo video to see Writing On Air in action!

![Demo Video](demo.mp4)

The demo showcases real-time testing and includes an animation to demonstrate how the project works.


## Getting Started

To get started, clone this repository to your local machine:

```
git clone https://github.com/ahmed-saad1997/Writing-On-Air.git
```

Then, install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

To run real-time testing on your webcam, navigate to the project directory and execute:

```
python run.py
```

If you want to save the output video, you can add the `--path` option followed by the path where you want to save the video:

```
python run.py --path output.mp4
```

## OCR Training

To train the OCR model from scratch, you can look into the `OCR_training` folder. The OCR system is built using deep learning techniques and trained on a custom dataset of handwritten characters.

## OCR Training Results

Here are some examples of OCR training results:

![OCR Training Results 1](OCR_training/results/1.png)

![OCR Training Results 2](OCR_training/results/2.png)

![OCR Training Results 3](OCR_training/results/3.png)

## Datasets

The datasets used in OCR training include:

- **Synthetic-data:** A synthetic dataset of handwritten characters generated using a script that simulates handwriting. The dataset includes a variety of fonts, styles, and sizes.

- **IAM:** A collection of handwritten text samples collected from forms filled out by writers. The dataset includes a wide range of writing styles and conditions.

## Credits

- The hand landmark detection model was implemented using the Mediapipe pre-trained model.
- The OCR model was trained from scratch using guides from "Modern Computer Vision with PyTorch" and a combination of the "IAM handwriting dataset" and "synthetic-data".

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

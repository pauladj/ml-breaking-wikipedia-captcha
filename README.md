<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="img/logo.jpg" alt="Project logo"></a>
</p>
<h3 align="center">Breaking Wikipedia's Captcha</h3>

<div align="center">
    <img src="https://img.shields.io/badge/python-v3.7.9-blue" />
    <img src="https://img.shields.io/badge/license-MIT-green" />
</div>

---

<p align="center"> Breaking 2020 Wikipedia's captcha using image processing and deep learning techniques. I am <b>not responsible</b> for how this code is used.
    <br> 
</p>

## 👓 Demo

<a href="https://breaking-wikipedia-captcha.herokuapp.com/"><img src="https://img.shields.io/badge/heroku-Open Web App-blue" /></a>

<img src="img/breaking-wikipedia-captcha.gif" />

## 📄 Results

I used the LeNet architecture, first introduced by LeCun et al. in their 1998 paper, *Gradient-Based Learning Applied to Document Recognition*. The authors' motivation behind implementing LeNet was primarly for Optical Character Recognition (OCR). It's a simple model with only two convolutional layers. I got 96% accuracy and 0.15 loss on both train and test set.

<img width="500" src="output/results.png" />



## 🔧 Setup 
```
git clone https://github.com/pauladj/breaking-wikipedia-captcha.git
cd breaking-wikipedia-captcha
pip install -r requirements.txt
```


## 🎈 Usage 
You can use the already downloaded captchas in the folder `downloads` or you can download more images using:

```
python download_images.py -o captcha_image_folder -n num_images_to_download
```

To get the text of the captcha images you just have to execute the next command:

```
python test_model.py -i captcha_image_folder -m output
```

## ⛏️ Built Using 
- [OpenCV](https://opencv.org/) - Computer vision library
- [Keras](https://keras.io/) - Deep learning library
- [Streamlit](https://www.streamlit.io/) - Deploy data apps

## 🎉 Acknowledgements 
- Inspired by the book *Deep Learning for Computer Vision with Python (Starter Bundle)* by Adrian Rosebrock.

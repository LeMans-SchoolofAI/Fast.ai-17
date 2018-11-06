# Giving eyes to blind people

Author - Pierre Cartier

We are teaching to an artificial Tensorflow Neural Network to classify images of crosswalk lights. It can be usefull for blind people helping project.
It's exported to a Tensorflow Lite model to be used on mobile devices like Android or iOS phones or Rasberry and other small devices.

WARNING : It's just an exemple of how to use machine learning on mobile, it is not fully usable, do not use it for blind people.

Prerequisites :
* Last version of tensorflow (containing tensorflow lite) : https://github.com/tensorflow/tensorflow and put it in this directory
* Python 2.7 or 3.6
* Bazel : https://bazel.build/
* Android Studio (or other, or something for iOS but not tried)

## 1. Collecting dataset

Project contains a small dataset for the exemple but you can extend it using *python scrap_google_image.py -s WHAT_I_SEARCH -n HOW_MANY -d WHERE_I_WANT_TO_SAVE*
Make sure to have a clear dataset without bad data.
# Giving eyes to blind people

Author - Pierre Cartier

We are teaching to an artificial Tensorflow Neural Network to classify images of crosswalk lights. It can be usefull for blind people helping project.
It's exported to a Tensorflow Lite model to be used on mobile devices like Android or iOS phones or Rasberry and other small devices.

WARNING : It's just an exemple of how to use machine learning on mobile, it is not fully usable, do not use it for blind people.

I followed this to help me : https://www.tensorflow.org/hub/tutorials/image_retraining

Prerequisites :
* Last version of tensorflow (containing tensorflow lite) : https://github.com/tensorflow/tensorflow and put it in this directory
* Python 2.7 or 3.6
* Bazel : https://bazel.build/
* Android Studio (or other, or something for iOS but not tried)

## 1. Collecting dataset

Project contains a small dataset for the exemple but you can extend it using
*python scrap_google_image.py -s WHAT_I_SEARCH -n HOW_MANY -d WHERE_I_WANT_TO_SAVE*

Make sure to have a clear dataset without bad data.

## 2. Retrain the model

We are using a pretrained model wich can already classify more than 1000 various objects. We retrain it to specialize it for our dataset.
*python retrain.py --image_dir ./data/all/*

The output is in *~/tmp/*

Now, we have *output_graph.pb* and *output_labels.txt*

We can test it with
*python label_image.py --graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt --input_layer=Placeholder --output_layer=final_result --image=MY_IMAGE.jpg*

## 3. Convert the model

Because of mobile low processor or graphic chipset, we need to transform the model graph into a lite version.
We are using TOCO included in Tensorflow but it need to be builded. So build it with bazel.
*bazel build tensorflow/contrib/lite/toco:toco*

Then run toco *./bazel-bin/tensorflow/contrib/lite/toco/toco --input_file="$HOME/tensorflow-master/output_graph.pb" --output_file="$HOME/output_graph.tflite" --input_arrays=Placeholder --output_arrays=final_result*

We get the tflite model graph.

## 4. Put it in your app

For this exemple, I'm using the Tensorflow Lite sample for android (*tensorflow-master/tensorflow/lite/java/demo*)
Sample can contains some gradle sync error, just fix it.

Add the files (.tflite and .txt) into assets folder (*tensorflow-master/tensorflow/lite/java/demo/app/src/main/assets*)
And don't forget to change their call into *ImageClassifierQuantizedMobileNet* class

Build and run.
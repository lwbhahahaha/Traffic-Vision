import tensorflow as tf


converter = tf.lite.TFLiteConverter.from_saved_model("C:/Users/Administrator.WIN-2EPKD7D6018/Desktop/models/research/object_detection/saved_model3/saved_model/")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

tflite_model = converter.convert()
with open('tf2.2.tflite', 'wb') as f:
  f.write(tflite_model)

  #tflite_convert --output_file 'model2.tflite'  --input_shapes=1,300,300,1 --saved_model_dir=C:/Users/Administrator.WIN-2EPKD7D6018/Desktop/models/research/object_detection/saved_model2/saved_model/

#!/usr/bin/env python3
"""
train_tf_recognizer.py

Train a TensorFlow/Keras classifier from images organized as:
  dataset/<person>/*.jpg

Outputs:
  - Saved Keras model directory (SavedModel)
  - Converted TensorFlow Lite model (.tflite)
  - labels.json mapping integer -> person name
  - optional confusion matrix plot

Usage examples:
	python scripts/train_tf_recognizer.py --dataset "C:/Users/jairo/Downloads/AI/dataset" --output_dir scripts/output --epochs 8

Notes:
  - Uses MobileNetV2 as backbone (transfer learning) for a mobile-friendly model.
  - Saves a float32 .tflite by default; enable --quantize_fp16 for smaller size.
"""
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import tensorflow as tf
from tensorflow import keras


# ---------------------- Configuration (edit values here) ----------------------
CONFIG = {
	# Path to dataset root (folders per person). Use forward slashes or raw strings on Windows.
	'dataset': r'C:/Users/jairo/Downloads/AI/data',
	# Where to save model artifacts
	'output_dir': r'scripts/output',
	# Image size (square)
	'img_size': 224,
	'batch_size': 16,
	'epochs': 8,
	'tflite_name': 'model.tflite',
	'quantize_fp16': False,
	# Sampling: if >0, take up to this many images per class for training/validation
	'max_train_per_class': 100,
	'max_val_per_class': 20,
	'sample_seed': 123,
}

# ------------------------------------------------------------------------------


def build_model(input_shape, num_classes, base_trainable=False):
	base = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
	base.trainable = base_trainable
	inputs = keras.Input(shape=input_shape)
	x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
	x = base(x, training=False)
	x = keras.layers.GlobalAveragePooling2D()(x)
	x = keras.layers.Dropout(0.3)(x)
	outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
	model = keras.Model(inputs, outputs)
	return model


def plot_confusion_matrix(cm, class_names, out_path):
	fig, ax = plt.subplots(figsize=(8, 8))
	im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	ax.figure.colorbar(im, ax=ax)
	ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=class_names, yticklabels=class_names, ylabel='True label', xlabel='Predicted label')
	plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


def main(args):
	dataset_dir = args.dataset
	out_dir = args.output_dir
	img_size = (args.img_size, args.img_size)
	batch = args.batch_size
	epochs = args.epochs

	# If sampling per-class is requested, create a temporary sampled dataset
	sampled_root = None
	if getattr(args, 'max_train_per_class', 0) and getattr(args, 'max_val_per_class', 0):
		import shutil
		from random import Random
		sampled_root = os.path.join(out_dir, 'sampled_dataset')
		# clear
		if os.path.exists(sampled_root):
			shutil.rmtree(sampled_root)
		train_out = os.path.join(sampled_root, 'train')
		val_out = os.path.join(sampled_root, 'val')
		os.makedirs(train_out, exist_ok=True)
		os.makedirs(val_out, exist_ok=True)

		rng = Random(getattr(args, 'sample_seed', 123))
		for person in sorted(os.listdir(dataset_dir)):
			person_dir = os.path.join(dataset_dir, person)
			if not os.path.isdir(person_dir):
				continue
			files = [f for f in sorted(os.listdir(person_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
			rng.shuffle(files)
			take_train = min(len(files), int(args.max_train_per_class))
			take_val = min(max(0, len(files) - take_train), int(args.max_val_per_class))

			# ensure folders
			os.makedirs(os.path.join(train_out, person), exist_ok=True)
			os.makedirs(os.path.join(val_out, person), exist_ok=True)

			# copy files
			for i, fname in enumerate(files):
				src = os.path.join(person_dir, fname)
				if i < take_train:
					dst = os.path.join(train_out, person, fname)
				elif i < take_train + take_val:
					dst = os.path.join(val_out, person, fname)
				else:
					continue
				shutil.copy2(src, dst)

		print('Created sampled dataset at', sampled_root)
		dataset_for_train = train_out
		dataset_for_val = val_out
	else:
		dataset_for_train = dataset_dir
		dataset_for_val = None

	Path(out_dir).mkdir(parents=True, exist_ok=True)

	print('Loading dataset from', dataset_for_train)
	if dataset_for_val:
		train_ds = tf.keras.preprocessing.image_dataset_from_directory(dataset_for_train, image_size=img_size, batch_size=batch)
		val_ds = tf.keras.preprocessing.image_dataset_from_directory(dataset_for_val, image_size=img_size, batch_size=batch)
	else:
		train_ds = tf.keras.preprocessing.image_dataset_from_directory(dataset_dir, validation_split=0.2, subset='training', seed=123, image_size=img_size, batch_size=batch)
		val_ds = tf.keras.preprocessing.image_dataset_from_directory(dataset_dir, validation_split=0.2, subset='validation', seed=123, image_size=img_size, batch_size=batch)

	class_names = train_ds.class_names
	num_classes = len(class_names)
	print(f'Found classes: {class_names} (num_classes={num_classes})')

	AUTOTUNE = tf.data.AUTOTUNE
	train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
	val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

	input_shape = (img_size[0], img_size[1], 3)
	model = build_model(input_shape, num_classes, base_trainable=False)

	model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	print(model.summary())

	history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

	# Evaluate
	loss, acc = model.evaluate(val_ds)
	print(f'Validation loss={loss:.4f}, acc={acc:.4f}')

	# Save SavedModel (handle Keras 3 api differences)
	saved_model_dir = os.path.join(out_dir, 'saved_model')
	try:
		# Keras 3 introduced `model.export()` for SavedModel export; prefer it when available
		if hasattr(model, 'export'):
			model.export(saved_model_dir)
			print('Exported model using model.export to', saved_model_dir)
		else:
			# older Keras/TensorFlow versions support model.save(dir) for SavedModel
			model.save(saved_model_dir)
			print('Saved Keras model to', saved_model_dir)
	except Exception as e:
		# Fall back to low-level tf.saved_model.save which works across versions
		print('Could not save with model.save/export (reason:', e, '). Falling back to tf.saved_model.save(...)')
		tf.saved_model.save(model, saved_model_dir)
		print('Saved model with tf.saved_model.save to', saved_model_dir)

	# Save labels
	labels_path = os.path.join(out_dir, 'labels.json')
	id2label = {i: name for i, name in enumerate(class_names)}
	with open(labels_path, 'w', encoding='utf-8') as fh:
		json.dump(id2label, fh, ensure_ascii=False, indent=2)
	print('Saved labels to', labels_path)

	# Confusion matrix
	try:
		from sklearn.metrics import confusion_matrix
		import numpy as np
		# Collect true and preds over validation set
		y_true = []
		y_pred = []
		for images, labels in val_ds:
			preds = model.predict(images)
			preds_ids = np.argmax(preds, axis=1)
			y_true.extend(labels.numpy().tolist())
			y_pred.extend(preds_ids.tolist())
		cm = confusion_matrix(y_true, y_pred)
		cm_path = os.path.join(out_dir, 'confusion_matrix.png')
		plot_confusion_matrix(cm, class_names, cm_path)
		print('Saved confusion matrix to', cm_path)
	except Exception as e:
		print('Could not compute confusion matrix (skipping). Reason:', e)

	# Convert to TFLite
	tflite_path = os.path.join(out_dir, args.tflite_name)
	converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
	if args.quantize_fp16:
		converter.optimizations = [tf.lite.Optimize.DEFAULT]
		converter.target_spec.supported_types = [tf.float16]
		tflite_model = converter.convert()
		with open(tflite_path, 'wb') as f:
			f.write(tflite_model)
		print('Saved FP16 quantized TFLite to', tflite_path)
	else:
		tflite_model = converter.convert()
		with open(tflite_path, 'wb') as f:
			f.write(tflite_model)
		print('Saved TFLite to', tflite_path)


if __name__ == '__main__':
	# All runtime variables are configured in the CONFIG dict near the top of this file.
	# Edit CONFIG to change dataset path, epochs, image size, etc.
	class Cfg:
		pass

	args = Cfg()
	args.dataset = CONFIG['dataset']
	args.output_dir = CONFIG['output_dir']
	args.img_size = CONFIG['img_size']
	args.batch_size = CONFIG['batch_size']
	args.epochs = CONFIG['epochs']
	args.tflite_name = CONFIG['tflite_name']
	args.quantize_fp16 = CONFIG['quantize_fp16']

	main(args)


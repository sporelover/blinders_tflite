import 'dart:typed_data';

import 'package:tflite_flutter/tflite_flutter.dart';

class TfliteService {
  static List<double> run(Interpreter interpreter, Uint8List rgbBytes, {int height = 224, int width = 224, int channels = 3}) {
    // Inspect input tensor to build the correct input object
    final inputTensors = interpreter.getInputTensors();
    if (inputTensors.isEmpty) {
      throw StateError('Interpreter has no input tensors');
    }
    final inT = inputTensors[0];
    final shape = inT.shape; // e.g., [1,224,224,3]
    final dt = inT.type.toString().toLowerCase();

    final int b = shape.isNotEmpty ? shape[0] : 1;
    final int h = shape.length > 1 ? shape[1] : height;
    final int w = shape.length > 2 ? shape[2] : width;
    final int ch = shape.length > 3 ? shape[3] : channels;

    Object input;
    if (dt.contains('float')) {
      final List<List<List<List<double>>>> batch = List.generate(b, (_) => List.generate(h, (_) => List.generate(w, (_) => List.filled(ch, 0.0))));
      int idx = 0;
      for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
          for (int c = 0; c < ch; c++) {
            batch[0][i][j][c] = rgbBytes[idx].toDouble();
            idx++;
          }
        }
      }
      input = batch;
    } else {
      final List<List<List<List<int>>>> batch = List.generate(b, (_) => List.generate(h, (_) => List.generate(w, (_) => List.filled(ch, 0))));
      int idx = 0;
      for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
          for (int c = 0; c < ch; c++) {
            batch[0][i][j][c] = rgbBytes[idx];
            idx++;
          }
        }
      }
      input = batch;
    }

    // Prepare output container based on output tensor shape
    final outputTensors = interpreter.getOutputTensors();
    Object output;
    if (outputTensors.isNotEmpty) {
      final outShape = outputTensors[0].shape; // e.g. [1,4]
      if (outShape.length == 2) {
        final int ob = outShape[0];
        final int oc = outShape[1];
        output = List.generate(ob, (_) => List.filled(oc, 0.0));
      } else {
        final int total = outShape.fold(1, (p, e) => p * e);
        output = List.filled(total, 0.0);
      }
    } else {
      output = List.filled(4, 0.0);
    }

    interpreter.run(input, output);

    // Normalize to flat List<double>
    if (output is List) {
      if (output.isNotEmpty && output[0] is List) {
        final first = output[0] as List;
        return first.map<double>((e) => (e is num) ? e.toDouble() : double.parse(e.toString())).toList();
      }
  return output.map<double>((e) => (e is num) ? e.toDouble() : double.parse(e.toString())).toList();
    }
    if (output is Float32List) {
      return output.map((e) => e.toDouble()).toList();
    }
    return <double>[];
  }
}

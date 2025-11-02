// no-op
import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

import 'image_utils.dart';

class MlKitService {
  final FaceDetector _detector;

  MlKitService()
      : _detector = FaceDetector(
          options: FaceDetectorOptions(
            performanceMode: FaceDetectorMode.fast,
            enableContours: false,
            enableLandmarks: false,
            enableClassification: false,
          ),
        );

  Future<RectLike?> detectLargestFace(CameraImage image) async {
    // Convert to NV21 and feed to ML Kit (more broadly supported)
    final bytes = ImageUtils.yuv420ToNv21(image);
    final metadata = InputImageMetadata(
      size: ui.Size(image.width.toDouble(), image.height.toDouble()),
      rotation: InputImageRotation.rotation0deg,
      format: InputImageFormat.nv21,
      bytesPerRow: image.planes[0].bytesPerRow,
    );

    final inputImage = InputImage.fromBytes(bytes: bytes, metadata: metadata);
    final faces = await _detector.processImage(inputImage);
    if (faces.isEmpty) return null;

    // Pick largest face by area
    faces.sort((a, b) => (b.boundingBox.width * b.boundingBox.height).compareTo(a.boundingBox.width * a.boundingBox.height));
    final bb = faces.first.boundingBox;
    return RectLike(left: bb.left.toDouble(), top: bb.top.toDouble(), right: bb.right.toDouble(), bottom: bb.bottom.toDouble());
  }

  Future<void> close() => _detector.close();
}

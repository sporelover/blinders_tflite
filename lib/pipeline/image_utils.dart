import 'dart:typed_data';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';

class ImageUtils {
  // Convert CameraImage in YUV_420_888 to NV21 byte layout (Y + interleaved VU)
  static Uint8List yuv420ToNv21(CameraImage image) {
    final int width = image.width;
    final int height = image.height;
    final int ySize = width * height;
    final int uvSize = ySize ~/ 2;
    final out = Uint8List(ySize + uvSize);

    final Plane y = image.planes[0];
    final Plane u = image.planes[1];
    final Plane v = image.planes[2];

    // Copy Y plane
    int outIndex = 0;
    final int yRowStride = y.bytesPerRow;
    final int yPixelStride = y.bytesPerPixel ?? 1;
    for (int i = 0; i < height; i++) {
      int yIndex = i * yRowStride;
      for (int j = 0; j < width; j++) {
        out[outIndex++] = y.bytes[yIndex];
        yIndex += yPixelStride;
      }
    }

    // Interleave V and U into NV21 (VU) at half resolution
    final int uRowStride = u.bytesPerRow;
    final int vRowStride = v.bytesPerRow;
    final int uPixelStride = u.bytesPerPixel ?? 1;
    final int vPixelStride = v.bytesPerPixel ?? 1;
    for (int i = 0; i < height ~/ 2; i++) {
      int uIndex = i * uRowStride;
      int vIndex = i * vRowStride;
      for (int j = 0; j < width ~/ 2; j++) {
        final int vByte = v.bytes[vIndex];
        final int uByte = u.bytes[uIndex];
        out[outIndex++] = vByte;
        out[outIndex++] = uByte;
        uIndex += uPixelStride;
        vIndex += vPixelStride;
      }
    }

    return out;
  }

  // Crop from Y plane and resize to outW x outH, then expand to RGB by repeating Y.
  // This avoids a full YUV->RGB conversion while producing a 3-channel input.
  static Uint8List cropResizeYToRGB(
    CameraImage image,
    RectLike crop,
    int outW,
    int outH,
  ) {
    final yPlane = image.planes[0];
    final Uint8List yBytes = yPlane.bytes;
    final int yStride = yPlane.bytesPerRow;
    final int yPixelStride = yPlane.bytesPerPixel ?? 1;
    final int srcW = image.width;
    final int srcH = image.height;

    // Clamp crop to image bounds
    final int x0 = crop.left.clamp(0, srcW - 1).toInt();
    final int y0 = crop.top.clamp(0, srcH - 1).toInt();
    final int x1 = crop.right.clamp(1, srcW).toInt();
    final int y1 = crop.bottom.clamp(1, srcH).toInt();
    final int cw = math.max(1, x1 - x0);
    final int ch = math.max(1, y1 - y0);

    final out = Uint8List(outW * outH * 3);

    for (int oy = 0; oy < outH; oy++) {
      final double sy = y0 + (oy + 0.5) * ch / outH - 0.5;
      int syi = sy.round().clamp(y0, y1 - 1);
      for (int ox = 0; ox < outW; ox++) {
        final double sx = x0 + (ox + 0.5) * cw / outW - 0.5;
        int sxi = sx.round().clamp(x0, x1 - 1);
        final int yIndex = syi * yStride + sxi * yPixelStride;
        final int yVal = yBytes[yIndex];
        final int dstIndex = (oy * outW + ox) * 3;
        out[dstIndex] = yVal; // R
        out[dstIndex + 1] = yVal; // G
        out[dstIndex + 2] = yVal; // B
      }
    }

    return out;
  }
}

// Minimal Rect-like abstraction to avoid importing dart:ui Rect into isolates if needed.
class RectLike {
  final double left;
  final double top;
  final double right;
  final double bottom;
  const RectLike({required this.left, required this.top, required this.right, required this.bottom});

  double get width => right - left;
  double get height => bottom - top;
}

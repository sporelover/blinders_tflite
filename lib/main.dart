import 'dart:async';
// dart:io not required in this scaffold
import 'dart:convert';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
// Note: `tflite_flutter_helper` removed from pubspec to avoid version conflicts.
// The preprocessing pipeline is implemented manually in this scaffold.
import 'pipeline/mlkit_service.dart';
import 'pipeline/image_utils.dart';
import 'pipeline/tflite_service.dart';

// IMPORTANT: copy `scripts/output/model.tflite` -> flutter app `assets/model.tflite`
//             copy `scripts/output/labels.json` -> flutter app `assets/labels.json`
// Then run: flutter pub get && flutter run

// This is a minimal, well-documented starter. It detects the largest face in the camera
// preview using ML Kit, crops and resizes the face to the model input size and runs the
// TFLite interpreter to get top-K labels. The displayed prediction is updated every
// UPDATE_INTERVAL seconds to avoid rapid flicker (default 10s), mirroring the desktop behavior.

const int INPUT_SIZE = 224;
const double UPDATE_INTERVAL = 10.0; // seconds (unused)
const int TOP_K = 4; // show top-4 labels in overlay
const int UPDATE_FRAMES = 5; // update percentages every N frames
const double WARMUP_SECONDS = 30.0; // wait this long after first face seen before showing predictions
const double FACE_LOST_RESET_SECONDS = 2.0; // if no face for this long, reset warmup

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  final firstCamera = cameras.isNotEmpty ? cameras.first : null;
  runApp(MyApp(camera: firstCamera));
}

class MyApp extends StatelessWidget {
  final CameraDescription? camera;
  const MyApp({super.key, this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter TFLite Face Recognizer',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: camera == null ? const Scaffold(body: Center(child: Text('No camera'))) : HomePage(camera: camera!),
    );
  }
}

class HomePage extends StatefulWidget {
  final CameraDescription camera;
  const HomePage({super.key, required this.camera});

  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  late CameraController _controller;
  Future<void>? _initializeControllerFuture;
  late Interpreter _interpreter;
  late List<String> _labels;
  late MlKitService _mlKit;
  // Camera management for switching
  List<CameraDescription>? _cameras;
  int _cameraIndex = 0;
  late CameraDescription _currentCamera;

  // Raw input buffer (kept minimal; processed just-in-time)

  List<double> _displayedProbs = [];
  // Face tracking and overlay
  RectLike? _lastFaceBox;
  Size? _lastImageSize;

  bool _isProcessing = false;
  int _frameCounter = 0;
  // Warmup gating before showing probabilities
  DateTime? _faceFirstSeenAt;
  DateTime? _lastFaceSeenAt;
  bool _readyToShow = false;

  @override
  void initState() {
    super.initState();
    // Initialize cameras list and controller. Start with the camera passed from main(),
    // but allow switching between available cameras via the UI button.
    availableCameras().then((cams) {
      _cameras = cams;
      // find index of provided camera, fallback to 0
      final idx = _cameras!.indexWhere((c) => c.lensDirection == widget.camera.lensDirection && c.sensorOrientation == widget.camera.sensorOrientation);
      _cameraIndex = idx >= 0 ? idx : 0;
      _currentCamera = _cameras![_cameraIndex];
      _initializeCameraController(_currentCamera);
    }).catchError((e) {
      debugPrint('availableCameras() failed: $e');
      // fallback: initialize with widget.camera if available
      _currentCamera = widget.camera;
      _initializeCameraController(_currentCamera);
    });

  // ML Kit service
  _mlKit = MlKitService();

    _loadModelAndLabels();
  }

  Future<void> _initializeCameraController(CameraDescription camera) async {
    try {
      // Dispose previous controller if present
      try {
        await _controller.stopImageStream();
      } catch (_) {}
      try {
        await _controller.dispose();
      } catch (_) {}
    } catch (_) {}

    _controller = CameraController(camera, ResolutionPreset.medium, enableAudio: false);
    _initializeControllerFuture = _controller.initialize();
    try {
      await _initializeControllerFuture;
      await _controller.startImageStream(_processCameraImage);
      setState(() {});
    } catch (e) {
      debugPrint('Camera init failed: $e');
    }
  }

  Future<void> _switchCamera() async {
    if (_cameras == null || _cameras!.length < 2) return;
    _cameraIndex = (_cameraIndex + 1) % _cameras!.length;
    _currentCamera = _cameras![_cameraIndex];
    await _initializeCameraController(_currentCamera);
  }

  Future<void> _loadModelAndLabels() async {
    // Load TFLite interpreter from assets
    try {
      // Prefer the asset path declared in pubspec.yaml (assets/model.tflite).
      // Try that first and fall back to 'model.tflite' only if needed.
      bool loaded = false;
      try {
        final b = await DefaultAssetBundle.of(context).load('assets/model.tflite');
        debugPrint('Asset assets/model.tflite found, size=${b.lengthInBytes}');
        _interpreter = await Interpreter.fromAsset('assets/model.tflite');
        loaded = true;
      } catch (_) {
        // ignore and try alternative
      }
      if (!loaded) {
        try {
          final b2 = await DefaultAssetBundle.of(context).load('model.tflite');
          debugPrint('Asset model.tflite found, size=${b2.lengthInBytes}');
          _interpreter = await Interpreter.fromAsset('model.tflite');
          loaded = true;
        } catch (_) {
          // not found
        }
      }
      if (!loaded) {
        throw Exception('model.tflite not found in assets. Please copy model.tflite into flutter_face_app/assets and declare it in pubspec.yaml');
      }
        // Allocate tensors and log input/output details for diagnostics
        try {
          _interpreter.allocateTensors();
          final inputs = _interpreter.getInputTensors();
          final outputs = _interpreter.getOutputTensors();
          if (inputs.isNotEmpty) debugPrint('Interpreter input[0] shape=${inputs[0].shape}, type=${inputs[0].type}');
          if (outputs.isNotEmpty) debugPrint('Interpreter output[0] shape=${outputs[0].shape}, type=${outputs[0].type}');
        } catch (e) {
          debugPrint('allocateTensors() failed: $e');
        }
    } catch (e) {
      debugPrint('Failed to load interpreter: $e');
      rethrow;
    }

    // Load labels
    final labelsData = await DefaultAssetBundle.of(context).loadString('assets/labels.json');
    try {
      final decoded = json.decode(labelsData);
      if (decoded is List) {
        _labels = decoded.map((e) => e.toString()).toList();
      } else if (decoded is Map) {
        // labels.json might be a mapping like {"0":"name","1":"other"}
        final m = Map<String, dynamic>.from(decoded);
        // build a list ordered by numeric keys if possible
        final entries = <MapEntry<int, String>>[];
        m.forEach((k, v) {
          final idx = int.tryParse(k);
          if (idx != null) entries.add(MapEntry(idx, v.toString()));
        });
        entries.sort((a, b) => a.key.compareTo(b.key));
        _labels = entries.map((e) => e.value).toList();
        // fallback: if map keys are non-numeric, use values order
        if (_labels.isEmpty) {
          _labels = m.values.map((v) => v.toString()).toList();
        }
      } else {
        _labels = [];
      }
    } catch (e) {
      debugPrint('Failed parsing labels.json: $e');
      _labels = [];
    }
    // NOTE: Accepts either an array of labels or a mapping {"0":"name"}.

    // Initialize displayed probs with zeros
    // Try to infer number of classes from interpreter output tensor shape
    final outputTensors = _interpreter.getOutputTensors();
    int numClasses = 0;
    if (outputTensors.isNotEmpty) {
      final shape = outputTensors[0].shape;
      if (shape.isNotEmpty) numClasses = shape.last;
    }
    _displayedProbs = List<double>.filled(numClasses, 0.0);
  }

  @override
  void dispose() {
    _controller.dispose();
    _mlKit.close();
    _interpreter.close();
    super.dispose();
  }

  // Placeholder converter removed; using ImageUtils.cropResizeYToRGB in pipeline.

  // Process frames from camera: detect largest face, crop+resize (Y->RGB), run TFLite
  Future<void> _processCameraImage(CameraImage cameraImage) async {
    if (_isProcessing) return; // drop frames if busy
    _isProcessing = true;

    try {
      // Skip until model/labels loaded
      if (_labels.isEmpty) {
        _isProcessing = false;
        return;
      }

      // Detect largest face (coordinates in CameraImage space; rotation kept at 0 in service)
      final faceBox = await _mlKit.detectLargestFace(cameraImage);
      if (faceBox == null) {
        // No face: consider resetting warmup if face has been gone for a while
        final now = DateTime.now();
        _lastFaceSeenAt ??= now;
        if (_lastFaceSeenAt != null && now.difference(_lastFaceSeenAt!).inMilliseconds / 1000.0 > FACE_LOST_RESET_SECONDS) {
          _faceFirstSeenAt = null;
          _lastFaceSeenAt = null;
          _readyToShow = false;
          // Optionally clear overlay box
          setState(() {
            _lastFaceBox = null;
          });
        }
        _isProcessing = false;
        return;
      }
      // Update the face box on-screen immediately so the square follows the face smoothly
      setState(() {
        _lastFaceBox = faceBox;
        _lastImageSize = Size(cameraImage.width.toDouble(), cameraImage.height.toDouble());
      });

      // Warmup timer: start counting when a face is first seen, reset if face lost for a while
      final now = DateTime.now();
      _lastFaceSeenAt = now;
      _faceFirstSeenAt ??= now;
      final warmupElapsed = now.difference(_faceFirstSeenAt!).inMilliseconds / 1000.0;
      if (!_readyToShow && warmupElapsed >= WARMUP_SECONDS) {
        _readyToShow = true;
      }

      // Crop from Y plane, resize to model input, expand to RGB (Y replicated)
  final processed = ImageUtils.cropResizeYToRGB(cameraImage, faceBox, INPUT_SIZE, INPUT_SIZE);

      // Run interpreter (expects a typed List/TypedData). Build proper shaped input
      try {
        // Prepare and run inference via service
        final probs = TfliteService.run(_interpreter, processed, height: INPUT_SIZE, width: INPUT_SIZE, channels: 3);
        // Update displayed probabilities every UPDATE_FRAMES frames (also during warmup)
        _frameCounter++;
        if (_frameCounter % UPDATE_FRAMES == 0) {
          setState(() {
            if (_displayedProbs.length != probs.length) {
              _displayedProbs = List<double>.from(probs);
            } else {
              for (int i = 0; i < probs.length; i++) {
                _displayedProbs[i] = probs[i];
              }
            }
          });
        }
      } catch (e) {
        debugPrint('Interpreter run failed: $e');
      }
    } catch (e) {
      debugPrint('Processing failed: $e');
    } finally {
      _isProcessing = false;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('TFLite Face Recognizer')),
      body: (_initializeControllerFuture == null)
          ? const Center(child: CircularProgressIndicator())
          : FutureBuilder<void>(
              future: _initializeControllerFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.done) {
                  return LayoutBuilder(
                    builder: (context, constraints) {
                      return Stack(
                        children: [
                          CameraPreview(_controller),
                          Positioned.fill(
                            child: CustomPaint(
                              painter: FaceBoxPainter(
                                _lastFaceBox,
                                _lastImageSize,
                                mirror: (_cameras != null && _cameras!.isNotEmpty)
                                    ? _currentCamera.lensDirection == CameraLensDirection.front
                                    : false,
                              ),
                            ),
                          ),
                          Positioned(
                            bottom: 16,
                            left: 16,
                            right: 16,
                            child: _buildOverlay(),
                          )
                        ],
                      );
                    },
                  );
                } else {
                  return const Center(child: CircularProgressIndicator());
                }
              },
            ),
      floatingActionButton: FloatingActionButton(
        onPressed: (_cameras == null || _cameras!.length < 2) ? null : _switchCamera,
        child: const Icon(Icons.cameraswitch),
        tooltip: 'Cambiar cámara',
      ),
    );
  }

  Widget _buildOverlay() {
    final List<Widget> children = [];
    // Warmup hint (show alongside probabilities)
    if (!_readyToShow) {
      String warmupMsg = 'Buscando rostro...';
      if (_faceFirstSeenAt != null) {
        final now = DateTime.now();
        final elapsed = now.difference(_faceFirstSeenAt!).inMilliseconds / 1000.0;
        final remaining = (WARMUP_SECONDS - elapsed).clamp(0, WARMUP_SECONDS);
        final secs = remaining.toStringAsFixed(0);
        warmupMsg = 'Preparando... ${secs}s';
      }
      children.add(Text(warmupMsg, style: const TextStyle(color: Colors.white70, fontSize: 12)));
    }

    // Top-K probabilities
    if (_displayedProbs.isNotEmpty) {
      final entries = _displayedProbs.asMap().entries.toList()
        ..sort((a, b) => b.value.compareTo(a.value));
      final k = entries.length < TOP_K ? entries.length : TOP_K;
      for (int i = 0; i < k; i++) {
        final idx = entries[i].key;
        final val = entries[i].value;
        final name = (idx < _labels.length) ? _labels[idx] : 'id:$idx';
        final pct = (val * 100.0).toStringAsFixed(1);
        children.add(Text('$name: $pct%', style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w600)));
      }
    } else {
      children.add(const Text('Buscando...', style: TextStyle(color: Colors.white)));
    }

    return Container(
      padding: const EdgeInsets.all(10),
      color: Colors.black54,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: children,
      ),
    );
  }
}

class FaceBoxPainter extends CustomPainter {
  final RectLike? box;
  final Size? imageSize;
  final bool mirror;
  FaceBoxPainter(this.box, this.imageSize, {this.mirror = false});

  @override
  void paint(Canvas canvas, Size size) {
    if (box == null || imageSize == null) return;
    final sx = size.width / imageSize!.width;
    final sy = size.height / imageSize!.height;
    double left = box!.left * sx;
    final double top = box!.top * sy;
    final double width = (box!.right - box!.left) * sx;
    final double height = (box!.bottom - box!.top) * sy;
    if (mirror) {
      // Mirror horizontally for front camera to match the preview
      left = size.width - (left + width);
    }
    final rect = Rect.fromLTWH(left, top, width, height);
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..color = Colors.greenAccent
      ..strokeWidth = 3;
    canvas.drawRect(rect, paint);
  }

  @override
  bool shouldRepaint(covariant FaceBoxPainter oldDelegate) {
    return oldDelegate.box != box || oldDelegate.imageSize != imageSize;
  }
}

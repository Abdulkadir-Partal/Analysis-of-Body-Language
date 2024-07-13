#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <mediapipe/framework/port/opencv_highgui_inc.h>
#include <mediapipe/framework/port/opencv_imgproc_inc.h>
#include <mediapipe/framework/port/status.h>
#include <mediapipe/framework/port/statusor.h>
#include <mediapipe/framework/deps/file_path.h>
#include <mediapipe/util/resource_util.h>
#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/image_frame_opencv.h>
#include <mediapipe/framework/formats/landmark.pb.h>
#include <mediapipe/calculators/util/detections_to_rects_calculator.pb.h>
#include <mediapipe/calculators/util/detections_to_rects_calculator_options.pb.h>
#include <mediapipe/calculators/util/rect_transformation_calculator.pb.h>
#include <mediapipe/calculators/util/rect_transformation_calculator_options.pb.h>
#include <mediapipe/calculators/tflite/tflite_inference_calculator.pb.h>
#include <mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.pb.h>
#include <mediapipe/calculators/tflite/tflite_tensors_to_floats_calculator.pb.h>
#include <mediapipe/framework/formats/classification.pb.h>
#include <mediapipe/framework/formats/classification_data.pb.h>
#include <mediapipe/framework/formats/location_data.pb.h>
#include <mediapipe/framework/formats/classification.pb.h>
#include <mediapipe/framework/port/file_helpers.h>
#include <mediapipe/framework/port/status.h>
#include <mediapipe/framework/port/ret_check.h>
#include <mediapipe/framework/port/parse_text_proto.h>
#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/image_frame_opencv.h>
#include <mediapipe/framework/port/opencv_imgcodecs_inc.h>
#include <mediapipe/framework/port/opencv_highgui_inc.h>
#include <mediapipe/util/resource_util.h>
#include <mediapipe/framework/port/ffmpeg.h>
#include <mediapipe/framework/formats/video_stream_header.pb.h>
#include <mediapipe/framework/formats/matrix_data.pb.h>

namespace mp = mediapipe;

mp::Status RunMPPGraph(cv::VideoCapture& capture) {
  // MediaPipe graph setup
  mp::CalculatorGraphConfig graph_config = ParseTextProtoOrDie<mp::CalculatorGraphConfig>(R"(
      input_stream: "input_video"
      output_stream: "output_video"
      node {
        calculator: "HolisticGraphCalculator"
        input_stream: "input_video"
        output_stream: "output_video"
      }
  )");

  mp::CalculatorGraph graph;
  RET_CHECK_OK(graph.Initialize(graph_config));

  // Capture loop
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                   graph.AddOutputStreamPoller("output_video"));
  RET_CHECK_OK(graph.StartRun({}));

  cv::Mat frame, output_frame;
  while (capture.isOpened()) {
    capture >> frame;
    if (frame.empty()) break;

    mp::StatusOr<bool> result = graph.AddPacketToInputStream(
        "input_video", mp::Adopt(new cv::Mat(frame.clone())));
    if (!result.ok()) {
      break;
    }

    // Get output frames from graph
    if (!poller.Next(&output_frame)) break;

    // Display output frames
    cv::imshow("Output", output_frame);
    if (cv::waitKey(5) == 'q') break;
  }

  // Close up
  RET_CHECK_OK(graph.CloseInputStream("input_video"));
  RET_CHECK_OK(graph.WaitUntilDone());
  return mp::OkStatus();
}

int main() {
  cv::VideoCapture capture(0);
  if (!capture.isOpened()) {
    std::cerr << "Cannot open camera" << std::endl;
    return -1;
  }

  mp::Status status = RunMPPGraph(capture);
  if (!status.ok()) {
    std::cerr << "MediaPipe failed: " << status.message() << std::endl;
    return -1;
  }

  return 0;
}

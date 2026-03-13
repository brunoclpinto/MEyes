import AVFoundation
import CoreImage
import CoreVideo
import Photos
internal import UIKit

final class CIImageRecorder {
  
  enum RecorderError: Error {
    case alreadyStarted
    case notStarted
    case failedToAddInput
    case missingPixelBufferPool
    case failedToCreatePixelBuffer
    case writerFailed(underlying: Error?)
    case invalidImageSize(expected: CGSize, got: CGSize)
    case noFramesAppended
    
    case photoPermissionDenied(status: PHAuthorizationStatus)
    case photoSaveFailed(underlying: Error?)
  }
  
  let fps: Int32
  let fileType: AVFileType
  let codec: AVVideoCodecType
  let frameSize: CGSize
  
  var outputDirectory: URL
  private(set) var currentOutputURL: URL?
  
  private let queue = DispatchQueue(label: "CIImageRecorder.queue")
  private let ciContext = CIContext()
  
  private var writer: AVAssetWriter?
  private var input: AVAssetWriterInput?
  private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
  
  private var finished = false
  private var didAppendAnyFrame = false
  private var startTime: CFAbsoluteTime = 0
  
  init(
    size: CGSize,
    fps: Int32 = 30,
    fileType: AVFileType = .mov,
    codec: AVVideoCodecType = .h264,
    outputDirectory: URL = FileManager.default.temporaryDirectory
  ) {
    self.frameSize = size
    self.fps = fps
    self.fileType = fileType
    self.codec = codec
    self.outputDirectory = outputDirectory
  }
  
  func start() {
    queue.async {
      guard self.writer == nil else { return }
      
      let ts = Self.timestampString(from: Date())
      let baseName = "MEyesVideo\(ts)"
      let url = Self.makeUniqueURL(in: self.outputDirectory, baseName: baseName, ext: "mov")
      self.currentOutputURL = url
      
      try? FileManager.default.removeItem(at: url)
      
      guard let w = try? AVAssetWriter(outputURL: url, fileType: self.fileType) else {
        return
      }
      
      let settings: [String: Any] = [
        AVVideoCodecKey: self.codec,
        AVVideoWidthKey: Int(self.frameSize.width),
        AVVideoHeightKey: Int(self.frameSize.height)
      ]
      
      let i = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
      i.expectsMediaDataInRealTime = true
      
      guard w.canAdd(i) else { return }
      w.add(i)
      
      let pixelAttrs: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
        kCVPixelBufferWidthKey as String: Int(self.frameSize.width),
        kCVPixelBufferHeightKey as String: Int(self.frameSize.height),
        kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
        kCVPixelBufferCGImageCompatibilityKey as String: true
      ]
      
      let a = AVAssetWriterInputPixelBufferAdaptor(
        assetWriterInput: i,
        sourcePixelBufferAttributes: pixelAttrs
      )
      
      w.startWriting()
      w.startSession(atSourceTime: .zero)
      
      self.writer = w
      self.input = i
      self.adaptor = a
      
      self.finished = false
      self.didAppendAnyFrame = false
      self.startTime = 0
    }
  }
  
  func append(
    _ frame: CIImage,
    timing: BusApproachTracker.TimingInfo?,
    currentFPS: Double,
    buses: String
  ) {
    queue.async {
      let image = self.overlayFrame(
        frame,
        timing: timing,
        currentFPS: currentFPS,
        buses: buses
      )
      let now = CFAbsoluteTimeGetCurrent()
      
      // Use wall-clock elapsed time as the presentation timestamp
      // so the video plays back at real-time speed.
      if self.startTime == 0 { self.startTime = now }
      let elapsed = now - self.startTime
      let t = CMTime(seconds: elapsed, preferredTimescale: 600)
      self.append(image, at: t)
    }
  }
  
  func append(_ image: CIImage, at time: CMTime) {
    queue.async {
      guard let _ = self.writer, let i = self.input, let a = self.adaptor else {
        return
      }
      guard !self.finished else { return }
      
      guard i.isReadyForMoreMediaData else {
        return
      }
      
      guard let pool = a.pixelBufferPool else {
        return
      }
      
      var outputBufferOpt: CVPixelBuffer?
      let status = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &outputBufferOpt)
      guard status == kCVReturnSuccess, let outputBuffer = outputBufferOpt else {
        return
      }
      
      self.ciContext.render(
        image,
        to: outputBuffer,
        bounds: image.extent,
        colorSpace: CGColorSpaceCreateDeviceRGB()
      )
      
      let ok = a.append(outputBuffer, withPresentationTime: time)
      if !ok {
        return
      }
      
      self.didAppendAnyFrame = true
    }
  }
  
  func stopAndSaveToPhotos(completion: @escaping (Result<Void, Error>) -> Void) {
    queue.async {
      guard let w = self.writer,
            let i = self.input,
            let url = self.currentOutputURL else {
        DispatchQueue.main.async {
          completion(.failure(RecorderError.notStarted))
        }
        return
      }
      
      guard !self.finished else {
        self.saveVideoToPhotos(url, completion: completion)
        return
      }
      
      self.finished = true
      i.markAsFinished()
      
      w.finishWriting {
        self.queue.async {
          self.writer = nil
          self.input = nil
          self.adaptor = nil
        }
        
        if let err = w.error {
          DispatchQueue.main.async {
            completion(.failure(RecorderError.writerFailed(underlying: err)))
          }
          return
        }
        
        if !self.didAppendAnyFrame {
          DispatchQueue.main.async {
            completion(.failure(RecorderError.noFramesAppended))
          }
          return
        }
        
        self.saveVideoToPhotos(url, completion: completion)
      }
    }
  }
  
  private func saveVideoToPhotos(_ url: URL, completion: @escaping (Result<Void, Error>) -> Void) {
    requestAddOnlyPhotoAuth { status in
      guard status == .authorized || status == .limited else {
        DispatchQueue.main.async {
          completion(.failure(RecorderError.photoPermissionDenied(status: status)))
        }
        return
      }
      
      PHPhotoLibrary.shared().performChanges({
        PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
      }, completionHandler: { success, error in
        DispatchQueue.main.async {
          if success {
            completion(.success(()))
          } else {
            completion(.failure(RecorderError.photoSaveFailed(underlying: error)))
          }
        }
      })
    }
  }
  
  private func requestAddOnlyPhotoAuth(_ done: @escaping (PHAuthorizationStatus) -> Void) {
    if #available(iOS 14, *) {
      let status = PHPhotoLibrary.authorizationStatus(for: .addOnly)
      if status == .notDetermined {
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { done($0) }
      } else {
        done(status)
      }
    } else {
      let status = PHPhotoLibrary.authorizationStatus()
      if status == .notDetermined {
        PHPhotoLibrary.requestAuthorization { done($0) }
      } else {
        done(status)
      }
    }
  }
  
  private static func timestampString(from date: Date) -> String {
    let f = DateFormatter()
    f.locale = Locale(identifier: "en_US_POSIX")
    f.dateFormat = "yyyyMMddHHmmss"
    return f.string(from: date)
  }
  
  private static func makeUniqueURL(in dir: URL, baseName: String, ext: String) -> URL {
    let fm = FileManager.default
    var candidate = dir.appendingPathComponent("\(baseName).\(ext)")
    var n = 1
    while fm.fileExists(atPath: candidate.path) {
      candidate = dir.appendingPathComponent("\(baseName)_\(n).\(ext)")
      n += 1
    }
    return candidate
  }
  
  private func overlayFrame(
    _ frame: CIImage,
    timing: BusApproachTracker.TimingInfo?,
    currentFPS: Double,
    buses: String
  ) -> CIImage {
    let extent = frame.extent
    let w = extent.width
    let h = extent.height
    guard w > 0, h > 0 else { return frame }

    var lines: [String] = []
    lines.append(String(format: "FPS: %.1f", currentFPS))

    if let timing {
      for (name, ms) in timing.flowTimings {
        lines.append(String(format: "%@: %.1f ms", name, ms))
      }
      lines.append(String(format: "workflow: %.1f ms", timing.totalMs))
    }
    lines.append("\n\(buses)")

    let maxOverlayWidth: CGFloat = 640
    let fontSize: CGFloat = 32
    let lineHeight = fontSize * 1.3
    let margin: CGFloat = 8
    let textBlockHeight = lineHeight * CGFloat(lines.count) + margin * 2
    let textBlockWidth = min(maxOverlayWidth, w)

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue

    guard let ctx = CGContext(
      data: nil, width: Int(w), height: Int(h),
      bitsPerComponent: 8, bytesPerRow: 0,
      space: colorSpace, bitmapInfo: bitmapInfo
    ) else { return frame }

    ctx.clear(CGRect(origin: .zero, size: CGSize(width: w, height: h)))

    // Semi-transparent black background for text
    ctx.setFillColor(red: 0, green: 0, blue: 0, alpha: 0.6)
    ctx.fill(CGRect(x: 0, y: 0, width: textBlockWidth, height: textBlockHeight))

    let attrs: [NSAttributedString.Key: Any] = [
      .font: UIFont.monospacedSystemFont(ofSize: fontSize, weight: .bold),
      .foregroundColor: UIColor.green
    ]

    // Draw text (flip coordinates for UIKit text drawing)
    UIGraphicsPushContext(ctx)
    ctx.saveGState()
    ctx.textMatrix = .identity
    ctx.translateBy(x: 0, y: h)
    ctx.scaleBy(x: 1, y: -1)

    let startY = h - textBlockHeight + margin
    for (i, line) in lines.enumerated() {
      let y = startY + CGFloat(i) * lineHeight
      let str = NSAttributedString(string: line, attributes: attrs)
      str.draw(at: CGPoint(x: margin, y: y))
    }

    ctx.restoreGState()
    UIGraphicsPopContext()

    guard let overlayCG = ctx.makeImage() else { return frame }
    let overlayCI = CIImage(cgImage: overlayCG)
    return overlayCI.composited(over: frame)
  }
}

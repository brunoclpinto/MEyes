import AVFoundation
import CoreImage
import CoreVideo
import Photos

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
    private var lastPresentationTime: CFAbsoluteTime = 0

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

    func start() throws {
        try queue.sync {
            guard writer == nil else { throw RecorderError.alreadyStarted }

            let ts = Self.timestampString(from: Date())
            let baseName = "MEyesVideo\(ts)"
            let url = Self.makeUniqueURL(in: outputDirectory, baseName: baseName, ext: "mov")
            currentOutputURL = url

            try? FileManager.default.removeItem(at: url)

            let w = try AVAssetWriter(outputURL: url, fileType: fileType)

            let settings: [String: Any] = [
                AVVideoCodecKey: codec,
                AVVideoWidthKey: Int(frameSize.width),
                AVVideoHeightKey: Int(frameSize.height)
            ]

            let i = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
            i.expectsMediaDataInRealTime = true

            guard w.canAdd(i) else { throw RecorderError.failedToAddInput }
            w.add(i)

            let pixelAttrs: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
                kCVPixelBufferWidthKey as String: Int(frameSize.width),
                kCVPixelBufferHeightKey as String: Int(frameSize.height),
                kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
                kCVPixelBufferCGImageCompatibilityKey as String: true
            ]

            let a = AVAssetWriterInputPixelBufferAdaptor(
                assetWriterInput: i,
                sourcePixelBufferAttributes: pixelAttrs
            )

            w.startWriting()
            w.startSession(atSourceTime: .zero)

            writer = w
            input = i
            adaptor = a

            finished = false
            didAppendAnyFrame = false
            startTime = 0
            lastPresentationTime = 0
        }
    }

    func append(_ image: CIImage) throws {
        let now = CFAbsoluteTimeGetCurrent()

        // Throttle: skip frames arriving faster than the target fps
        let minInterval = 1.0 / CFAbsoluteTime(fps)
        if lastPresentationTime > 0, (now - lastPresentationTime) < minInterval { return }
        lastPresentationTime = now

        // Use wall-clock elapsed time as the presentation timestamp
        // so the video plays back at real-time speed.
        if startTime == 0 { startTime = now }
        let elapsed = now - startTime
        let t = CMTime(seconds: elapsed, preferredTimescale: 600)
        try append(image, at: t)
    }

    func append(_ image: CIImage, at time: CMTime) throws {
        try queue.sync {
            guard let w = writer, let i = input, let a = adaptor else {
                throw RecorderError.notStarted
            }
            guard !finished else { return }

            guard i.isReadyForMoreMediaData else {
                return
            }

            guard let pool = a.pixelBufferPool else {
                throw RecorderError.missingPixelBufferPool
            }

            var outputBufferOpt: CVPixelBuffer?
            let status = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &outputBufferOpt)
            guard status == kCVReturnSuccess, let outputBuffer = outputBufferOpt else {
                throw RecorderError.failedToCreatePixelBuffer
            }

            ciContext.render(
                image,
                to: outputBuffer,
                bounds: image.extent,
                colorSpace: CGColorSpaceCreateDeviceRGB()
            )

            let ok = a.append(outputBuffer, withPresentationTime: time)
            if !ok {
                throw RecorderError.writerFailed(underlying: w.error)
            }

            didAppendAnyFrame = true
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
}

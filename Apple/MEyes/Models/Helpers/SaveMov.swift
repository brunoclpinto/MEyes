import AVFoundation
import CoreGraphics
import Photos

final class CGImageVideoRecorder {

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

    // MARK: Public config
    let fps: Int32
    let fileType: AVFileType
    let codec: AVVideoCodecType
    let frameSize: CGSize

    /// Where the .mov is written before importing into Photos.
    var outputDirectory: URL

    /// The URL created on the most recent start().
    private(set) var currentOutputURL: URL?

    // MARK: Private
    private let queue = DispatchQueue(label: "CGImageVideoRecorder.queue")

    private var writer: AVAssetWriter?
    private var input: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?

    private var finished = false
    private var frameCount: Int64 = 0
    private var didAppendAnyFrame = false

    init(size: CGSize,
         fps: Int32 = 30,
         fileType: AVFileType = .mov,
         codec: AVVideoCodecType = .h264,
         outputDirectory: URL = FileManager.default.temporaryDirectory) {
        self.frameSize = size
        self.fps = fps
        self.fileType = fileType
        self.codec = codec
        self.outputDirectory = outputDirectory
    }

    /// Creates a new file like MEyesVideo20260219142530.mov (timestamp at start time).
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
                AVVideoHeightKey: Int(frameSize.height),
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

            let a = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: i,
                                                        sourcePixelBufferAttributes: pixelAttrs)

            // Start immediately at t=0 (simplifies restarts; avoids “never started” edge cases)
            w.startWriting()
            w.startSession(atSourceTime: .zero)

            writer = w
            input = i
            adaptor = a

            finished = false
            frameCount = 0
            didAppendAnyFrame = false
        }
    }

    /// Append a frame at fixed FPS timeline (0, 1/fps, 2/fps, ...)
    func append(_ image: CGImage) throws {
        let t = CMTime(value: frameCount, timescale: fps)
        try append(image, at: t)
        frameCount += 1
    }

    /// Append a frame at an explicit time.
    func append(_ image: CGImage, at time: CMTime) throws {
        try queue.sync {
            guard let w = writer, let i = input, let a = adaptor else {
                throw RecorderError.notStarted
            }
            guard !finished else { return }

            let gotSize = CGSize(width: image.width, height: image.height)
            if gotSize != frameSize {
                throw RecorderError.invalidImageSize(expected: frameSize, got: gotSize)
            }

            guard i.isReadyForMoreMediaData else {
                // Drop frame for debug recording; if you prefer “no drops”, tell me and I’ll change this.
                return
            }

            guard let pool = a.pixelBufferPool else {
                throw RecorderError.missingPixelBufferPool
            }

            var pbOut: CVPixelBuffer?
            let status = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pbOut)
            guard status == kCVReturnSuccess, let pb = pbOut else {
                throw RecorderError.failedToCreatePixelBuffer
            }

            CVPixelBufferLockBaseAddress(pb, [])
            defer { CVPixelBufferUnlockBaseAddress(pb, []) }

            guard let base = CVPixelBufferGetBaseAddress(pb) else {
                throw RecorderError.failedToCreatePixelBuffer
            }

            let bytesPerRow = CVPixelBufferGetBytesPerRow(pb)
            let colorSpace = CGColorSpaceCreateDeviceRGB()

            let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue
                | CGImageAlphaInfo.premultipliedFirst.rawValue

            guard let ctx = CGContext(
                data: base,
                width: Int(frameSize.width),
                height: Int(frameSize.height),
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: bitmapInfo
            ) else {
                throw RecorderError.failedToCreatePixelBuffer
            }

            ctx.clear(CGRect(origin: .zero, size: frameSize))

            // If your output is vertically flipped, uncomment:
            // ctx.translateBy(x: 0, y: frameSize.height)
            // ctx.scaleBy(x: 1, y: -1)

            ctx.draw(image, in: CGRect(origin: .zero, size: frameSize))

            let ok = a.append(pb, withPresentationTime: time)
            if !ok {
                throw RecorderError.writerFailed(underlying: w.error)
            }

            didAppendAnyFrame = true
        }
    }

    /// Stops writing and saves the resulting video into Photos.
    func stopAndSaveToPhotos(completion: @escaping (Result<Void, Error>) -> Void) {
        queue.async {
            guard let w = self.writer,
                  let i = self.input,
                  let url = self.currentOutputURL
            else {
                DispatchQueue.main.async { completion(.failure(RecorderError.notStarted)) }
                return
            }

            guard !self.finished else {
                self.saveVideoToPhotos(url, completion: completion)
                return
            }

            self.finished = true
            i.markAsFinished()

            w.finishWriting {
                // Release writer state ASAP after finishing
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

    // MARK: Photos saving

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

    // MARK: Filename helpers

    /// "yyyyMMddHHmmss" (year, month, day, hour, minute, seconds)
    private static func timestampString(from date: Date) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyyMMddHHmmss"
        return f.string(from: date)
    }

    /// If MEyesVideo<ts>.mov exists, returns MEyesVideo<ts>_1.mov, etc.
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
